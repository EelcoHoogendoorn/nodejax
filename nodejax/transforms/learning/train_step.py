"""First-class gradient optimization as cyclic Nodes (`optimizer`, `train_step`, `trained`).

`train_step` composes a model, a scalar loss, and an optimizer into an ordinary
cyclic `Node` whose live carry holds both the model's own state and the
optimizer's evolving parameter and momentum trees, while its outer `param` tree
retains starting weights (and any learned optimizer hyperparameters such as
`learned_sgd`). Because the resulting trainer satisfies the standard `Node`
contract, outer transforms (`scan`, `trained`, `ensemble`, `finetune`, `ttt`)
compose over gradient updates without bespoke training-loop runners.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp

from nodejax.core.ambient import node
from nodejax.core.authoring import Leaf
from nodejax.core.binding import Aux, split_aux
from nodejax.core.composite import Composite
from nodejax.core.contract import CallField, CallForm, Contract
from nodejax.core.node import BaseNode, Node, _is_node
from nodejax.core.pnode import PNode
from nodejax.core.rng import MaybeKeyStream
from nodejax.core.types import LossFn, PyTree
from nodejax.core.wrapper import Wrapper
from nodejax.struct import Struct
from nodejax.transforms.iteration.scan import (
    _fresh_step_state, _internalized_form, _sequence_parameterize,
    _split_state_fields,
)
from nodejax.transforms.transform import bind, scan_steps, transform

if TYPE_CHECKING:
    import optax


@node
def optimizer(tx: optax.GradientTransformation, name: str = 'optimizer') -> PNode:
    """Adapts an `optax.GradientTransformation` into a cyclic optimizer `PNode`.

    Priming from an initial parameter tree builds `Struct(params=..., opt=...)`
    in state, and each `apply` call consumes a gradient tree to advance both
    `params` and `opt`.
    """
    import optax

    def init(input):
        return Struct(params=input, opt=tx.init(input))

    def apply(state, input):
        updates, opt = tx.update(input, state.opt, state.params)
        return Struct(params=optax.apply_updates(state.params, updates), opt=opt), None

    return Leaf(apply, init=init, name=name)


@node
def learned_sgd(lr0: float, name: str = 'learned_sgd') -> PNode:
    """Constructs a gradient-descent optimizer Node with a learnable per-element step size initialized to `lr0`."""
    def param(node):
        return jax.tree.map(lambda x: jnp.full_like(x, lr0), node.input)

    def init(input):
        return Struct(params=input)

    def apply(param, state, input):
        params = jax.tree.map(lambda w, g, lr: w - lr * g,
                              state.params, input, param)
        return Struct(params=params), None

    return Leaf(apply, param=param, init=init, name=name)


def _as_optimizer(tx: PyTree) -> Node:
    return tx.node if _is_node(tx) else optimizer(tx).node


@transform(preserves='param')
def opt_reinit(inner: Node, boundary: str) -> Node:
    """Resets `inner`'s `opt` state slot at `boundary` while keeping adapted `params` intact."""
    def action(carried, init, decided):
        return decided.replace(opt=init.opt)

    return Wrapper(inner=inner)(
        name=f'opt_reinit({inner.name})',
        boundary={boundary: action})


def _opt_param(opt: Contract, weights: PyTree, rng) -> PyTree:
    if not opt.parametric:
        return ()
    return opt.for_input(weights).param(Struct(), rng)


def _path_name(path) -> str:
    return '.'.join(
        str(getattr(key, 'name', getattr(key, 'key', getattr(key, 'idx', key))))
        for key in path)


def _selection(trainable: str | Callable | None) -> Callable:
    if trainable is None:
        return lambda path: True
    if isinstance(trainable, str):
        return lambda path: trainable in path.split('.')
    return trainable


def _is_missing(leaf) -> bool:
    return leaf is None


def _without_missing(tree: PyTree) -> PyTree:
    if tree is None:
        return None
    if issubclass(type(tree), Struct):
        kept = {field: _without_missing(value) for field, value in tree.__items__}
        kept = {field: value for field, value in kept.items() if value is not None}
        return type(tree)(**kept) if kept else None
    return tree


def _partition(tree: PyTree, select: Callable) -> tuple[PyTree, PyTree]:
    """Splits `tree` by dotted leaf path into `(selected_subtree, rest_with_none_placeholders)`."""
    chosen = jax.tree_util.tree_map_with_path(
        lambda path, leaf: leaf if select(_path_name(path)) else None, tree)
    rest = jax.tree_util.tree_map_with_path(
        lambda path, leaf: None if select(_path_name(path)) else leaf, tree)
    return _without_missing(chosen), rest


def _merge(chosen: PyTree, rest: PyTree) -> PyTree:
    """Fills `None` placeholders in `rest` with matching path-addressed leaves from `chosen`."""
    leaves = {
        _path_name(path): leaf
        for path, leaf in jax.tree_util.tree_flatten_with_path(chosen)[0]}
    return jax.tree_util.tree_map_with_path(
        lambda path, leaf: leaves[_path_name(path)] if leaf is None else leaf,
        rest, is_leaf=_is_missing)


def _require_train_step(step: PyTree, who: str) -> Node:
    if not _is_node(step):
        raise TypeError(
            f'{who} consumes the product of train_step: '
            'a cyclic parametric node holding an objective and optimizer. Got '
            f'{step!r}; spell it {who}(train_step(model, loss_fn, '
            'optimizer))')
    node = step.node
    structured = ('opt' in node.members and 'objective' in node.members)
    objective = node.members.objective if structured else None
    if not (node.cyclic and node.parametric and structured
            and 'model' in objective.members and 'loss' in objective.members):
        raise TypeError(
            f'{who} consumes the product of train_step: '
            'a cyclic parametric node holding an objective and optimizer. Got '
            f'{step!r}; spell it {who}(train_step(model, loss_fn, '
            'optimizer))')
    return node


def _require_trained_step(step: PyTree) -> Node:
    if not _is_node(step):
        raise TypeError(
            'trained consumes a cyclic parametric Node with a trained() '
            f'method; got {step!r}')
    node = step.node
    if not (node.cyclic and node.parametric and 'trained' in node._def.methods):
        raise TypeError(
            'trained consumes a cyclic parametric Node with a trained() '
            f'method; got {step!r}')
    return node


def _as_loss(loss_fn: LossFn | BaseNode) -> BaseNode:
    if _is_node(loss_fn):
        return loss_fn
    if not callable(loss_fn):
        raise TypeError(
            'train_step loss must be a Node or callable; '
            f'got {loss_fn!r}')
    return Leaf(loss_fn)


def _loss_call(loss: Contract) -> tuple[str, tuple[str, ...], bool]:
    form = loss._apply_form
    if form.open:
        raise TypeError(
            f'train_step loss {loss.name!r} must declare a closed call form')
    fields = loss.apply_fields
    if not fields:
        raise TypeError(
            f'train_step loss {loss.name!r} must consume the model output')
    output_field = fields[0]
    side_fields = tuple(field for field in fields[1:] if field != 'aux')
    takes_aux = 'aux' in fields[1:]
    return output_field, side_fields, takes_aux


def _training_fields(model: Contract, loss: Contract) -> tuple[str, ...]:
    if model._apply_form.open:
        raise TypeError(
            f'train_step model {model.name!r} must declare a closed call form')
    _, side_fields, _ = _loss_call(loss)
    return model.apply_fields + tuple(
        field for field in side_fields if field not in model.apply_fields)


def _training_input_spec(model: Contract, loss: Contract) -> Struct | None:
    _, side_fields, _ = _loss_call(loss)
    values = {}
    if model.apply_fields:
        model_spec = model.input_spec
        if model_spec is None:
            return None
        values.update(dict(model_spec.__items__))
    loss_spec = loss.input_spec
    for field in side_fields:
        if field in values:
            continue
        if loss_spec is None:
            return None
        values[field] = loss_spec[field]
    return Struct(**values)


def _loss_needs_construction_call(loss: BaseNode) -> bool:
    param = loss._def.calls.param
    init = loss._def.calls.init
    return bool(
        (not loss.bound and param is not None and param.reads_def)
        or (not loss.state_bound and init is not None
            and (init.reads_def or init.requires_input))
    )


def _select(bundle: Struct, fields: tuple[str, ...]) -> Struct:
    return Struct(**{field: bundle[field] for field in fields})


def _loss_takes_aux(loss_fn: LossFn) -> bool:
    try:
        signature = inspect.signature(loss_fn)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f'train_step loss {loss_fn!r} must have an inspectable signature'
        ) from error
    takes_aux = 'aux' in signature.parameters
    keyword = {'aux': object()} if takes_aux else {}
    try:
        signature.bind(object(), object(), **keyword)
    except TypeError as error:
        raise TypeError(
            f'train_step loss {signature} must accept (output, target) or '
            "accept those arguments plus an 'aux' keyword") from error
    return takes_aux


def map_loss_target(
    loss_fn: LossFn,
    target_fn: Callable,
) -> LossFn:
    """Transforms the second positional argument of `loss_fn` with `target_fn` while preserving `aux` opt-in."""
    if _loss_takes_aux(loss_fn):
        def loss(output, data, *, aux):
            return loss_fn(output, target_fn(data), aux=aux)
    else:
        def loss(output, data):
            return loss_fn(output, target_fn(data))
    return loss


@node
def _objective(model: BaseNode, loss: BaseNode) -> BaseNode:
    """Composes `model` and `loss` into a scalar objective while preserving `model` output in `Aux`."""
    if not _is_node(model) or not _is_node(loss):
        raise TypeError('an objective requires model and loss Nodes')

    model_fields = model.contract.apply_fields
    loss_output, _, _ = _loss_call(loss.contract)

    call_loss_during_construction = _loss_needs_construction_call(loss)

    def routed(self, input):
        output = self.model(bundle=_select(input, model_fields))
        if not call_loss_during_construction:
            return output
        loss_input = {
            field: (output if field == loss_output else
                    Aux() if field == 'aux' else input[field])
            for field in loss.contract.apply_fields
        }
        return self.loss(bundle=Struct(**loss_input))

    # Authored Composite owns parameterization, initialization, and shape walks;
    # runtime apply is overridden below so `model` Aux forwards to `loss` when requested.
    scaffold = Composite(model=model, loss=loss)(
        routed, name='objective')

    def apply_fn(contract, param, state, input, rng):
        current_model = contract.members.model
        current_loss = contract.members.loss
        values = contract.intake(input)
        model_input = _select(values, model_fields)

        model_state, model_result = current_model.apply(
            getattr(param, 'model') if current_model.parametric else (),
            getattr(state, 'model') if current_model.cyclic else (),
            current_model.feed_bundle(model_input),
            rng.child(current_model.apply_takes_rng),
        )
        output, model_aux = split_aux(model_result)
        available_aux = Aux() if model_aux is None else model_aux
        loss_input = Struct(**{
            field: (output if field == loss_output else
                    available_aux if field == 'aux' else values[field])
            for field in current_loss.apply_fields
        })
        loss_state, loss_result = current_loss.apply(
            getattr(param, 'loss') if current_loss.parametric else (),
            getattr(state, 'loss') if current_loss.cyclic else (),
            current_loss.feed_bundle(loss_input),
            rng.child(current_loss.apply_takes_rng),
        )
        scalar, loss_aux = split_aux(loss_result)
        states = {
            **({'model': model_state} if current_model.cyclic else {}),
            **({'loss': loss_state} if current_loss.cyclic else {}),
        }
        retained = Aux(
            output=output,
            model=Aux() if model_aux is None else model_aux,
            loss=Aux() if loss_aux is None else loss_aux,
        )
        return Struct(**states) if states else (), (scalar, retained)

    calls = scaffold.contract._roles(apply=apply_fn)
    result = scaffold._with_definition(scaffold._def.copy(calls=calls))

    parameters_ready = all(
        not member.parametric or value.bound
        for member, value in (
            (model.contract, model), (loss.contract, loss)))
    if not parameters_ready:
        return result
    parameters = Struct(**{
        **({'model': model.param} if model.parametric else {}),
        **({'loss': loss.param} if loss.parametric else {}),
    })
    result = result.bind(parameters if parameters else ())

    state_ready = model.state_bound and (
        not loss.cyclic or loss.state_bound)
    if not state_ready:
        return result
    states = Struct(**{
        **({'model': model.state} if model.cyclic else {}),
        **({'loss': loss.state} if loss.cyclic else {}),
    })
    return result.bind(state=states if states else ())


def _model_contract(step: Contract) -> Contract:
    return step.members.objective.members.model


@node
def _build_train_step(objective: Node, opt: Node, trainable: str | Callable | None) -> Node:
    """Builds a single-step gradient and optimizer update over `objective`."""
    select = _selection(trainable)
    model = objective.members.model
    loss = objective.members.loss
    apply_fields = _training_fields(model.contract, loss.contract)
    input_spec = _training_input_spec(model.contract, loss.contract)
    objective_param_form = (
        objective.contract._def.calls.param.form
        if objective.parametric else CallForm.from_values(Struct())
    )
    param_form = CallForm(Struct(
        objective=CallField.nested(objective_param_form),
    ))

    def apply_fn(contract, param, state, input, rng):
        current_objective = contract.members.objective
        current_opt = contract.members.opt

        _, constants = _partition(
            current_objective._sparse_param(param.objective), select)

        def loss_wrapper(weights):
            objective_state, result = current_objective.apply(
                _merge(weights, constants),
                state.objective,
                current_objective.feed(input),
                rng.child(current_objective.apply_takes_rng),
            )
            scalar, retained = split_aux(result)
            return scalar, Struct(state=objective_state, retained=retained)

        (scalar, objective_call), grads = jax.value_and_grad(
            loss_wrapper, has_aux=True)(state.opt.params)
        opt_state, _ = current_opt.apply(
            param.opt if current_opt.parametric else (), state.opt,
            current_opt.feed(grads),
            rng.child(current_opt.apply_takes_rng),
        )
        retained = objective_call.retained
        objective_aux = {}
        if retained.model:
            objective_aux['model'] = retained.model
        if retained.loss:
            objective_aux['loss'] = retained.loss
        aux_fields = {'loss': scalar}
        if objective_aux:
            aux_fields['objective'] = Aux(**objective_aux)
        return Struct(
            objective=objective_call.state,
            opt=opt_state,
        ), (retained.output, Aux(**aux_fields))

    def param_fn(contract, param_input, rng):
        current_objective = contract.members.objective
        current_opt = contract.members.opt
        objective_contract = current_objective.for_input(
            contract.input_spec)
        weights = objective_contract.param(
            param_input.objective,
            rng.child(objective_contract.param_takes_rng),
        )
        optimized, _ = _partition(objective_contract._sparse_param(weights), select)
        return Struct(
            opt=_opt_param(
                current_opt, optimized,
                rng.child(current_opt.param_takes_rng)),
            objective=weights,
        )

    def state_inputs(state_input):
        return state_input.opt, state_input.objective

    def initialized(contract, param, opt_input, objective_state, rng):
        current_objective = contract.members.objective
        current_opt = contract.members.opt
        optimized, _ = _partition(
            current_objective._sparse_param(param.objective), select)
        opt_state = current_opt.prime(
            param.opt if current_opt.parametric else (),
            opt_input,
            optimized,
            rng.child(current_opt.init_takes_rng),
        )
        return Struct(objective=objective_state, opt=opt_state)

    def prime_fn(contract, param, state_input, input, rng):
        current_objective = contract.members.objective
        opt_input, objective_input = state_inputs(state_input)
        objective_state = current_objective.prime(
            param.objective,
            objective_input,
            input,
            rng.child(current_objective.init_takes_rng),
        )
        return initialized(
            contract, param, opt_input, objective_state, rng)

    def init_fn(contract, param, state_input, rng):
        current_objective = contract.members.objective
        opt_input, objective_input = state_inputs(state_input)
        objective_contract = current_objective.for_input(
            contract.input_spec)
        objective_state = objective_contract.init(
            param.objective,
            objective_input,
            rng.child(objective_contract.init_takes_rng),
        )
        return initialized(
            contract, param, opt_input, objective_state, rng)

    def objective_params(node, param, state):
        current_objective = node.members.objective
        _, constants = _partition(
            current_objective.contract._sparse_param(param.objective), select)
        return _merge(state.opt.params, constants)

    def params(node, param, state):
        """Returns the current `model` parameter tree merging adapted trainable leaves with frozen constants."""
        current_model = node.members.objective.members.model
        return (getattr(objective_params(node, param, state), 'model')
                if current_model.parametric else ())

    def trained(node, param, state):
        """Returns `model` bound to adapted parameters (as `PNode` when acyclic, `PSNode` with live state when cyclic)."""
        current_model = node.members.objective.members.model
        model_param = (getattr(objective_params(node, param, state), 'model')
                       if current_model.parametric else ())
        if current_model.cyclic:
            return current_model.bind(
                model_param, state=getattr(state.objective, 'model'))
        return current_model.bind(model_param)

    return Composite(objective=objective, opt=opt)._roles_with_forms(
        apply_fn,
        param=param_fn,
        init=init_fn,
        prime=prime_fn,
        name=f'train_step({model.name})',
        methods={'params': params, 'trained': trained},
        param_form=param_form,
        apply_fields=apply_fields,
        input_spec=input_spec,
        requires_input=objective.contract.init_requires_input,
        tags=model.tags,
    )


@node
def train_step(model: BaseNode, loss_fn: LossFn | BaseNode,
               tx: BaseNode | optax.GradientTransformation, *,
               trainable: str | Callable | None = None) -> BaseNode:
    """Composes `model`, `loss_fn`, and optimizer `tx` into a cyclic gradient-update `Node`.

    `loss_fn` may be a Python callable or a `Node`; its first declared input
    receives the clean `model` output by position, any parameter named `aux`
    receives `model`'s `Aux` payload, and any remaining loss fields join
    `model`'s call fields on the public step signature. Each step emits the
    pre-update `model` output, sows the scalar objective under `Aux(loss=...)`
    (plus any nested `Aux` under `aux.objective`), and exposes `.params()` and
    `.trained()` methods on bound views.

    `trainable` restricts gradient computation and optimizer state to matching
    dotted parameter paths: `None` selects all parameters, a `str` matches any
    parameter path containing that segment (such as `'policy'`), and a callable
    `predicate(path) -> bool` filters paths directly while holding unselected
    parameters fixed in `.param.objective`.
    """
    if not _is_node(model):
        raise TypeError(
            'train_step takes a Node, PNode, or PSNode; '
            f'got {model!r}')
    objective = _objective(model, _as_loss(loss_fn))
    step = _build_train_step(objective.node, _as_optimizer(tx), trainable)
    if not objective.bound:
        return step

    current_opt = step.members.opt
    optimized, _ = _partition(objective.param, _selection(trainable))
    param = (Struct(
        opt=_opt_param(
            current_opt.contract, optimized,
            MaybeKeyStream()),
        objective=objective.param,
    ) if current_opt.parametric else Struct(objective=objective.param))
    bound_step = step.bind(param)
    if not objective.state_bound:
        return bound_step
    opt_state = current_opt.contract.prime(
        param.opt if current_opt.parametric else (),
        Struct(),
        optimized,
        MaybeKeyStream(),
    )
    return bound_step.bind(
        state=Struct(objective=objective.state, opt=opt_state))


@node
def trained(step: BaseNode) -> Node | PNode:
    """Scans optimization `step` across a data sequence and returns the final `.trained()` model.

    When `step` is a `PSNode`, optimization resumes from its bound state;
    otherwise each call primes fresh optimizer and model state from the first
    sequence element before scanning updates and discarding optimizer state at
    the end of the run.
    """
    step_node = _require_trained_step(step)
    starts_bound = step.state_bound
    starting_state = (step_node.contract._dense_state(step.state)
                      if starts_bound else None)

    def finalize(current, param, final, aux):
        done = bind(current, param, state=final).trained()
        return done if aux is None else (done, aux)

    fields = () if starts_bound else step_node.contract.state_input_fields

    def apply_fn(contract, param, input, rng):
        current = contract.members.step
        state_input, sequence = _split_state_fields(input, fields)
        initial = (starting_state if starts_bound else
                   _fresh_step_state(current, param, state_input, sequence, rng))
        final, outputs = scan_steps(
            current, param, initial, sequence, rng)
        _, aux = split_aux(outputs)
        return finalize(current, param, final, aux)

    result = Wrapper(step=step_node).roles(
        name=f'trained({step_node.name})',
        param=_sequence_parameterize('step', fields),
        init=False,
        apply=apply_fn,
        **_internalized_form(step_node.contract, fields),
        apply_takes_rng=(
            (not starts_bound and step_node.contract.init_takes_rng)
            or step_node.contract.apply_takes_rng),
    )
    return step._transfer_bindings(result, ('param',))
