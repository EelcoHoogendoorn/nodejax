from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.core.types import LossFn
from nodejax.core.wrapper import Wrapper
from nodejax.core.binding import (
    Aux, split_aux,
)
from nodejax.core.composite import Composite
from nodejax.core.contract import CallField, CallForm, Contract
from nodejax.core.node import BaseNode, Node, _is_node
from nodejax.core.pnode import PNode
from nodejax.core.rng import MaybeKeyStream
from nodejax.core.ambient import node
from nodejax.core.authoring import Leaf
from nodejax.transforms.iteration.scan import (
    _fresh_step_state, _internalized_form, _sequence_parameterize,
    _split_state_fields, _state_fields,
)
from nodejax.transforms.transform import (
    bind, scan_steps, transform,
)

if TYPE_CHECKING:
    import optax


@node
def optimizer(tx: optax.GradientTransformation, name: str = 'optimizer') -> PNode:
    """Wrap an Optax transformation as a stateful optimizer node.

    State contains both the optimized parameters and Optax's own state. The
    initializer is primed from the initial parameter tree.
    """
    import optax

    def init(input):
        return Struct(params=input, opt=tx.init(input))

    def apply(state, input):                       # input: the gradients
        updates, opt = tx.update(input, state.opt, state.params)
        return Struct(params=optax.apply_updates(state.params, updates), opt=opt), None

    return Leaf(apply, init=init, name=name)


@node
def learned_sgd(lr0: float, name: str = 'learned_sgd') -> PNode:
    """SGD with one learnable step size per parameter leaf."""
    def param(node):
        return jax.tree.map(lambda x: jnp.full_like(x, lr0), node.input)

    def init(input):                       # primed from the initial weights
        return Struct(params=input)        # no optimizer state: SGD carries none

    def apply(param, state, input):        # input: the gradients
        params = jax.tree.map(lambda w, g, lr: w - lr * g,
                              state.params, input, param)
        return Struct(params=params), None

    return Leaf(apply, param=param, init=init, name=name)


def _as_optimizer(tx) -> Node:
    """Normalize an optimizer node or Optax transformation to a Node."""
    return tx.node if _is_node(tx) else optimizer(tx).node


@transform(preserves='param')
def opt_reinit(inner: Node, boundary: str) -> Node:
    """Reset optimizer internals at ``boundary`` while retaining parameters."""
    def action(carried, init, decided):
        return decided.replace(opt=init.opt)

    return Wrapper(inner=inner)(
        name=f'opt_reinit({inner.name})',
        boundary={boundary: action})


def _opt_param(opt: Contract, weights: Any, rng) -> Any:
    """Build learned optimizer parameters from the objective parameter shapes."""
    if not opt.parametric:
        return ()
    return opt.for_input(weights).param(Struct(), rng)


def _require_train_step(step: Any, who: str) -> Node:
    """Return the unbound ``train_step`` node required by ``who``."""
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


def _require_trained_step(step: Any) -> Node:
    """Return a stateful optimization step with a trained-model view."""
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
    """Lift a Python loss into the same Node contract as an authored loss."""
    if _is_node(loss_fn):
        return loss_fn
    if not callable(loss_fn):
        raise TypeError(
            'train_step loss must be a Node or callable; '
            f'got {loss_fn!r}')
    return Leaf(loss_fn)


def _loss_call(loss: Contract) -> tuple[str, tuple[str, ...], bool]:
    """Split a loss call into its model output, side inputs, and Aux opt-in."""
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
    """The model call followed by loss inputs not supplied by the objective."""
    if model._apply_form.open:
        raise TypeError(
            f'train_step model {model.name!r} must declare a closed call form')
    _, side_fields, _ = _loss_call(loss)
    return model.apply_fields + tuple(
        field for field in side_fields if field not in model.apply_fields)


def _training_input_spec(model: Contract, loss: Contract):
    """Merge available evidence for every public training input."""
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
    """Whether a constructor needs evidence from the loss call site."""
    param = loss._def.calls.param
    init = loss._def.calls.init
    return bool(
        (not loss.bound and param is not None and param.reads_def)
        or (not loss.state_bound and init is not None
            and (init.reads_def or init.requires_input))
    )


def _select(bundle: Struct, fields: tuple[str, ...]) -> Struct:
    """Select a closed child call from one training-step input bundle."""
    return Struct(**{field: bundle[field] for field in fields})


def _loss_takes_aux(loss_fn: LossFn) -> bool:
    """Whether a loss explicitly declares the model's Aux."""
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
    target_fn: Callable[[Any], Any],
) -> LossFn:
    """Map loss data to a target while preserving explicit Aux opt-in."""
    if _loss_takes_aux(loss_fn):
        def loss(output, data, *, aux):
            return loss_fn(output, target_fn(data), aux=aux)
    else:
        def loss(output, data):
            return loss_fn(output, target_fn(data))
    return loss


@node
def _objective(model: BaseNode, loss: BaseNode) -> BaseNode:
    """Compose a model and loss while retaining their seam as Aux."""
    if not _is_node(model) or not _is_node(loss):
        raise TypeError('an objective requires model and loss Nodes')

    model_fields = model.contract.apply_fields
    loss_output, _, _ = _loss_call(loss.contract)

    call_loss_during_construction = _loss_needs_construction_call(loss)

    def routed(self, input):
        """Author the construction walk through the same model-loss seam."""
        output = self.model(bundle=_select(input, model_fields))
        if not call_loss_during_construction:
            return output
        loss_input = {
            field: (output if field == loss_output else
                    Aux() if field == 'aux' else input[field])
            for field in loss.contract.apply_fields
        }
        return self.loss(bundle=Struct(**loss_input))

    # The authored Composite owns parameterization, initialization, and their
    # shape walks. Only runtime glue is replaced so model Aux can be forwarded
    # to a loss that explicitly requests it.
    #
    # TODO: Once authored wiring can explicitly read the Aux emitted by a
    # member it just called, express this whole seam as an ordinary authored
    # Composite. The required surface is local member-result access, not
    # ambient access to incoming Aux. It should then replace the manual member
    # calls, child-key allocation, state collection, and Aux routing below.
    scaffold = Composite(model=model, loss=loss)(
        routed, name='objective')

    def apply_fn(contract, param, state, input, rng):
        """Run both Nodes and retain the model output outside the scalar."""
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
    """The model trained by a canonical train-step contract."""
    return step.members.objective.members.model


@node
def _build_train_step(objective: Node, opt: Node) -> Node:
    """Build one optimizer update over a scalar objective Node."""
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

        def loss_wrapper(weights):
            objective_state, result = current_objective.apply(
                weights,
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
        optimized = objective_contract._sparse_param(weights)
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
        optimized = current_objective._sparse_param(param.objective)
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

    def params(node, state):
        """The model parameters as the optimizer currently holds them."""
        current_model = node.members.objective.members.model
        return (getattr(state.opt.params, 'model')
                if current_model.parametric else ())

    def trained(node, state):
        """The model bound to the parameters and state this step has reached."""
        current_model = node.members.objective.members.model
        model_param = (getattr(state.opt.params, 'model')
                       if current_model.parametric else ())
        model_state = (getattr(state.objective, 'model')
                       if current_model.cyclic else ())
        return current_model.bind(
            model_param, state=model_state)

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
               tx: optax.GradientTransformation) -> BaseNode:
    """Optimize the scalar composition of ``model`` and a loss Node.

    A callable loss is lifted to a Node. Its first declared input receives the
    clean model output by position, irrespective of that field's name. Its
    remaining inputs join the model's public call fields.
    A field named ``aux`` instead receives model Aux. The step returns the
    model output and reports the scalar under ``aux.loss``. Aux emitted by
    either objective member is nested under ``aux.objective.model`` or
    ``aux.objective.loss``.
    """
    if not _is_node(model):
        raise TypeError(
            'train_step takes a Node, PNode, or PSNode; '
            f'got {model!r}')
    objective = _objective(model, _as_loss(loss_fn))
    step = _build_train_step(objective.node, _as_optimizer(tx))
    if not objective.bound:
        return step

    current_opt = step.members.opt
    param = (Struct(
        opt=_opt_param(
            current_opt.contract, objective.param,
            MaybeKeyStream()),
        objective=objective.param,
    ) if current_opt.parametric else Struct(objective=objective.param))
    bound_step = step.bind(param)
    if not objective.state_bound:
        return bound_step
    opt_state = current_opt.contract.prime(
        param.opt if current_opt.parametric else (),
        Struct(),
        param.objective,
        MaybeKeyStream(),
    )
    return bound_step.bind(
        state=Struct(objective=objective.state, opt=opt_state))


@node
def trained(step: BaseNode) -> Node | PNode:
    """Run an optimization step over a sequence and return its final model.

    A state-bound step continues from its state; otherwise each call starts
    from a fresh initialization. The step's ``trained()`` method defines the
    returned model, leaving optimizer-specific state behind.
    """
    step_node = _require_trained_step(step)
    starts_bound = step.state_bound
    starting_state = (step_node.contract._dense_state(step.state)
                      if starts_bound else None)

    def finalize(current, param, final, aux):
        done = bind(current, param, state=final).trained()
        return done if aux is None else (done, aux)

    fields = () if starts_bound else _state_fields(step_node.contract)

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
