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
from nodejax.core.contract import CallForm, Contract
from nodejax.core.node import BaseNode, Node, _is_node
from nodejax.core.pnode import PNode
from nodejax.core.rng import MaybeKeyStream
from nodejax.core.ambient import node
from nodejax.core.authoring import Leaf
from nodejax.transforms.iteration.scan import (
    _fresh_step_state, _sequence_parameterize, _sequence_spec,
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
    """Build learned optimizer parameters from the model parameter shapes."""
    if not opt.parametric:
        return ()
    return opt.for_input(weights).param(Struct(), rng)


def _require_train_step(step: Any, who: str) -> Node:
    """Return the unbound ``train_step`` node required by ``who``."""
    if not _is_node(step):
        raise TypeError(
            f'{who} consumes the product of train_step: '
            'a cyclic parametric node holding the model at opt/model. Got '
            f'{step!r}; spell it {who}(train_step(model, loss_fn, '
            'optimizer))')
    node = step.node
    if not (node.cyclic and node.parametric
            and 'opt' in node.members and 'model' in node.members):
        raise TypeError(
            f'{who} consumes the product of train_step: '
            'a cyclic parametric node holding the model at opt/model. Got '
            f'{step!r}; spell it {who}(train_step(model, loss_fn, '
            'optimizer))')
    return node


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
def _build_train_step(model: Node, loss_fn: LossFn, opt: Node) -> Node:
    """Build the stateful ``train_step`` node over ``model`` and ``opt``."""
    loss_takes_aux = _loss_takes_aux(loss_fn)

    def apply_fn(contract, param, state, input, rng):
        current_model = contract.members.model
        current_opt = contract.members.opt
        model_rng = rng.child(current_model.apply_takes_rng)
        model_input = (current_model.feed(input.input)
                       if current_model._accepts_input else
                       current_model.feed_bundle(input.input))

        def loss_wrapper(weights):
            model_state, output = current_model.apply(
                weights, state.model, model_input, model_rng)
            clean, model_aux = split_aux(output)
            loss_aux = Aux() if model_aux is None else model_aux
            loss = (loss_fn(clean, input.target, aux=loss_aux)
                    if loss_takes_aux else
                    loss_fn(clean, input.target))
            return loss, Struct(
                state=model_state, output=output)

        (loss, model_call), grads = jax.value_and_grad(
            loss_wrapper, has_aux=True)(state.opt.params)
        opt_state, _ = current_opt.apply(
            param.opt if current_opt.parametric else (), state.opt,
            current_opt.feed(grads),
            rng.child(current_opt.apply_takes_rng),
        )
        clean, model_aux = split_aux(model_call.output)
        aux = (Aux(loss=loss) if model_aux is None else
               Aux(loss=loss, model=model_aux))
        return Struct(opt=opt_state, model=model_call.state), (clean, aux)

    def param_fn(contract, param_input, rng):
        current_model = contract.members.model
        current_opt = contract.members.opt
        model_contract = (current_model.for_input(
            contract.input_spec_for('input'))
            if current_model._accepts_input else current_model)
        weights = model_contract.param(
            param_input,
            rng.child(model_contract.param_takes_rng),
        )
        return Struct(
            opt=_opt_param(
                current_opt, weights,
                rng.child(current_opt.param_takes_rng)),
            model=weights,
        )

    def state_inputs(state_input):
        return state_input.opt, state_input.model

    def initialized(contract, param, opt_input, model_state, rng):
        current_opt = contract.members.opt
        opt_state = current_opt.prime(
            param.opt if current_opt.parametric else (), opt_input,
            param.model,
            rng.child(current_opt.init_takes_rng),
        )
        return Struct(opt=opt_state, model=model_state)

    def prime_fn(contract, param, state_input, input, rng):
        current_model = contract.members.model
        opt_input, model_input = state_inputs(state_input)
        model_state = current_model.prime(
            param.model, model_input, input.input,
            rng.child(current_model.init_takes_rng),
        )
        return initialized(
            contract, param, opt_input, model_state, rng)

    def init_fn(contract, param, state_input, rng):
        current_model = contract.members.model
        opt_input, model_input = state_inputs(state_input)
        model_contract = (current_model.for_input(
            contract.input_spec_for('input'))
            if current_model._accepts_input else current_model)
        model_state = model_contract.init(
            param.model, model_input,
            rng.child(model_contract.init_takes_rng),
        )
        return initialized(
            contract, param, opt_input, model_state, rng)

    return Composite(opt=opt, model=model)._roles_with_forms(
        apply_fn,
        param=param_fn,
        init=init_fn,
        prime=prime_fn,
        name=f'train_step({model.name})',
        param_form=(model._def.calls.param.form
                    if model.parametric
                    else CallForm.from_values(Struct())),
        apply_fields=('input', 'target'),
        requires_input=model.contract.init_requires_input,
        tags=model.tags,
    )


@node
def train_step(model: BaseNode, loss_fn: LossFn,
               tx: optax.GradientTransformation) -> BaseNode:
    """Internalize one optimizer update per call while preserving bindings.

    The loss receives clean ``output`` and ``target``. A loss that declares
    ``aux`` also receives the model's Aux, or an empty ``Aux()`` when the
    model emitted none.
    """
    if not _is_node(model):
        raise TypeError(
            'train_step takes a Node, PNode, or PSNode; '
            f'got {model!r}')
    step = _build_train_step(model.node, loss_fn, _as_optimizer(tx))
    if not model.bound:
        return step

    current_opt = step.members.opt
    param = (Struct(
        opt=_opt_param(
            current_opt.contract, model.param,
            MaybeKeyStream()),
        model=model.param,
    ) if current_opt.parametric else Struct(model=model.param))
    bound_step = step.bind(param)
    if not model.state_bound:
        return bound_step
    opt_state = current_opt.contract.prime(
        param.opt if current_opt.parametric else (), Struct(), param.model,
        MaybeKeyStream(),
    )
    return bound_step.bind(
        state=Struct(opt=opt_state, model=model.state))


@node
def trained(step: BaseNode) -> Node | PNode:
    """Run a ``train_step`` node over a sequence and return its final model.

    A state-bound step continues from its state; otherwise each call starts
    from a fresh initialization. Optimizer state is omitted from the result.
    """
    step_node = _require_train_step(step, 'trained')
    starts_bound = step.state_bound
    starting_state = (step_node.contract._dense_state(step.state)
                      if starts_bound else None)

    def finalize(current, final, aux):
        done = bind(
            current.members.model,
            final.opt.params, state=final.model)
        return done if aux is None else (done, aux)

    def apply_fn(contract, param, input, rng):
        current = contract.members.step
        initial = (starting_state if starts_bound else
                   _fresh_step_state(current, param, input, rng))
        final, outputs = scan_steps(
            current, param, initial, input, rng)
        _, aux = split_aux(outputs)
        return finalize(current, final, aux)

    result = Wrapper(step=step_node).roles(
        name=f'trained({step_node.name})',
        param=_sequence_parameterize('step'),
        init=False,
        apply=apply_fn,
        input_spec=_sequence_spec(step_node.contract),
        apply_takes_rng=(
            (not starts_bound and step_node.contract.init_takes_rng)
            or step_node.contract.apply_takes_rng),
    )
    return step._transfer_bindings(result, ('param',))
