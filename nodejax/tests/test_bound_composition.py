"""All-bound composition closes over values without inspecting them."""

import jax.numpy as jnp

from nodejax import Leaf, Node, PNode, serial


def ArrayParameter(name: str) -> Node:
    def param(value):
        return jnp.asarray(value)

    def apply(param, input):
        return input + param

    return Leaf(apply, param=param, name=name)


def EmptyParameter() -> Node:
    def param():
        return ()

    def apply(param, input):
        return input

    return Leaf(apply, param=param, name='empty_parameter')


def test_all_bound_composition_accepts_array_parameters() -> None:
    left = ArrayParameter('left').parameterize(
        value=jnp.asarray([1.0, 2.0]))
    right = ArrayParameter('right').parameterize(
        value=jnp.asarray([3.0, 4.0]))

    pipe = serial(left=left, right=right)

    assert type(pipe) is PNode
    assert jnp.allclose(pipe.param.left, left.param)
    assert jnp.allclose(pipe.param.right, right.param)
    assert jnp.allclose(pipe.apply(jnp.zeros(2)), jnp.asarray([4.0, 6.0]))


def test_all_bound_composition_retains_a_parametric_empty_tree() -> None:
    identity = Leaf(lambda input: input, name='identity')
    pipe = serial(empty=EmptyParameter().parameterize(), identity=identity)

    assert type(pipe) is PNode
    assert pipe.param.__keys__ == ('empty',)
    assert pipe.param.empty == ()
    assert jnp.allclose(pipe.apply(jnp.asarray([-1.0, 1.0])),
                        jnp.asarray([-1.0, 1.0]))
