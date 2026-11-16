"""Verifies multi-axis, routing, surgery, and PRNG-splitting transform compositions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from nodejax.struct import Struct
from nodejax.core.authoring import Leaf
from nodejax import KeyStream
from nodejax.core.compose import serial
from nodejax import Node, nn
from nodejax.transforms.structure import map_members
from nodejax.transforms.policy import freeze, detach, tree_freeze, tree_detach
from nodejax.transforms.axes.batch import batch
from nodejax.transforms.axes.ensemble import ensemble
from nodejax.transforms.iteration.stack import stack
from nodejax.transforms.iteration.repeat import repeat
from nodejax.transforms.iteration.scan import scan, scanned
from nodejax.transforms.wiring.at import at
from nodejax.transforms.policy.taps import taps


def StatefulLinearNode(D: int = 4) -> Node:
    """Builds a parametric cyclic linear Node that increments a step counter."""
    def param(node, rng: KeyStream):
        return Struct(w=jax.random.normal(rng.next(), (D, D)))

    def init(param):
        return Struct(count=jnp.array(0))

    def apply(param, state, input):
        return state.replace(count=state.count + 1), jnp.dot(input, param.w)

    return Leaf(apply, param=param, init=init, name='stateful_linear')


def StochasticNode() -> Node:
    """Builds a parametric cyclic Node that draws `apply`-time PRNG noise."""
    def param(node, rng: KeyStream):
        return Struct(scale=jnp.array(1.0, dtype=jnp.float32))

    def init(param):
        return Struct(count=jnp.array(0))

    def apply(param, state, x, rng):
        draw = jax.random.normal(rng.next(), x.shape)
        return state.replace(count=state.count + 1), x * param.scale + draw

    return Leaf(apply, param=param, init=init, name='stochastic_node')


def test_batch_ensemble_and_ensemble_batch():
    """Verifies both `batch(ensemble(node))` and `ensemble(batch(node))` axis orderings."""
    D = 4
    N_ens = 3
    B = 5

    StatefulLinear = StatefulLinearNode(D)

    be = batch(ensemble(StatefulLinear, n=N_ens), n=B).with_input(jnp.zeros((B, D))).parameterize(rng=jax.random.PRNGKey(0))
    s_be = be.init()
    assert jax.tree.leaves(be.param)[0].shape == (N_ens, D, D)
    assert s_be.count.shape == (B, N_ens)

    x_batch = jnp.ones((B, D))
    s_be2, out_be = be.apply(s_be, x_batch)
    assert out_be.shape == (B, N_ens, D)
    assert jnp.all(s_be2.count == 1)

    eb = ensemble(batch(StatefulLinear, n=B), n=N_ens).with_input(jnp.zeros((B, D))).parameterize(rng=jax.random.PRNGKey(0))
    s_eb = eb.init()
    assert jax.tree.leaves(eb.param)[0].shape == (N_ens, D, D)

    s_eb2, out_eb = eb.apply(s_eb, x_batch)
    assert out_eb.shape == (N_ens, B, D)


def test_batch_stack_and_stack_batch():
    """Verifies both `batch(stack(node))` and `stack(batch(node))` compositions."""
    D = 4
    L = 3
    B = 5

    StatefulLinear = StatefulLinearNode(D)

    bs = batch(stack(StatefulLinear, n=L), n=B).with_input(jnp.zeros((B, D))).parameterize(rng=jax.random.PRNGKey(0))
    s_bs = bs.init()
    assert jax.tree.leaves(bs.param)[0].shape == (L, D, D)
    assert s_bs.count.shape == (B, L)

    x_batch = jnp.ones((B, D))
    s_bs2, out_bs = bs.apply(s_bs, x_batch)
    assert out_bs.shape == (B, D)

    sb = stack(batch(StatefulLinear, n=B), n=L).with_input(jnp.zeros((B, D))).parameterize(rng=jax.random.PRNGKey(0))
    s_sb = sb.init()
    assert jax.tree.leaves(sb.param)[0].shape == (L, D, D)

    s_sb2, out_sb = sb.apply(s_sb, x_batch)
    assert out_sb.shape == (B, D)


def test_batch_scan_and_scan_batch():
    """Verifies both `batch(scanned(node))` and `scanned(batch(node))` axis layouts."""
    D = 4
    T = 6
    B = 5

    StatefulLinear = StatefulLinearNode(D)

    bsc = batch(scanned(StatefulLinear), n=B).with_input(jnp.zeros((B, T, D))).parameterize(rng=jax.random.PRNGKey(0))
    s_bsc = bsc.init()
    assert s_bsc == ()

    x_seq_batch = jnp.ones((B, T, D))
    out_bsc = bsc.apply(x_seq_batch)
    assert out_bsc.shape == (B, T, D)

    scb = scanned(batch(StatefulLinear, n=B)).with_input(jnp.zeros((T, B, D))).parameterize(rng=jax.random.PRNGKey(0))
    s_scb = scb.init()
    assert s_scb == ()

    x_time_first = jnp.ones((T, B, D))
    out_scb = scb.apply(x_time_first)
    assert out_scb.shape == (T, B, D)


def test_ensemble_stack_and_ensemble_scan():
    """Verifies `ensemble(stack(node))` and `ensemble(scanned(node))` compositions."""
    D = 4
    N_ens = 3
    L = 4
    T = 5

    StatefulLinear = StatefulLinearNode(D)

    es = ensemble(stack(StatefulLinear, n=L), n=N_ens).with_input(jnp.zeros((D,))).parameterize(rng=jax.random.PRNGKey(0))
    s_es = es.init()
    assert jax.tree.leaves(es.param)[0].shape == (N_ens, L, D, D)
    assert s_es.count.shape == (N_ens, L)

    x = jnp.ones((D,))
    s_es2, out_es = es.apply(s_es, x)
    assert out_es.shape == (N_ens, D)

    escan = ensemble(scanned(StatefulLinear), n=N_ens).with_input(jnp.zeros((T, D))).parameterize(rng=jax.random.PRNGKey(0))
    s_escan = escan.init()
    assert s_escan == ()

    x_seq = jnp.ones((T, D))
    out_escan = escan.apply(x_seq)
    assert out_escan.shape == (N_ens, T, D)


def test_batch_freeze_and_freeze_batch():
    """Verifies that `batch(freeze(node, st))` and `freeze(batch(node), batch_st)` agree."""
    D = 4
    B = 5
    StatefulLinear = StatefulLinearNode(D)
    node = StatefulLinear.with_input(jnp.zeros((D,))).parameterize(rng=jax.random.PRNGKey(0))
    st = node.init()

    bf = batch(freeze(node, st), n=B).with_input(jnp.zeros((B, D))).parameterize(rng=jax.random.PRNGKey(0))
    s_bf = bf.init()
    assert s_bf == ()

    x_batch = jnp.ones((B, D))
    out_bf = bf.apply(x_batch)
    assert out_bf.shape == (B, D)

    b_nd = batch(StatefulLinear, n=B).with_input(jnp.zeros((B, D)))
    s_b = b_nd.parameterize(rng=jax.random.PRNGKey(0)).init()
    fb = freeze(b_nd, s_b).parameterize(rng=jax.random.PRNGKey(0))
    s_fb = fb.init()
    assert s_fb == ()

    out_fb = fb.apply(x_batch)
    assert jnp.allclose(out_bf, out_fb)


def test_tree_freeze_on_nested_transform_towers():
    """Verifies `tree_freeze` across a nested `batch(ensemble(serial(...)))` tower."""
    D = 4
    N_ens = 3
    B = 5

    pipe = serial(
        l1=StatefulLinearNode(D),
        l2=StatefulLinearNode(D),
    )
    tower = batch(ensemble(pipe, n=N_ens), n=B).with_input(jnp.zeros((B, D))).parameterize(rng=jax.random.PRNGKey(0))
    st = tower.init()

    frozen_tower = tree_freeze(tower, st)
    s_ft = frozen_tower.init()
    assert s_ft == ()

    x_batch = jnp.ones((B, D))
    out_ft = frozen_tower.apply(x_batch)
    assert out_ft.shape == (B, N_ens, D)


def test_detach_stack_and_stack_detach():
    """Verifies that `stack(detach(node))` zeroes gradients across all stacked layers."""
    D = 4
    L = 3
    StatefulLinear = StatefulLinearNode(D)

    sd = stack(detach(StatefulLinear), n=L).with_input(jnp.zeros((D,))).parameterize(rng=jax.random.PRNGKey(0))
    s_sd = sd.init()

    def loss(p):
        bound_sd = sd.node.bind(p)
        _, out = bound_sd.apply(s_sd, jnp.ones((D,)))
        return jnp.sum(out)

    grads = jax.grad(loss)(sd.param)
    assert jnp.all(grads.w == 0.0)


def test_batch_at_and_stack_at():
    """Verifies field routing via `at` inside `batch` and `stack`."""
    D = 4
    B = 5
    L = 3
    StatefulLinear = StatefulLinearNode(D)

    sample_input = Struct(x=jnp.zeros((D,)), y=jnp.zeros((2,)))

    bat = batch(at(StatefulLinear, 'x'), n=B).with_input(Struct(x=jnp.zeros((B, D)), y=jnp.zeros((B, 2)))).parameterize(rng=jax.random.PRNGKey(0))
    s_bat = bat.init()

    s_bat2, out_bat = bat.apply(s_bat, x=jnp.ones((B, D)), y=jnp.ones((B, 2)))
    assert out_bat.x.shape == (B, D)
    assert jnp.allclose(out_bat.y, jnp.ones((B, 2)))

    sat = stack(at(StatefulLinear, 'x'), n=L).with_input(sample_input).parameterize(rng=jax.random.PRNGKey(0))
    s_sat = sat.init()

    s_sat2, out_sat = sat.apply(s_sat, bundle=sample_input)
    assert out_sat.x.shape == (D,)
    assert jnp.allclose(out_sat.y, sample_input.y)


def test_batch_taps_and_stack_taps():
    """Verifies intermediate wire recording via `taps` under `batch` and `stack`."""
    D = 4
    B = 5
    L = 3
    StatefulLinear1 = StatefulLinearNode(D)
    StatefulLinear2 = StatefulLinearNode(D)

    pipe = serial(a=StatefulLinear1, b=StatefulLinear2)

    btap = batch(taps(pipe), n=B).with_input(jnp.zeros((B, D))).parameterize(rng=jax.random.PRNGKey(0))
    s_btap = btap.init()

    x_batch = jnp.ones((B, D))
    s_btap2, (out_val, out_taps) = btap.apply(s_btap, x_batch)
    assert out_taps.a.shape == (B, D)
    assert out_taps.b.shape == (B, D)

    stap = stack(taps(pipe), n=L).with_input(jnp.zeros((D,))).parameterize(rng=jax.random.PRNGKey(0))
    s_stap = stap.init()

    x_single = jnp.ones((D,))
    s_stap2, (out_val_s, out_taps_s) = stap.apply(s_stap, x_single)
    assert out_taps_s.a.shape == (L, D)
    assert out_taps_s.b.shape == (L, D)


def test_rng_splitting_across_batch_and_ensemble():
    """Verifies `apply`-time PRNG key splitting across combined `batch` and `ensemble` axes."""
    D = 4
    N_ens = 3
    B = 5

    stoch = StochasticNode()
    be_stoch = batch(ensemble(stoch, n=N_ens), n=B).with_input(jnp.zeros((B, D))).parameterize(rng=jax.random.PRNGKey(0))
    s = be_stoch.init()

    x_batch = jnp.ones((B, D))
    s2, out = be_stoch.apply(s, x=x_batch, rng=jax.random.PRNGKey(42))
    assert out.shape == (B, N_ens, D)


def test_map_members_rebuild_on_nested_towers():
    """Verifies `map_members` replacing a leaf member inside `batch(stack(serial(...)))`."""
    D = 4
    L = 3
    B = 5

    pipe = serial(
        l1=nn.Linear(D),
        l2=nn.Linear(D),
    )
    tower = batch(stack(pipe, n=L), n=B).with_input(jnp.zeros((B, D))).parameterize(rng=jax.random.PRNGKey(0))

    new_l2 = nn.Linear(D)
    rebuilt_tower = map_members(
        tower.node, lambda m: new_l2 if m.name == 'l2' else m)

    rebuilt_tower = rebuilt_tower.with_input(jnp.zeros((B, D))).parameterize(rng=jax.random.PRNGKey(1))

    x_batch = jnp.ones((B, D))
    out_reb = rebuilt_tower.apply(x_batch)
    assert out_reb.shape == (B, D)


def test_batch_ensemble_scan_triple_tower():
    """Verifies `batch(ensemble(scanned(node)))` over batched sequence inputs."""
    D = 4
    N_ens = 3
    T = 6
    B = 5

    StatefulLinear = StatefulLinearNode(D)
    bes = batch(ensemble(scanned(StatefulLinear), n=N_ens), n=B).with_input(jnp.zeros((B, T, D))).parameterize(rng=jax.random.PRNGKey(0))
    s_bes = bes.init()
    assert s_bes == ()

    x_seq_batch = jnp.ones((B, T, D))
    out_bes = bes.apply(x_seq_batch)
    assert out_bes.shape == (B, N_ens, T, D)


def test_batch_ensemble_stack_triple_tower():
    """Verifies `batch(ensemble(stack(node, n=L)))` parameter and state shapes."""
    D = 4
    N_ens = 3
    L = 4
    B = 5

    StatefulLinear = StatefulLinearNode(D)
    best = batch(ensemble(stack(StatefulLinear, n=L), n=N_ens), n=B).with_input(jnp.zeros((B, D))).parameterize(rng=jax.random.PRNGKey(0))
    s_best = best.init()
    assert s_best.count.shape == (B, N_ens, L)
    assert jax.tree.leaves(best.param)[0].shape == (N_ens, L, D, D)

    x_batch = jnp.ones((B, D))
    s_best2, out_best = best.apply(s_best, x_batch)
    assert out_best.shape == (B, N_ens, D)
    assert s_best2.count.shape == (B, N_ens, L)
    assert jnp.all(s_best2.count == 1)


def test_batch_repeat_freeze_triple_tower():
    """Verifies `batch(repeat(freeze(node, st), n=T))` over frozen state."""
    D = 4
    T = 6
    B = 5

    StatefulLinear = StatefulLinearNode(D)
    node = StatefulLinear.parameterize(rng=jax.random.PRNGKey(0))
    st = node.init()

    brf = batch(repeat(freeze(node, st), n=T), n=B).with_input(jnp.zeros((B, T, D))).parameterize(rng=jax.random.PRNGKey(0))
    s_brf = brf.init()
    assert s_brf == ()

    x_seq_batch = jnp.ones((B, T, D))
    out_brf = brf.apply(x_seq_batch)
    assert out_brf.shape == (B, T, D)
