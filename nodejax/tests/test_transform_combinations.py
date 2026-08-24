from __future__ import annotations

import jax
import jax.numpy as jnp
from nodejax.struct import Struct
from nodejax.authoring import Leaf
from nodejax import KeyStream
from nodejax.compose import serial
from nodejax import Node, nn
from nodejax.transforms.tree_utils import map_members
from nodejax.transforms.freeze import freeze, detach, tree_freeze, tree_detach
from nodejax.transforms.batch import batch
from nodejax.transforms.ensemble import ensemble
from nodejax.transforms.stack import stack
from nodejax.transforms.repeat import repeat
from nodejax.transforms.scan import scan, scanned
from nodejax.transforms.at import at
from nodejax.transforms.taps import taps


# --- Helper Test Nodes ---

def StatefulLinearNode(D: int = 4) -> Node:
    """A parametric, cyclic node with internal state."""
    def param(node, rng: KeyStream):
        return Struct(w=jax.random.normal(rng.next(), (D, D)))

    def init(param):
        return Struct(count=jnp.array(0))

    def apply(param, state, input):
        return state.replace(count=state.count + 1), jnp.dot(input, param.w)

    return Leaf(apply, param=param, init=init, name='stateful_linear')


def StochasticNode() -> Node:
    """A parametric, cyclic node consuming apply-side RNG."""
    def param(node, rng: KeyStream):
        return Struct(scale=jnp.array(1.0, dtype=jnp.float32))

    def init(param):
        return Struct(count=jnp.array(0))

    def apply(param, state, x, rng):
        draw = jax.random.normal(rng.next(), x.shape)
        return state.replace(count=state.count + 1), x * param.scale + draw

    return Leaf(apply, param=param, init=init, name='stochastic_node')


# --- 1. Multi-Axis Lifts (Vmap + Scan Towers) ---

def test_batch_ensemble_and_ensemble_batch():
    """Dual vmap towers: batch of ensemble and ensemble of batch."""
    D = 4
    N_ens = 3
    B = 5

    StatefulLinear = StatefulLinearNode(D)

    # 1. batch(ensemble(node))
    be = batch(ensemble(StatefulLinear, n=N_ens), n=B).with_input(jnp.zeros((B, D))).parameterize(rng=jax.random.PRNGKey(0))
    s_be = be.init()
    assert jax.tree.leaves(be.param)[0].shape == (N_ens, D, D)
    assert s_be.count.shape == (B, N_ens)

    x_batch = jnp.ones((B, D))
    s_be2, out_be = be.apply(s_be, x_batch)
    assert out_be.shape == (B, N_ens, D)
    assert jnp.all(s_be2.count == 1)

    # 2. ensemble(batch(node))
    eb = ensemble(batch(StatefulLinear, n=B), n=N_ens).with_input(jnp.zeros((B, D))).parameterize(rng=jax.random.PRNGKey(0))
    s_eb = eb.init()
    assert jax.tree.leaves(eb.param)[0].shape == (N_ens, D, D)

    s_eb2, out_eb = eb.apply(s_eb, x_batch)
    assert out_eb.shape == (N_ens, B, D)


def test_batch_stack_and_stack_batch():
    """Batched layer scan: batch(stack) and stack(batch)."""
    D = 4
    L = 3
    B = 5

    StatefulLinear = StatefulLinearNode(D)

    # 1. batch(stack(node, n=L))
    bs = batch(stack(StatefulLinear, n=L), n=B).with_input(jnp.zeros((B, D))).parameterize(rng=jax.random.PRNGKey(0))
    s_bs = bs.init()
    assert jax.tree.leaves(bs.param)[0].shape == (L, D, D)
    assert s_bs.count.shape == (B, L)

    x_batch = jnp.ones((B, D))
    s_bs2, out_bs = bs.apply(s_bs, x_batch)
    assert out_bs.shape == (B, D)

    # 2. stack(batch(node), n=L)
    sb = stack(batch(StatefulLinear, n=B), n=L).with_input(jnp.zeros((B, D))).parameterize(rng=jax.random.PRNGKey(0))
    s_sb = sb.init()
    assert jax.tree.leaves(sb.param)[0].shape == (L, D, D)

    s_sb2, out_sb = sb.apply(s_sb, x_batch)
    assert out_sb.shape == (B, D)


def test_batch_scan_and_scan_batch():
    """Sequence processing over batched data: batch(scan) and scanned(batch)."""
    D = 4
    T = 6
    B = 5

    StatefulLinear = StatefulLinearNode(D)

    # 1. batch(scanned(node)) - scan input is (T, D), batched input is (B, T, D)
    bsc = batch(scanned(StatefulLinear), n=B).with_input(jnp.zeros((B, T, D))).parameterize(rng=jax.random.PRNGKey(0))
    s_bsc = bsc.init()
    assert s_bsc == ()

    x_seq_batch = jnp.ones((B, T, D))
    out_bsc = bsc.apply(x_seq_batch)
    assert out_bsc.shape == (B, T, D)

    # 2. scanned(batch(node)) - batch input is (B, D), sequence input is (T, B, D)
    scb = scanned(batch(StatefulLinear, n=B)).with_input(jnp.zeros((T, B, D))).parameterize(rng=jax.random.PRNGKey(0))
    s_scb = scb.init()
    assert s_scb == ()

    x_time_first = jnp.ones((T, B, D))
    out_scb = scb.apply(x_time_first)
    assert out_scb.shape == (T, B, D)


def test_ensemble_stack_and_ensemble_scan():
    """Population of stacked layers and scanned sequence models."""
    D = 4
    N_ens = 3
    L = 4
    T = 5

    StatefulLinear = StatefulLinearNode(D)

    # 1. ensemble(stack(node, n=L), n=N_ens)
    es = ensemble(stack(StatefulLinear, n=L), n=N_ens).with_input(jnp.zeros((D,))).parameterize(rng=jax.random.PRNGKey(0))
    s_es = es.init()
    assert jax.tree.leaves(es.param)[0].shape == (N_ens, L, D, D)
    assert s_es.count.shape == (N_ens, L)

    x = jnp.ones((D,))
    s_es2, out_es = es.apply(s_es, x)
    assert out_es.shape == (N_ens, D)

    # 2. ensemble(scanned(node), n=N_ens)
    escan = ensemble(scanned(StatefulLinear), n=N_ens).with_input(jnp.zeros((T, D))).parameterize(rng=jax.random.PRNGKey(0))
    s_escan = escan.init()
    assert s_escan == ()

    x_seq = jnp.ones((T, D))
    out_escan = escan.apply(x_seq)
    assert out_escan.shape == (N_ens, T, D)


# --- 2. State & Gradient Surgery under Lifts ---

def test_batch_freeze_and_freeze_batch():
    """Batched evaluation under frozen state."""
    D = 4
    B = 5
    StatefulLinear = StatefulLinearNode(D)
    node = StatefulLinear.with_input(jnp.zeros((D,))).parameterize(rng=jax.random.PRNGKey(0))
    st = node.init()

    # 1. batch(freeze(node, st))
    bf = batch(freeze(node, st), n=B).with_input(jnp.zeros((B, D))).parameterize(rng=jax.random.PRNGKey(0))
    s_bf = bf.init()
    assert s_bf == ()

    x_batch = jnp.ones((B, D))
    out_bf = bf.apply(x_batch)
    assert out_bf.shape == (B, D)

    # 2. freeze(batch(node), batch_st)
    b_nd = batch(StatefulLinear, n=B).with_input(jnp.zeros((B, D)))
    s_b = b_nd.parameterize(rng=jax.random.PRNGKey(0)).init()
    fb = freeze(b_nd, s_b).parameterize(rng=jax.random.PRNGKey(0))
    s_fb = fb.init()
    assert s_fb == ()

    out_fb = fb.apply(x_batch)
    assert jnp.allclose(out_bf, out_fb)


def test_tree_freeze_on_nested_transform_towers():
    """tree_freeze on a nested batch(ensemble(serial(...))) tower."""
    D = 4
    N_ens = 3
    B = 5

    pipe = serial(
        l1=StatefulLinearNode(D),
        l2=StatefulLinearNode(D),
    )
    tower = batch(ensemble(pipe, n=N_ens), n=B).with_input(jnp.zeros((B, D))).parameterize(rng=jax.random.PRNGKey(0))
    st = tower.init()

    # Freeze state in the nested tower
    frozen_tower = tree_freeze(tower, st)
    s_ft = frozen_tower.init()
    assert s_ft == ()                # everything frozen: no state at all

    x_batch = jnp.ones((B, D))
    out_ft = frozen_tower.apply(x_batch)
    assert out_ft.shape == (B, N_ens, D)


def test_detach_stack_and_stack_detach():
    """Gradient detachment across stacked layers."""
    D = 4
    L = 3
    StatefulLinear = StatefulLinearNode(D)

    # 1. stack(detach(node), n=L)
    sd = stack(detach(StatefulLinear), n=L).with_input(jnp.zeros((D,))).parameterize(rng=jax.random.PRNGKey(0))
    s_sd = sd.init()

    def loss(p):
        bound_sd = sd.node.bind(p)
        _, out = bound_sd.apply(s_sd, jnp.ones((D,)))
        return jnp.sum(out)

    grads = jax.grad(loss)(sd.param)
    assert jnp.all(grads.w == 0.0)


# --- 3. Structural Routing under Lifts (at & taps) ---

def test_batch_at_and_stack_at():
    """Field routing via at() under batch and stack transforms."""
    D = 4
    B = 5
    L = 3
    StatefulLinear = StatefulLinearNode(D)

    # Struct with field 'x'
    sample_input = Struct(x=jnp.zeros((D,)), y=jnp.zeros((2,)))

    # 1. batch(at(node, 'x'))
    bat = batch(at(StatefulLinear, 'x'), n=B).with_input(Struct(x=jnp.zeros((B, D)), y=jnp.zeros((B, 2)))).parameterize(rng=jax.random.PRNGKey(0))
    s_bat = bat.init()

    s_bat2, out_bat = bat.apply(s_bat, x=jnp.ones((B, D)), y=jnp.ones((B, 2)))
    assert out_bat.x.shape == (B, D)
    assert jnp.allclose(out_bat.y, jnp.ones((B, 2)))

    # 2. stack(at(node, 'x'), n=L)
    sat = stack(at(StatefulLinear, 'x'), n=L).with_input(sample_input).parameterize(rng=jax.random.PRNGKey(0))
    s_sat = sat.init()

    s_sat2, out_sat = sat.apply(s_sat, bundle=sample_input)
    assert out_sat.x.shape == (D,)
    assert jnp.allclose(out_sat.y, sample_input.y)


def test_batch_taps_and_stack_taps():
    """Tapping intermediate outputs under batch and stack lifts."""
    D = 4
    B = 5
    L = 3
    StatefulLinear1 = StatefulLinearNode(D)
    StatefulLinear2 = StatefulLinearNode(D)

    pipe = serial(a=StatefulLinear1, b=StatefulLinear2)

    # 1. batch(taps(pipe))
    btap = batch(taps(pipe), n=B).with_input(jnp.zeros((B, D))).parameterize(rng=jax.random.PRNGKey(0))
    s_btap = btap.init()

    x_batch = jnp.ones((B, D))
    s_btap2, (out_val, out_taps) = btap.apply(s_btap, x_batch)
    assert out_taps.a.shape == (B, D)
    assert out_taps.b.shape == (B, D)

    # 2. stack(taps(pipe), n=L)
    stap = stack(taps(pipe), n=L).with_input(jnp.zeros((D,))).parameterize(rng=jax.random.PRNGKey(0))
    s_stap = stap.init()

    x_single = jnp.ones((D,))
    s_stap2, (out_val_s, out_taps_s) = stap.apply(s_stap, x_single)
    assert out_taps_s.a.shape == (L, D)
    assert out_taps_s.b.shape == (L, D)


# --- 4. Apply-Side RNG Splitting across Multi-Axis Towers ---

def test_rng_splitting_across_batch_and_ensemble():
    """Apply-side PRNG key splitting across batch and ensemble dimensions."""
    D = 4
    N_ens = 3
    B = 5

    stoch = StochasticNode()
    be_stoch = batch(ensemble(stoch, n=N_ens), n=B).with_input(jnp.zeros((B, D))).parameterize(rng=jax.random.PRNGKey(0))
    s = be_stoch.init()

    x_batch = jnp.ones((B, D))
    # Calling apply with the key beside the field, split per element
    s2, out = be_stoch.apply(s, x=x_batch, rng=jax.random.PRNGKey(42))
    assert out.shape == (B, N_ens, D)


# --- 5. Deep Tower Rebuilds ---

def test_map_members_rebuild_on_nested_towers():
    """map_members on a nested batch(stack(serial(...))) tower."""
    D = 4
    L = 3
    B = 5

    pipe = serial(
        l1=nn.Linear(D),
        l2=nn.Linear(D),
    )
    tower = batch(stack(pipe, n=L), n=B).with_input(jnp.zeros((B, D))).parameterize(rng=jax.random.PRNGKey(0))

    # Replace l2 member deep in the tower
    new_l2 = nn.Linear(D)
    rebuilt_tower = map_members(
        tower.node, lambda m: new_l2 if m.name == 'l2' else m)

    # Re-parameterize and run forward pass
    rebuilt_tower = rebuilt_tower.with_input(jnp.zeros((B, D))).parameterize(rng=jax.random.PRNGKey(1))

    x_batch = jnp.ones((B, D))
    out_reb = rebuilt_tower.apply(x_batch)
    assert out_reb.shape == (B, D)


# --- 6. Triple Transform Composition Towers ---

def test_batch_ensemble_scan_triple_tower():
    """Batch of ensembles of sequence models: batch(ensemble(scanned(node)))."""
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
    """Batch of ensembles of stacked layers: batch(ensemble(stack(node, n=L)))."""
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
    """Batched sequence processing over a frozen non-cyclic model: batch(repeat(freeze(node, st)))."""
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
