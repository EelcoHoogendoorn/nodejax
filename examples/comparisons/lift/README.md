# Reusable Lifted Stack (`lift`)

Each implementation builds one reusable `stack` transform and tests it against [`lift_common.py`](lift_common.py) across seven invariants:

- Independent parameter initialization per layer.
- Sequential forward execution where each layer's primary output feeds the next layer.
- Independent mutable state per layer.
- Explicit call-time PRNG splitting and deterministic replay.
- Rejection of both missing required PRNG keys and unsolicited PRNG keys.
- Auxiliary outputs stacked along the layer axis without entering the inter-layer carry.
- Compatibility with layers that emit no auxiliary values.

All three implementations (`lift_nodejax.py` 243 lines, `lift_nnx.py` 344 lines, `lift_equinox.py` 359 lines) pass those seven checks on a single level, and also test self-nesting (`stack(stack(layer))`) plus each framework's secondary state mechanism.

## Inspection vs. declared contracts

A generic `stack` transform needs five facts about its wrapped layer before executing: how parameters are constructed, whether state is carried, whether call-time randomness is consumed, what inputs `apply` accepts, and whether auxiliary data accompanies the primary output.

`lift_nnx.py` and `lift_equinox.py` already have to declare two of those facts explicitly (`make_layer` factories for independent construction and `WithAux` wrappers to disambiguate tuple outputs), while attempting to discover RNG consumption via `inspect.signature(member.__call__)`. Because `LayerStack.__call__` must declare an optional `rng` parameter in case it wraps a stochastic layer, signature inspection confuses forwarding an optional `rng` with requiring `rng`, breaking self-nesting (`stack(stack(bare_layer))` demands an `rng` key even though `bare_layer` is deterministic). Replacing that third inspected fact with an explicit declaration converges piecemeal onto the `Def` and `Contract` that NodeJAX defines once for every `Node`.

## Split state mechanisms

Each file also tests wrapping the framework's secondary state representation (`single_batch_state`-tagged state in NodeJAX, `nnx.BatchStat` in NNX, and `eqx.nn.StateIndex` / `eqx.nn.State` in Equinox):

| Layer capability | NodeJAX (`lift_nodejax.py`) | Flax NNX (`lift_nnx.py`) | Equinox (`lift_equinox.py`) |
| :--- | :--- | :--- | :--- |
| Parameters only | carried | carried | carried |
| Call-time RNG | carried | carried | carried |
| Primary carried state | carried | carried | carried |
| State + call-time RNG | carried | carried | carried |
| Secondary state mechanism | carried (`single_batch_state` tag) | carried (`nnx.BatchStat`) | rejected (`eqx.nn.State` incompatible) |
| Self-nesting `stack(stack(bare))` | preserves `takes_rng=False` | breaks via signature inspection | breaks via signature inspection |

Equinox's `eqx.nn.StateIndex` fails to allocate independent per-layer slots under `eqx.filter_vmap`, forcing `lift_equinox.py` onto a `(successor, output)` return convention. That convention is incompatible with `eqx.nn.StatefulLayer` (`eqx.nn.BatchNorm`), splitting Equinox's state-carrying modules into two mutually incompatible worlds (`lift_equinox.py` vs. [`residual_equinox.py`](../residual/residual_equinox.py)).

Run the three columns directly:

```sh
python -m examples.comparisons.lift.lift_nodejax
python -m examples.comparisons.lift.lift_nnx
python -m examples.comparisons.lift.lift_equinox
```
