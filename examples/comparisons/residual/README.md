# Transparent Residual Wrapper (`residual`)

A residual wrapper (`x + body(x)`) owns no parameters, state, PRNG keys, or initialization logic of its own, making it the minimal test of whether a framework can express a set-and-forget library transform.

In NodeJAX ([`nodejax/transforms/wiring/residual.py`](../../../nodejax/transforms/wiring/residual.py)), the entire transform is six lines:

```python
@node
def residual(body):
    wrapped = Wrapper(body=body)

    def apply(self, input):
        return input + self.body(input)

    return wrapped(apply, name=f'res({body.name})')
```

`Wrapper` forwards parameter construction, state initialization and input priming, call-time `KeyStream` routing, `Aux` propagation, member views, and `Generic` specialization automatically through the shared `Node` contract.

## Protocol fractures in NNX and Equinox

All three implementations (`residual_nodejax.py` 197 lines, `residual_nnx.py` 244 lines, `residual_equinox.py` 269 lines) handle unary deterministic and stochastic nesting when the signal is the first argument. Fractures appear as soon as the wrapped body uses recurrent carries, downstream lifted transforms, or error-safe state transitions:

| Property | NodeJAX (`residual_nodejax.py`) | Flax NNX (`residual_nnx.py`) | Equinox (`residual_equinox.py`) |
| :--- | :--- | :--- | :--- |
| Deterministic & stochastic nesting | yes | yes | yes |
| Graph / `StatefulLayer` state | yes (unified `state`) | yes (`nnx.Variable`) | yes (`eqx.nn.State`) |
| Recurrent / `(successor, output)` state | yes (same `state` contract) | no (`functional-state=NO`, `cell-interface=NO`) | no (`successor-state=NO`) |
| State intact if `x + out` raises | yes (pure return before bind) | no (`Variable` already mutated) | no (`State.set` invalidates old `State`) |
| Input-primed state & member methods | yes (`priming=yes`, `methods=yes`) | manual per constructor | manual per constructor |

In Flax NNX, `nnx.RNNCellBase` cells take `(carry, input) -> (carry, output)` and publish `initialize_carry`, whereas unary modules take `input -> output` and mutate `nnx.Variable` attributes. One `Residual` class cannot cover both without per-protocol adapters, and mutating a `Variable` inside `body(x)` corrupts module state if the outer `+` subsequently fails on a shape mismatch.

In Equinox, subclassing `eqx.nn.StatefulLayer` works with `eqx.nn.make_with_state` and `eqx.nn.BatchNorm`, yet rejects `(successor, output)` modules, which is the exact convention [`lift_equinox.py`](../lift/lift_equinox.py) is forced to adopt because `eqx.nn.StateIndex` breaks under `filter_vmap`. Similarly, calling `state.set(...)` inside `body` marks the input `State` stale before `x + out` runs.

In NodeJAX, `residual` is intentionally unary on `input` (`extra-arguments=NO` when a body expects extra runtime call fields unless a multi-argument wiring transform names which argument is the residual path), while every cyclic, primed, stochastic, and `Aux`-emitting unary `Node` shares one `Contract`.

Run the three columns directly:

```sh
python -m examples.comparisons.residual.residual_nodejax
python -m examples.comparisons.residual.residual_nnx
python -m examples.comparisons.residual.residual_equinox
```
