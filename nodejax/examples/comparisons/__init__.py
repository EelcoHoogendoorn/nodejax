"""The same model, four ways: hands-on framework comparisons.

One test-time-training RNN (adapted weights over a live hidden state),
written in raw jax by hand, in flax nnx, in pytorch, and in nodejax
(the nodejax version is `examples/comparisons/ttt_nodejax.py`: one `ttt`
wrapper over the one contract). The point is not benchmarking; it is
reading the four side by side and counting what each framework makes
you say. The flax and torch versions import their frameworks; install
those separately to run them.

`imu_equinox.py` and `imu_flax.py` are a second comparison with a
different moral: the compositional IMU from `imu_nodejax.py`
rebuilt in idiomatic equinox and in flax nnx, showing where each
framework puts the state work (a hand-written container and threading
step there, per-boundary split/merge ceremony here) that nodejax
derives from the contract.
"""
