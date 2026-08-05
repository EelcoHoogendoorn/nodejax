"""The same model, four ways: hands-on framework comparisons.

One test-time-training RNN (adapted weights over a live hidden state),
written in raw jax by hand, in flax nnx, in pytorch, and in nodejax
(the nodejax version is `examples/meta_comparison.py`: one `ttt`
wrapper over the one contract). The point is not benchmarking; it is
reading the four side by side and counting what each framework makes
you say. The flax and torch versions import their frameworks; install
those separately to run them.

`imu_equinox.py` is a second comparison with a different moral: the
compositional IMU from `nodejax/tests/test_imu.py`, rebuilt in
idiomatic equinox, where the state container, its init composition,
and the threading step are written by hand.
"""
