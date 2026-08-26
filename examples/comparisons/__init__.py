"""Runnable formulations of shared problems across model libraries.

The examples compare where each framework records params, evolving state,
transform axes, reset behavior, and sharing. They use the framework's current
native mechanisms, including Equinox State, Flax Linen lifted transforms,
Flax NNX graph-aware transforms, Haiku transformed functions, and PyTorch
modules with `torch.func` where functional parameter updates are needed.

The chunk family has the strongest numerical checks: every implementation
must reproduce the same plain-JAX references across nested state lifetimes and
training. The tower combines depth, time, ensembling, and MAML as a structural
stress test. The tie, mode, generics, IMU, and TTT families isolate narrower
design choices.

These are source comparisons, not performance rankings. A result from one file
supports a claim about that formulation, not a universal limit on its
framework.
"""
