"""nodejax.nn: stock neural blocks with input shapes inferred.

Factories return Nodes that participate in the same shape resolution,
composition, and transform APIs whether or not they own params or state.

Organized into modular subfiles:
- linear: Linear, Projection, Reshape, flat
- activations: gelu, relu, silu, sigmoid, tanh, elu, leaky_relu,
  softplus, softmax, log_softmax, identity
- norm: LayerNorm, RMSNorm, L2Norm, BatchNorm, Whiten
- conv: Conv, ConvTranspose
- spatial: MaxPool, AvgPool, GlobalAvgPool, Upsample, Downsample
- attention: Attention, TransformerBlock, PosEmbed, tokens
- mlp: MLP, SwiGLU, MoE
- embeddings: OneHot, Embed, Unembed
- recurrent: RNN, GRU, MinGRU, LSTM
- stochastic: Dropout, DropPath, GaussianNoise
- filtering: EMA
"""

from nodejax.nn.linear import Linear, Projection, flat, Reshape
from nodejax.nn.activations import (
    elu, gelu, identity, leaky_relu, log_softmax, relu, sigmoid, silu,
    softmax, softplus, tanh,
)
from nodejax.nn.norm import LayerNorm, RMSNorm, L2Norm, BatchNorm, Whiten
from nodejax.nn.conv import Conv, ConvTranspose
from nodejax.nn.spatial import (
    MaxPool, AvgPool, GlobalAvgPool, Upsample, Downsample,
)
from nodejax.nn.attention import Attention, TransformerBlock, tokens, PosEmbed
from nodejax.nn.mlp import MLP, SwiGLU, MoE
from nodejax.nn.embeddings import OneHot, Embed, Unembed
from nodejax.nn.recurrent import RNN, GRU, MinGRU, LSTM
from nodejax.nn.stochastic import Dropout, DropPath, GaussianNoise
from nodejax.nn.filtering import EMA

__all__ = [
    'Linear', 'Projection', 'flat', 'Reshape',
    'gelu', 'relu', 'silu', 'sigmoid', 'tanh', 'elu', 'leaky_relu',
    'softplus', 'softmax', 'log_softmax', 'identity',
    'LayerNorm', 'RMSNorm', 'L2Norm', 'BatchNorm', 'Whiten',
    'Conv', 'ConvTranspose',
    'MaxPool', 'AvgPool', 'GlobalAvgPool', 'Upsample', 'Downsample',
    'Attention', 'TransformerBlock', 'tokens', 'PosEmbed',
    'MLP', 'SwiGLU', 'MoE',
    'OneHot', 'Embed', 'Unembed',
    'RNN', 'GRU', 'MinGRU', 'LSTM',
    'Dropout', 'DropPath', 'GaussianNoise',
    'EMA',
]
