"""Provides neural network Leaf and Composite factories with input shapes inferred at bind time."""

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
from nodejax.nn.mode import eval_mode

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
    'eval_mode',
]
