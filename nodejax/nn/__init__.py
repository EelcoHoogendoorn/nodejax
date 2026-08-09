"""nodejax.nn — stock neural blocks with in-shapes inferred.

Capitalized names are units that own params or state; lowercase ones are
pure functions of their input, carrying neither.

Organized into modular subfiles:
- linear: Linear, flat
- activations: gelu, relu, silu, sigmoid, tanh
- norm: LayerNorm, BatchNorm, Whiten
- conv: Conv
- attention: Attention, Block, PosEmbed, tokens
- mlp: MLP, MoE
- embeddings: Embed, Unembed
- recurrent: RNN
- stochastic: Dropout
- filtering: EMA
"""

from nodejax.nn.linear import Linear, flat
from nodejax.nn.activations import gelu, relu, silu, sigmoid, tanh
from nodejax.nn.norm import LayerNorm, BatchNorm, Whiten
from nodejax.nn.conv import Conv
from nodejax.nn.attention import Attention, Block, tokens, PosEmbed
from nodejax.nn.mlp import MLP, MoE
from nodejax.nn.embeddings import Embed, Unembed
from nodejax.nn.recurrent import RNN
from nodejax.nn.stochastic import Dropout
from nodejax.nn.filtering import EMA

__all__ = [
    'Linear', 'flat',
    'gelu', 'relu', 'silu', 'sigmoid', 'tanh',
    'LayerNorm', 'BatchNorm', 'Whiten',
    'Conv',
    'Attention', 'Block', 'tokens', 'PosEmbed',
    'MLP', 'MoE',
    'Embed', 'Unembed',
    'RNN',
    'Dropout',
    'EMA',
]
