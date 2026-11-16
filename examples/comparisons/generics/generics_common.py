"""The generics comparison: one deep architecture, configured late,
reconfigured after training, and read back.

The architecture, identical in every column: a committee of towers.
Each tower is an entry linear (4 -> width), `depth` blocks of
[linear (width -> width), then tanh(x / temperature)], and a readout
(width -> 1); the committee averages `members` towers. Four knobs
(width, depth, members, temperature) live at three different depths of
the composition.

Three exercises, identical in every column:
  1. Configure late: the architecture is defined once with nothing
     decided; each config in the grid becomes a trained variant.
  2. Reconfigure built: the trained first variant flips its deepest
     knob (temperature) and is evaluated again, weights kept.
  3. Read back: print the configuration as data, if the framework can.

The measure is the config threading tax: constructor parameters that
exist only to forward a value to a deeper level unmodified, counted by
hand per column, plus the honesty rules of the comparison files: the
threading is explicit, no globals smuggle configuration, and where a
framework forces something inline, the column says so directly.
"""

import os

import numpy as np

SMOKE = os.environ.get('NODEJAX_SMOKE') == '1'
DATA_KEY, PARAM_KEY = 0, 1
SAMPLES = 16 if SMOKE else 64
IN_DIM = 4
TRAIN_STEPS = 2 if SMOKE else 150
LR = 1e-2
RETEMPERED = 2.0                       # the after-training temperature flip

CONFIGS = [
    dict(width=16, depth=2, members=2, temperature=1.0),
    dict(width=32, depth=3, members=2, temperature=0.5),
    dict(width=8, depth=4, members=3, temperature=1.0),
]


def make_data() -> tuple:
    """Generates the shared regression dataset in NumPy."""
    rows = np.random.RandomState(DATA_KEY).randn(SAMPLES, IN_DIM).astype('float32')
    targets = np.sin(rows @ np.array([1.0, -2.0, 0.5, 3.0], dtype='float32'))
    return rows, targets.reshape(-1, 1)


def expected_parameters(config: dict) -> int:
    """Computes the architecture's parameter count from `config` alone as a
    parity check across columns."""
    width = config['width']
    entry = IN_DIM * width + width
    blocks = config['depth'] * (width * width + width)
    readout = width + 1
    return config['members'] * (entry + blocks + readout)


def report(name: str, rows: list, retempered_shift: float,
           threading_tax: int) -> None:
    """Asserts parameter-count parity, loss finiteness, and retempering
    sensitivity, then prints one summary line per variant plus the config
    threading census. `rows` carries (config, parameter_count, first_loss,
    last_loss)."""
    for config, parameters, first_loss, last_loss in rows:
        assert parameters == expected_parameters(config), (
            f'{name}: {parameters} parameters, the shared architecture '
            f'has {expected_parameters(config)}: not the same model')
        assert np.isfinite(first_loss) and np.isfinite(last_loss), (
            f'{name}: non-finite loss {first_loss} -> {last_loss}')
        print(f'[{name}] {config}: {parameters} params, '
              f'loss {first_loss:.3f} -> {last_loss:.3f}')
    assert np.isfinite(retempered_shift) and retempered_shift > 0.0, (
        f'{name}: the temperature flip changed nothing measurable')
    print(f'[{name}] retempered eval shifted by {retempered_shift:.4f} | '
          f'config threading tax: {threading_tax} parameters')
