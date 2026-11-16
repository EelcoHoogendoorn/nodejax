"""Shared trajectory and verification checks for the IMU sensor comparison."""

import numpy as np

DT = 0.01
RES = 0.05
DENSITY = 0.05
TAU = 1.0
DRIFT_DENSITY = 0.2


def trajectory() -> tuple[np.ndarray, np.ndarray]:
    """Generates timestamps and offset sinusoidal positions (`x[0] == 2.0`)."""
    t = np.arange(0.0, 2.0 * np.pi, DT)
    positions = (np.sin(t) + 2.0).astype(np.float32)
    return t, positions


def verify_imu(name: str, positions: np.ndarray, accel: np.ndarray,
               accel_quiet: np.ndarray) -> None:
    """Asserts priming, quantization grid, acceleration tracking, and quiet-mode parity."""
    pos = np.asarray(positions)
    acc = np.asarray(accel)
    quiet = np.asarray(accel_quiet)

    assert abs(float(acc[0])) < 1.0, f'{name}: unprimed first-step spike {acc[0]}'
    assert np.allclose(acc, np.round(acc / RES) * RES, atol=1e-5), f'{name}: off quantization grid'

    true_accel = np.gradient(np.gradient(pos, DT), DT)
    corr = float(np.corrcoef(acc[5:], true_accel[5:])[0, 1])
    assert corr > 0.9, f'{name}: poor acceleration correlation {corr}'

    exact = (pos[2:] - 2.0 * pos[1:-1] + pos[:-2]) / (DT ** 2)
    quantized = np.round(exact / RES) * RES
    assert np.allclose(quiet[2:], quantized, atol=1e-3), f'{name}: quiet trajectory mismatch'

    print(f'{name:10s} corr={corr:.3f} first={float(acc[0]):+.3f} quiet_exact=True', flush=True)
