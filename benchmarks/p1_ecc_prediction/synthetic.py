"""Synthetic ECC failure dataset for P1 benchmark CI mode."""
import numpy as np

_SEQ = 600
_FEAT = 9


def _healthy(rng: np.random.Generator) -> np.ndarray:
    w = np.zeros((_SEQ, _FEAT), dtype=np.float32)
    w[:, 0] = rng.exponential(0.01, _SEQ)           # ecc_corr_rate
    w[:, 1] = 0.0                                    # ecc_uncorr_rate
    w[:, 2] = rng.uniform(50, 70, _SEQ)             # temp
    w[:, 3] = rng.uniform(0.7, 0.95, _SEQ)          # sm_util
    w[:, 4] = rng.uniform(0.5, 0.9, _SEQ)           # mem_bw
    w[:, 5] = rng.uniform(400, 600, _SEQ)           # nvlink_bw
    w[:, 6] = rng.uniform(150, 200, _SEQ)           # ib_bw
    w[1:, 7] = np.diff(w[:, 0])                     # ecc_corr_delta
    w[:, 8] = 0.0                                    # ecc_uncorr_delta
    return w


def _failing(rng: np.random.Generator, ramp_offset: int) -> np.ndarray:
    """Ramp up ECC errors in the last (600 - ramp_offset) steps."""
    w = _healthy(rng)
    start = max(0, _SEQ - ramp_offset)
    n = _SEQ - start
    ramp = np.linspace(0, 10.0, n, dtype=np.float32)
    w[start:, 0] = ramp + rng.exponential(0.1, n).astype(np.float32)
    w[start:, 2] = np.linspace(65, 90, n, dtype=np.float32)   # thermal
    w[start:, 3] = np.linspace(0.9, 0.3, n, dtype=np.float32) # SM throttle
    w[1:, 7] = np.diff(w[:, 0])
    return w


def generate_dataset(
    n_healthy: int = 10_000,
    n_failing: int = 1_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) where X.shape=(N,600,9), y.shape=(N,3) with 0/1 labels."""
    rng = np.random.default_rng(seed)
    windows, labels = [], []

    for _ in range(n_healthy):
        windows.append(_healthy(rng))
        labels.append([0.0, 0.0, 0.0])

    # Failure within 1h — ramp over last 360 steps (all horizons positive)
    for _ in range(n_failing):
        windows.append(_failing(rng, ramp_offset=360))
        labels.append([1.0, 1.0, 1.0])

    # Failure within 2h but not 1h — ramp starts earlier
    for _ in range(n_failing):
        windows.append(_failing(rng, ramp_offset=500))
        labels.append([0.0, 1.0, 1.0])

    # Failure within 3h but not 2h
    for _ in range(n_failing):
        windows.append(_failing(rng, ramp_offset=580))
        labels.append([0.0, 0.0, 1.0])

    X = np.stack(windows)
    y = np.array(labels, dtype=np.float32)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]
