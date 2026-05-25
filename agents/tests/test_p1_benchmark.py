import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchmarks" / "p1_ecc_prediction"))

import numpy as np


def test_generate_dataset_shapes():
    from synthetic import generate_dataset
    X, y = generate_dataset(n_healthy=100, n_failing=20, seed=42)
    assert X.ndim == 3
    assert X.shape[1] == 600
    assert X.shape[2] == 9
    assert y.shape == (X.shape[0], 3)


def test_generate_dataset_label_range():
    from synthetic import generate_dataset
    X, y = generate_dataset(n_healthy=100, n_failing=20, seed=42)
    assert set(y.flatten().tolist()).issubset({0.0, 1.0})


def test_healthy_windows_have_low_ecc():
    from synthetic import generate_dataset
    X, y = generate_dataset(n_healthy=200, n_failing=0, seed=0)
    # All labels are 0 — healthy windows
    assert (y == 0).all()
    # ECC correctable rate stays low
    assert X[:, :, 0].max() < 5.0


def test_failing_windows_have_positive_labels():
    from synthetic import generate_dataset
    X, y = generate_dataset(n_healthy=0, n_failing=50, seed=0)
    # At least some windows have positive labels in the 2h column
    assert y[:, 1].sum() > 0


def test_compute_f1_perfect():
    import importlib.util, types
    # Import compute_f1 directly without running main
    spec = importlib.util.spec_from_file_location(
        "run",
        Path(__file__).parent.parent.parent / "benchmarks" / "p1_ecc_prediction" / "run.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    preds = np.array([[0.9, 0.9, 0.9], [0.1, 0.1, 0.1]])
    labels = np.array([[1, 1, 1], [0, 0, 0]], dtype=np.float32)
    f1 = mod.compute_f1(preds, labels, horizon=1)
    assert abs(f1 - 1.0) < 1e-6
