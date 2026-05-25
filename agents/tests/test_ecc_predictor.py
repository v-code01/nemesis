import numpy as np
import torch
from nemesis.models.ecc_predictor import (
    EccPredictor, TemporalBlock, TemporalConvNet,
    N_FEATURES, SEQ_LEN, THRESHOLD, FEATURE_NAMES,
)


def test_feature_names_count():
    assert len(FEATURE_NAMES) == N_FEATURES == 9


def test_temporal_block_output_shape():
    block = TemporalBlock(in_channels=9, out_channels=64, kernel_size=7, dilation=1)
    x = torch.randn(2, 9, 600)
    out = block(x)
    assert out.shape == (2, 64, 600), f"expected (2, 64, 600), got {out.shape}"


def test_temporal_conv_net_output_shape():
    tcn = TemporalConvNet(num_inputs=9, channels=[64, 128, 128, 64], kernel_size=7)
    x = torch.randn(4, 9, 600)
    out = tcn(x)
    assert out.shape == (4, 64, 600)


def test_ecc_predictor_forward_shape():
    model = EccPredictor()
    x = torch.randn(8, 600, 9)
    out = model(x)
    assert out.shape == (8, 3)


def test_ecc_predictor_output_in_01():
    model = EccPredictor()
    x = torch.randn(4, 600, 9)
    out = model(x)
    assert (out >= 0).all() and (out <= 1).all()


def test_infer_returns_three_floats():
    model = EccPredictor()
    window = np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32)
    p1h, p2h, p3h = model.infer(window)
    assert isinstance(p1h, float)
    assert isinstance(p2h, float)
    assert isinstance(p3h, float)
    assert 0.0 <= p1h <= 1.0
    assert 0.0 <= p2h <= 1.0
    assert 0.0 <= p3h <= 1.0


def test_explain_sums_to_one():
    model = EccPredictor()
    window = np.random.randn(SEQ_LEN, N_FEATURES).astype(np.float32)
    evidence = model.explain(window)
    assert set(evidence.keys()) == set(FEATURE_NAMES)
    total = sum(evidence.values())
    assert abs(total - 1.0) < 1e-4, f"importances sum to {total}, expected 1.0"


def test_save_load_roundtrip(tmp_path):
    model = EccPredictor()
    path = tmp_path / "ecc.pt"
    model.save(path)
    loaded = EccPredictor.load(path)
    window = np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32)
    p_orig = model.infer(window)
    p_loaded = loaded.infer(window)
    assert p_orig == p_loaded


def test_threshold_constant():
    assert THRESHOLD == 0.85
