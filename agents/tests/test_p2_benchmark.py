import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchmarks" / "p2_scheduler_mfu"))


def test_bandwidth_model_nvlink_faster_than_pcie():
    from bandwidth_model import BandwidthModel
    nvl = BandwidthModel(bw_gbps=600.0, n_gpus=8)
    pcie = BandwidthModel(bw_gbps=25.0, n_gpus=8)
    assert nvl.all_reduce_ns(bytes_=2 * 175_000_000_000 * 4) < pcie.all_reduce_ns(bytes_=2 * 175_000_000_000 * 4)


def test_bandwidth_model_mfu_in_range():
    from bandwidth_model import BandwidthModel
    nvl = BandwidthModel(bw_gbps=600.0, n_gpus=8)
    mfu = nvl.mfu()
    assert 0.0 < mfu < 1.0


def test_mfu_ratio_exceeds_gate():
    from bandwidth_model import BandwidthModel, NEMESIS_BW_GBPS, K8S_BW_GBPS
    nemesis = BandwidthModel(bw_gbps=NEMESIS_BW_GBPS, n_gpus=8)
    k8s = BandwidthModel(bw_gbps=K8S_BW_GBPS, n_gpus=8)
    ratio = nemesis.mfu() / k8s.mfu()
    assert ratio >= 1.4, f"MFU ratio {ratio:.3f} < 1.4 hard gate"


def test_all_reduce_formula():
    from bandwidth_model import BandwidthModel
    # 2*(N-1)/N * bytes / (bw/8) in nanoseconds
    # N=2, bytes=8, bw=8 Gbps → 2*(1/2)*8 / (1e9/8) * 1e9 = 64 ns
    model = BandwidthModel(bw_gbps=8.0, n_gpus=2)
    expected_ns = 2 * (2 - 1) / 2 * 8 / (8e9 / 8) * 1e9
    assert abs(model.all_reduce_ns(8) - expected_ns) < 1e-3
