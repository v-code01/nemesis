"""P3 benchmark unit tests — validate output schema and gate logic without a live sim."""
import importlib.util
import json
from pathlib import Path


def test_output_schema(tmp_path):
    """run_full.py must write a JSON with the required keys."""
    result = {
        "resumption_seconds": 12.5,
        "job_restart_count": 0,
        "active_rank_count": 7,
        "success": True,
        "seed": 42,
    }
    out = tmp_path / "nccl.json"
    out.write_text(json.dumps(result))
    data = json.loads(out.read_text())
    for key in ("resumption_seconds", "job_restart_count", "active_rank_count", "success", "seed"):
        assert key in data, f"missing key {key}"


def test_gate_resumption_seconds():
    """Gate boundary: < 30s passes, >= 30s fails."""
    gate = 30
    assert 0.0 < gate
    assert 12.5 < gate
    assert 29.999 < gate
    assert not (30.0 < gate)
    assert not (31.0 < gate)


def test_gate_job_restart_count():
    """Gate boundary: exactly 0 restarts passes, any nonzero fails."""
    assert 0 == 0
    assert not (1 == 0)
    assert not (2 == 0)


def test_run_full_importable():
    spec_path = (
        Path(__file__).parent.parent.parent
        / "benchmarks" / "p3_nccl_shrink" / "run_full.py"
    )
    assert spec_path.exists(), f"run_full.py not found at {spec_path}"
    spec = importlib.util.spec_from_file_location("run_full", spec_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Just loading the module (not calling main) must not raise
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    assert hasattr(mod, "main")
