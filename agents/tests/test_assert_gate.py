import json
import subprocess
import sys
from pathlib import Path

_GATE = Path(__file__).parent.parent.parent / "benchmarks" / "assert_gate.py"


def _run(result: dict, metric: str, *, min_: float | None = None, max_: float | None = None, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    out = tmp_path / "result.json"
    out.write_text(json.dumps(result))
    cmd = [sys.executable, str(_GATE), "--result", str(out), "--metric", metric]
    if min_ is not None:
        cmd += ["--min", str(min_)]
    if max_ is not None:
        cmd += ["--max", str(max_)]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_pass_min(tmp_path):
    proc = _run({"f1_2h": 0.92}, "f1_2h", min_=0.90, tmp_path=tmp_path)
    assert proc.returncode == 0
    assert "PASS" in proc.stdout


def test_fail_min(tmp_path):
    proc = _run({"f1_2h": 0.85}, "f1_2h", min_=0.90, tmp_path=tmp_path)
    assert proc.returncode == 1
    assert "FAIL" in proc.stdout


def test_pass_max(tmp_path):
    proc = _run({"resumption_seconds": 12.5}, "resumption_seconds", max_=30.0, tmp_path=tmp_path)
    assert proc.returncode == 0


def test_fail_max(tmp_path):
    proc = _run({"resumption_seconds": 45.0}, "resumption_seconds", max_=30.0, tmp_path=tmp_path)
    assert proc.returncode == 1


def test_output_format_contains_metric(tmp_path):
    proc = _run({"mfu_ratio": 1.87}, "mfu_ratio", min_=1.4, tmp_path=tmp_path)
    assert "mfu_ratio" in proc.stdout
    assert "1.87" in proc.stdout
    assert ">= 1.4" in proc.stdout


def test_missing_metric_exits_2(tmp_path):
    proc = _run({"other_metric": 1.0}, "nonexistent", min_=0.5, tmp_path=tmp_path)
    assert proc.returncode == 2


def test_missing_result_file_exits_2(tmp_path):
    cmd = [sys.executable, str(_GATE), "--result", str(tmp_path / "nope.json"), "--metric", "x", "--min", "0"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 2


def test_no_threshold_exits_2(tmp_path):
    out = tmp_path / "r.json"
    out.write_text(json.dumps({"x": 1.0}))
    cmd = [sys.executable, str(_GATE), "--result", str(out), "--metric", "x"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 2
