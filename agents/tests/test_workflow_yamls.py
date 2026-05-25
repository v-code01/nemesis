import yaml
from pathlib import Path

_WF_DIR = Path(__file__).parent.parent.parent / ".github" / "workflows"
assert _WF_DIR.is_dir(), f"Workflows directory not found at {_WF_DIR}"


def _load(name: str) -> dict:
    return yaml.safe_load((_WF_DIR / name).read_text())


def _gate_steps(steps: list) -> list[str]:
    """Return the run string of every step that invokes assert_gate.py."""
    return [s["run"] for s in steps if isinstance(s.get("run"), str) and "assert_gate.py" in s["run"]]


def test_bench_ecc_exists():
    assert (_WF_DIR / "bench-ecc.yml").exists()


def test_bench_scheduler_exists():
    assert (_WF_DIR / "bench-scheduler.yml").exists()


def test_bench_nccl_exists():
    assert (_WF_DIR / "bench-nccl.yml").exists()


def test_bench_ecc_runs_gate():
    wf = _load("bench-ecc.yml")
    steps = wf["jobs"]["bench"]["steps"]
    gates = _gate_steps(steps)
    assert len(gates) >= 1, "bench-ecc.yml has no assert_gate.py step"
    assert any("f1_2h" in g for g in gates), "bench-ecc.yml gate does not check f1_2h"
    assert any("0.90" in g for g in gates), "bench-ecc.yml gate threshold is not 0.90"


def test_bench_scheduler_runs_gate():
    wf = _load("bench-scheduler.yml")
    steps = wf["jobs"]["bench"]["steps"]
    gates = _gate_steps(steps)
    assert len(gates) >= 1, "bench-scheduler.yml has no assert_gate.py step"
    assert any("mfu_ratio" in g for g in gates), "bench-scheduler.yml gate does not check mfu_ratio"
    assert any("1.4" in g for g in gates), "bench-scheduler.yml gate threshold is not 1.4"


def test_bench_nccl_runs_two_gates():
    wf = _load("bench-nccl.yml")
    steps = wf["jobs"]["bench"]["steps"]
    gates = _gate_steps(steps)
    assert len(gates) >= 2, f"bench-nccl.yml has {len(gates)} gate step(s), expected >= 2"
    assert any("resumption_seconds" in g for g in gates), "bench-nccl.yml missing resumption_seconds gate"
    assert any("job_restart_count" in g for g in gates), "bench-nccl.yml missing job_restart_count gate"
