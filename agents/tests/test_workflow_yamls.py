import yaml
from pathlib import Path

_WF_DIR = Path(__file__).parent.parent.parent / ".github" / "workflows"


def _load(name: str) -> dict:
    return yaml.safe_load((_WF_DIR / name).read_text())


def test_bench_ecc_exists():
    assert (_WF_DIR / "bench-ecc.yml").exists()


def test_bench_scheduler_exists():
    assert (_WF_DIR / "bench-scheduler.yml").exists()


def test_bench_nccl_exists():
    assert (_WF_DIR / "bench-nccl.yml").exists()


def test_bench_ecc_runs_gate():
    wf = _load("bench-ecc.yml")
    steps = wf["jobs"]["bench"]["steps"]
    run_cmds = " ".join(s.get("run", "") for s in steps)
    assert "assert_gate.py" in run_cmds
    assert "f1_2h" in run_cmds


def test_bench_scheduler_runs_gate():
    wf = _load("bench-scheduler.yml")
    steps = wf["jobs"]["bench"]["steps"]
    run_cmds = " ".join(s.get("run", "") for s in steps)
    assert "assert_gate.py" in run_cmds
    assert "mfu_ratio" in run_cmds


def test_bench_nccl_runs_two_gates():
    wf = _load("bench-nccl.yml")
    steps = wf["jobs"]["bench"]["steps"]
    run_cmds = " ".join(s.get("run", "") for s in steps)
    assert "resumption_seconds" in run_cmds
    assert "job_restart_count" in run_cmds
    assert run_cmds.count("assert_gate.py") >= 2
