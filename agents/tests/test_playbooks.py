import yaml
from pathlib import Path

_PLAYBOOK_DIR = Path(__file__).parent.parent / "playbooks"

_REQUIRED_NAMES = [
    "ecc_isolation",
    "thermal_throttle_recovery",
    "nvlink_reset",
    "spot_preemption_checkpoint",
    "topology_rebinding",
    "dram_bandwidth_degradation",
    "sm_utilization_anomaly",
    "ib_congestion_recovery",
    "communicator_deadlock_recovery",
    "multi_gpu_correlated_failure",
    "numa_rebalance",
    "gradient_explosion_isolation",
    "checkpoint_corruption_recovery",
    "network_partition",
    "power_capping_event",
    "driver_hang_recovery",
    "pcie_error_recovery",
    "xid_error_triage",
    "compute_timeout",
    "memory_fragmentation_recovery",
]


def test_all_20_playbooks_exist():
    missing = []
    for name in _REQUIRED_NAMES:
        p = _PLAYBOOK_DIR / f"{name}.yaml"
        if not p.exists():
            missing.append(name)
    assert not missing, f"Missing playbooks: {missing}"


def test_all_playbooks_valid_yaml():
    for path in _PLAYBOOK_DIR.glob("*.yaml"):
        data = yaml.safe_load(path.read_text())
        assert data is not None, f"{path.name} is empty"


def test_all_playbooks_have_required_fields():
    required = {"name", "description", "trigger", "steps"}
    for path in _PLAYBOOK_DIR.glob("*.yaml"):
        data = yaml.safe_load(path.read_text())
        assert isinstance(data, dict), f"{path.name} top-level must be a mapping, got {type(data).__name__}"
        missing = required - set(data.keys())
        assert not missing, f"{path.name} missing fields: {missing}"


def test_all_playbooks_have_nonempty_steps():
    for path in _PLAYBOOK_DIR.glob("*.yaml"):
        data = yaml.safe_load(path.read_text())
        assert len(data["steps"]) >= 1, f"{path.name} has no steps"


def test_all_steps_have_action_key():
    for path in _PLAYBOOK_DIR.glob("*.yaml"):
        data = yaml.safe_load(path.read_text())
        for i, step in enumerate(data["steps"]):
            assert isinstance(step, dict), f"{path.name} step {i} is not a mapping"
            assert "action" in step, f"{path.name} step {i} missing 'action' key"


def test_playbook_count():
    names = {p.stem for p in _PLAYBOOK_DIR.glob("*.yaml")}
    extra = names - set(_REQUIRED_NAMES)
    assert not extra, f"Unexpected playbooks: {extra}"
    assert len(names) == 20, f"Expected 20 playbooks, found {len(names)}"
