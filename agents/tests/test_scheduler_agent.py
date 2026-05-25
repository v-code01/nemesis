from unittest.mock import MagicMock
from nemesis.agents.scheduler import SchedulerAgent, SCHEDULER_TOOLS, SYSTEM_PROMPT


def test_scheduler_tools_count():
    assert len(SCHEDULER_TOOLS) == 4


def test_scheduler_tool_names():
    names = {t["name"] for t in SCHEDULER_TOOLS}
    assert names == {"validate_topology", "schedule_job", "release_job", "get_topology"}


def test_all_tools_have_input_schema():
    for tool in SCHEDULER_TOOLS:
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"


def test_system_prompt_mentions_validate_before_schedule():
    assert "validate" in SYSTEM_PROMPT.lower()
    assert "schedule" in SYSTEM_PROMPT.lower()


def test_dispatch_get_topology():
    agent = SchedulerAgent("localhost:50051")
    stub = MagicMock()
    stub.GetTopology.return_value = MagicMock(
        nodes=[MagicMock(gpu_id="gpu-0", healthy=True)],
        edges=[],
    )
    result = agent._dispatch_tool(stub, "get_topology", {})
    assert "nodes" in result
    assert result["nodes"][0]["gpu_id"] == "gpu-0"


def test_dispatch_release_job():
    agent = SchedulerAgent("localhost:50051")
    stub = MagicMock()
    result = agent._dispatch_tool(stub, "release_job", {"job_id": "job-1"})
    assert result == {"released": True}
    stub.Release.assert_called_once()


def test_dispatch_unknown_tool():
    agent = SchedulerAgent("localhost:50051")
    result = agent._dispatch_tool(MagicMock(), "nonexistent", {})
    assert "error" in result


def test_dispatch_validate_topology():
    agent = SchedulerAgent("localhost:50051")
    stub = MagicMock()
    stub.Validate.return_value = MagicMock(valid=True, errors=[])
    result = agent._dispatch_tool(stub, "validate_topology", {
        "job_id": "j1", "topology_dsl": "TP8_NVL12", "gpu_count": 8,
    })
    assert result["valid"] is True
    assert result["errors"] == []


def test_dispatch_schedule_job_placed():
    agent = SchedulerAgent("localhost:50051")
    stub = MagicMock()
    stub.Schedule.return_value = MagicMock(placed=True, gpu_ids=["gpu-0", "gpu-1"])
    result = agent._dispatch_tool(stub, "schedule_job", {
        "job_id": "j1", "topology_dsl": "TP2_NVL2", "gpu_count": 2,
    })
    assert result == ["gpu-0", "gpu-1"]


def test_dispatch_schedule_job_rejected():
    agent = SchedulerAgent("localhost:50051")
    stub = MagicMock()
    stub.Schedule.return_value = MagicMock(placed=False, rejection_reason="insufficient GPUs")
    result = agent._dispatch_tool(stub, "schedule_job", {
        "job_id": "j2", "topology_dsl": "TP8_NVL12", "gpu_count": 8,
    })
    assert "error" in result
    assert result["error"] == "insufficient GPUs"
