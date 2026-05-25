from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from nemesis.agents.healer import HealerAgent, HEALER_TOOLS, SYSTEM_PROMPT


def test_healer_tools_count():
    assert len(HEALER_TOOLS) == 4


def test_healer_tool_names():
    names = {t["name"] for t in HEALER_TOOLS}
    assert names == {"get_active_jobs", "shrink_communicator", "expand_communicator", "execute_playbook"}


def test_shrink_tool_is_annotated_irreversible():
    shrink_tool = next(t for t in HEALER_TOOLS if t["name"] == "shrink_communicator")
    assert "irreversible" in shrink_tool["description"].lower()


def test_system_prompt_has_decision_tiers():
    assert "0.95" in SYSTEM_PROMPT
    assert "0.85" in SYSTEM_PROMPT
    assert "30" in SYSTEM_PROMPT


def test_register_job_stores_comm_id():
    agent = HealerAgent("localhost:50051")
    with patch("nemesis.agents.healer.grpc.insecure_channel") as mock_ch:
        mock_stub = MagicMock()
        mock_stub.RegisterJob.return_value = MagicMock(communicator_id="comm-42")
        mock_ch.return_value.__enter__ = MagicMock(return_value=mock_ch.return_value)
        mock_ch.return_value.__exit__ = MagicMock(return_value=False)
        with patch("nemesis.agents.healer.healer_pb2_grpc.HealerServiceStub", return_value=mock_stub):
            comm_id = agent.register_job("job-1", rank=0, world_size=8)
    assert comm_id == "comm-42"
    assert agent._jobs["job-1"] == "comm-42"


@pytest.mark.asyncio
async def test_dispatch_get_active_jobs():
    agent = HealerAgent("localhost:50051")
    agent._jobs = {"job-abc": "comm-xyz"}
    result = await agent._dispatch_tool("get_active_jobs", {}, MagicMock())
    assert result["jobs"][0]["job_id"] == "job-abc"
    assert result["jobs"][0]["comm_id"] == "comm-xyz"


@pytest.mark.asyncio
async def test_dispatch_unknown_tool():
    agent = HealerAgent("localhost:50051")
    result = await agent._dispatch_tool("nonexistent", {}, MagicMock())
    assert "error" in result


@pytest.mark.asyncio
async def test_dispatch_shrink_communicator():
    agent = HealerAgent("localhost:50051")
    stub = AsyncMock()
    stub.ShrinkCommunicator.return_value = MagicMock(success=True, duration_ns=5_000_000_000, active_rank_count=7)
    result = await agent._dispatch_tool("shrink_communicator", {
        "communicator_id": "comm-1", "job_id": "job-1", "exclude_ranks": [3],
    }, stub)
    stub.ShrinkCommunicator.assert_called_once()
    assert result["success"] is True
    assert result["active"] == 7


@pytest.mark.asyncio
async def test_dispatch_expand_communicator():
    agent = HealerAgent("localhost:50051")
    stub = AsyncMock()
    stub.ExpandCommunicator.return_value = MagicMock(success=True, active_rank_count=8)
    result = await agent._dispatch_tool("expand_communicator", {
        "communicator_id": "comm-1", "job_id": "job-1", "new_gpu_ids": ["gpu-8"],
    }, stub)
    stub.ExpandCommunicator.assert_called_once()
    assert result["success"] is True
    assert result["active"] == 8


@pytest.mark.asyncio
async def test_dispatch_execute_playbook():
    agent = HealerAgent("localhost:50051")
    stub = AsyncMock()
    stub.ExecutePlaybook.return_value = MagicMock(success=True, actions_taken=["drain", "notify"])
    result = await agent._dispatch_tool("execute_playbook", {"name": "ecc_drain"}, stub)
    stub.ExecutePlaybook.assert_called_once()
    assert result["success"] is True
    assert "drain" in result["actions"]


@pytest.mark.asyncio
async def test_react_terminates_on_max_tokens():
    """_react must not loop forever when stop_reason is max_tokens."""
    agent = HealerAgent("localhost:50051")
    event = MagicMock()
    event.gpu_id = "gpu-0"
    event.confidence = 0.97
    event.predicted_failure_ns = 0
    event.evidence = {}

    response = MagicMock()
    response.stop_reason = "max_tokens"
    response.content = []

    mock_create = MagicMock(return_value=response)
    with patch.object(agent._client.messages, "create", mock_create):
        await agent._react(event, AsyncMock())

    # Loop must break on max_tokens — exactly one API call
    assert mock_create.call_count == 1
