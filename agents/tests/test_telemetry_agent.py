from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest
from nemesis.models.ecc_predictor import SEQ_LEN, N_FEATURES
from nemesis.agents.telemetry import TelemetryAgent, POLL_INTERVAL_S


def _make_sample(gpu_id: str, ecc_corr: float = 0.0) -> MagicMock:
    s = MagicMock()
    s.gpu_id = gpu_id
    s.ecc_correctable_rate = ecc_corr
    s.ecc_uncorrectable_rate = 0.0
    s.temperature_celsius = 60.0
    s.sm_utilization = 0.8
    s.memory_bandwidth_utilization = 0.5
    s.nvlink_bandwidth_gbps = 500.0
    s.ib_bandwidth_gbps = 180.0
    return s


def test_poll_interval_default():
    assert POLL_INTERVAL_S == 5.0


def test_window_accumulates_samples():
    agent = TelemetryAgent("localhost:50051", model_path=None)
    sample = _make_sample("gpu-0", ecc_corr=0.1)
    for _ in range(10):
        agent._append_sample(sample)
    assert len(agent._windows["gpu-0"]) == 10


def test_window_capped_at_seq_len():
    agent = TelemetryAgent("localhost:50051", model_path=None)
    sample = _make_sample("gpu-0")
    for _ in range(SEQ_LEN + 50):
        agent._append_sample(sample)
    assert len(agent._windows["gpu-0"]) == SEQ_LEN


def test_no_event_below_threshold():
    agent = TelemetryAgent("localhost:50051", model_path=None)
    with patch.object(agent._model, "infer", return_value=(0.1, 0.3, 0.5)):
        published = agent._should_publish("gpu-0", np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32))
    assert not published


def test_event_above_threshold():
    agent = TelemetryAgent("localhost:50051", model_path=None)
    with patch.object(agent._model, "infer", return_value=(0.9, 0.9, 0.9)):
        published = agent._should_publish("gpu-0", np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32))
    assert published


def test_ecc_delta_computed():
    agent = TelemetryAgent("localhost:50051", model_path=None)
    s1 = _make_sample("gpu-1", ecc_corr=1.0)
    s2 = _make_sample("gpu-1", ecc_corr=3.0)
    agent._append_sample(s1)
    agent._append_sample(s2)
    buf = agent._windows["gpu-1"]
    # delta is stored at index 7
    assert buf[-1][7] == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_poll_and_infer_no_publish_below_threshold():
    """_poll_and_infer must NOT call PublishEvent when p2h <= THRESHOLD."""
    agent = TelemetryAgent("localhost:50051", model_path=None)
    sample = _make_sample("gpu-2")

    # Pre-fill window so it's ready for inference
    for _ in range(SEQ_LEN):
        agent._append_sample(sample)

    snapshot = MagicMock()
    snapshot.latest = [sample]
    stub = AsyncMock()
    stub.GetClusterState = AsyncMock(return_value=snapshot)

    with patch.object(agent._model, "infer", return_value=(0.1, 0.3, 0.5)):
        await agent._poll_and_infer(stub)

    stub.PublishEvent.assert_not_called()


@pytest.mark.asyncio
async def test_poll_and_infer_publishes_above_threshold():
    """_poll_and_infer must call PublishEvent when p2h > THRESHOLD."""
    agent = TelemetryAgent("localhost:50051", model_path=None)
    sample = _make_sample("gpu-3")

    for _ in range(SEQ_LEN):
        agent._append_sample(sample)

    snapshot = MagicMock()
    snapshot.latest = [sample]
    stub = AsyncMock()
    stub.GetClusterState = AsyncMock(return_value=snapshot)
    stub.PublishEvent = AsyncMock()

    with patch.object(agent._model, "infer", return_value=(0.95, 0.95, 0.95)), \
         patch.object(agent._model, "explain", return_value={}):
        await agent._poll_and_infer(stub)

    stub.PublishEvent.assert_called_once()
