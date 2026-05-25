from unittest.mock import MagicMock, patch
from nemesis.hook import NemesisHook


def _make_hook(comm_id: str = "comm-001") -> tuple[NemesisHook, MagicMock]:
    mock_channel = MagicMock()
    mock_healer = MagicMock()
    mock_healer.RegisterJob.return_value = MagicMock(communicator_id=comm_id)
    mock_tel = MagicMock()
    mock_tel.SubscribeEvents.return_value = iter([])  # no events

    with patch("nemesis.hook.grpc.insecure_channel", return_value=mock_channel), \
         patch("nemesis.hook.healer_pb2_grpc.HealerServiceStub", return_value=mock_healer), \
         patch("nemesis.hook.telemetry_pb2_grpc.TelemetryServiceStub", return_value=mock_tel), \
         patch("nemesis.hook.threading.Thread"):
        hook = NemesisHook(job_id="job-1", substrate="localhost:50051", rank=0, world_size=8)

    return hook, mock_healer


def test_register_job_called_on_init():
    _, mock_healer = _make_hook("comm-007")
    mock_healer.RegisterJob.assert_called_once()
    call_args = mock_healer.RegisterJob.call_args[0][0]
    assert call_args.job_id == "job-1"
    assert call_args.rank == 0
    assert call_args.world_size == 8


def test_comm_id_stored():
    hook, _ = _make_hook("comm-abc")
    assert hook._comm_id == "comm-abc"


def test_step_noop_when_no_shrink_pending():
    hook, _ = _make_hook()
    pg = object()
    result = hook.step(step_idx=0, process_group=pg)
    assert result is pg


def test_step_returns_none_on_shrink():
    hook, mock_healer = _make_hook()
    mock_healer.ShrinkCommunicator.return_value = MagicMock(success=True, duration_ns=5_000_000_000)
    hook._shrink_pending.set()
    result = hook.step(step_idx=1, process_group=object())
    assert result is None
    mock_healer.ShrinkCommunicator.assert_called_once()


def test_step_keeps_pg_on_failed_shrink():
    hook, mock_healer = _make_hook()
    mock_healer.ShrinkCommunicator.return_value = MagicMock(success=False, duration_ns=0)
    hook._shrink_pending.set()
    pg = object()
    result = hook.step(step_idx=1, process_group=pg)
    assert result is pg


def test_listen_sets_pending_on_high_confidence():
    hook, _ = _make_hook()
    event = MagicMock()
    event.confidence = 0.97
    mock_tel = MagicMock()
    mock_tel.SubscribeEvents.return_value = iter([event])
    hook._tel = mock_tel
    hook._listen()
    assert hook._shrink_pending.is_set()


def test_listen_ignores_low_confidence_events():
    hook, _ = _make_hook()
    event = MagicMock()
    event.confidence = 0.80
    mock_tel = MagicMock()
    mock_tel.SubscribeEvents.return_value = iter([event])
    hook._tel = mock_tel
    hook._listen()
    assert not hook._shrink_pending.is_set()


def test_close_closes_channel():
    hook, _ = _make_hook()
    hook.close()
    hook._channel.close.assert_called_once()
