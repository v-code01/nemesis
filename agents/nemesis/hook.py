"""NemesisHook: four-line training loop integration for NEMESIS-aware distributed training."""
from __future__ import annotations

import logging
import threading

import grpc

from nemesis.grpc import healer_pb2, healer_pb2_grpc, telemetry_pb2, telemetry_pb2_grpc

log = logging.getLogger(__name__)


class NemesisHook:
    """Attach to training loop via four lines:

        hook = NemesisHook(job_id="run-001", substrate="unix:///tmp/nemesis.sock")
        for step in range(max_steps):
            pg = hook.step(step, pg)
            loss = model(batch, process_group=pg)
    """

    def __init__(
        self,
        job_id: str,
        substrate: str,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self._job_id = job_id
        self._channel = grpc.insecure_channel(substrate)
        self._healer = healer_pb2_grpc.HealerServiceStub(self._channel)
        self._tel = telemetry_pb2_grpc.TelemetryServiceStub(self._channel)
        self._shrink_pending = threading.Event()

        resp = self._healer.RegisterJob(healer_pb2.RegisterJobRequest(
            job_id=job_id, rank=rank, world_size=world_size,
        ))
        self._comm_id = resp.communicator_id
        log.info("NemesisHook registered job=%s comm=%s", job_id, self._comm_id)

        t = threading.Thread(target=self._listen, daemon=True)
        t.start()

    def _listen(self) -> None:
        filt = telemetry_pb2.EventFilter(
            kinds=[telemetry_pb2.HardwareEvent.HARDWARE_FAILURE_PREDICTED],
        )
        try:
            for event in self._tel.SubscribeEvents(filt):
                if event.confidence >= 0.95:
                    log.info("NemesisHook: high-confidence prediction, priming shrink")
                    self._shrink_pending.set()
        except grpc.RpcError:
            pass  # channel closed on hook.close()

    def step(self, step_idx: int, process_group: object | None = None) -> object | None:
        """No-op in normal path (O(1) atomic check).
        Blocks <30s during shrink; returns None so caller rebuilds process group.
        """
        if not self._shrink_pending.is_set():
            return process_group

        self._shrink_pending.clear()
        log.info("NemesisHook: executing shrink at step %d", step_idx)
        result = self._healer.ShrinkCommunicator(healer_pb2.ShrinkRequest(
            communicator_id=self._comm_id,
            job_id=self._job_id,
            exclude_ranks=[],
        ))
        if result.success:
            log.info("NemesisHook: shrink complete in %.3fs", result.duration_ns / 1e9)
            return None
        log.warning("NemesisHook: shrink failed, continuing with existing pg")
        return process_group

    def close(self) -> None:
        self._channel.close()
