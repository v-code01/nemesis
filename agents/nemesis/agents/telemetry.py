"""TelemetryAgent: polls cluster state, runs ECC TCN inference, publishes predictions."""
from __future__ import annotations

import asyncio
import collections
import logging
import time

import grpc
import numpy as np

from nemesis.grpc import telemetry_pb2, telemetry_pb2_grpc
from nemesis.models.ecc_predictor import EccPredictor, SEQ_LEN, THRESHOLD

log = logging.getLogger(__name__)

POLL_INTERVAL_S = 5.0


class TelemetryAgent:
    def __init__(
        self,
        substrate_addr: str,
        model_path: str | None = None,
        poll_interval: float = POLL_INTERVAL_S,
    ) -> None:
        self._addr = substrate_addr
        self._poll_interval = poll_interval
        self._model = EccPredictor.load(model_path) if model_path else EccPredictor()
        self._windows: dict[str, collections.deque[list[float]]] = {}

    def _append_sample(self, sample: object) -> None:
        gpu_id = sample.gpu_id  # type: ignore[attr-defined]
        buf = self._windows.setdefault(gpu_id, collections.deque(maxlen=SEQ_LEN))
        prev_corr = buf[-1][0] if buf else sample.ecc_correctable_rate  # type: ignore[attr-defined]
        prev_uncorr = buf[-1][1] if buf else sample.ecc_uncorrectable_rate  # type: ignore[attr-defined]
        buf.append([
            sample.ecc_correctable_rate,  # type: ignore[attr-defined]
            sample.ecc_uncorrectable_rate,  # type: ignore[attr-defined]
            sample.temperature_celsius,  # type: ignore[attr-defined]
            sample.sm_utilization,  # type: ignore[attr-defined]
            sample.memory_bandwidth_utilization,  # type: ignore[attr-defined]
            sample.nvlink_bandwidth_gbps,  # type: ignore[attr-defined]
            sample.ib_bandwidth_gbps,  # type: ignore[attr-defined]
            sample.ecc_correctable_rate - prev_corr,   # type: ignore[attr-defined]  # ecc_corr_delta
            sample.ecc_uncorrectable_rate - prev_uncorr,  # type: ignore[attr-defined]  # ecc_uncorr_delta
        ])

    def _should_publish(self, gpu_id: str, window: np.ndarray) -> bool:
        _, p2h, _ = self._model.infer(window)
        return p2h > THRESHOLD

    async def run(self) -> None:
        async with grpc.aio.insecure_channel(self._addr) as channel:
            stub = telemetry_pb2_grpc.TelemetryServiceStub(channel)
            while True:
                try:
                    await self._poll_and_infer(stub)
                except grpc.aio.AioRpcError as exc:
                    # sleep-after-catch is intentional: back off before retrying
                    log.warning("gRPC poll error: %s", exc)
                await asyncio.sleep(self._poll_interval)

    async def _poll_and_infer(self, stub: telemetry_pb2_grpc.TelemetryServiceStub) -> None:
        snapshot = await stub.GetClusterState(telemetry_pb2.Void())
        now_ns = time.time_ns()

        for sample in snapshot.latest:
            self._append_sample(sample)
            buf = self._windows.get(sample.gpu_id, collections.deque())
            if len(buf) < SEQ_LEN:
                continue

            window = np.array(buf, dtype=np.float32)
            if not self._should_publish(sample.gpu_id, window):
                continue

            evidence = self._model.explain(window)
            _, p2h, _ = self._model.infer(window)
            event = telemetry_pb2.HardwareEvent(
                kind=telemetry_pb2.HardwareEvent.HARDWARE_FAILURE_PREDICTED,
                gpu_id=sample.gpu_id,
                confidence=p2h,
                predicted_failure_ns=now_ns + int(2 * 3600 * 1e9),
                event_ns=now_ns,
                evidence=evidence,
            )
            await stub.PublishEvent(event)
            log.info("HARDWARE_FAILURE_PREDICTED gpu=%s p2h=%.3f", sample.gpu_id, p2h)
