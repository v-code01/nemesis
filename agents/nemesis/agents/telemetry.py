"""TelemetryAgent: polls cluster state, runs ECC TCN inference, publishes predictions."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

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
        model_path: Optional[str] = None,
        poll_interval: float = POLL_INTERVAL_S,
    ) -> None:
        self._addr = substrate_addr
        self._poll_interval = poll_interval
        self._model = EccPredictor.load(model_path) if model_path else EccPredictor()
        self._windows: dict[str, list[list[float]]] = {}

    def _append_sample(self, sample) -> None:
        gpu_id = sample.gpu_id
        buf = self._windows.setdefault(gpu_id, [])
        prev_corr = buf[-1][0] if buf else sample.ecc_correctable_rate
        prev_uncorr = buf[-1][1] if buf else sample.ecc_uncorrectable_rate
        row = [
            sample.ecc_correctable_rate,
            sample.ecc_uncorrectable_rate,
            sample.temperature_celsius,
            sample.sm_utilization,
            sample.memory_bandwidth_utilization,
            sample.nvlink_bandwidth_gbps,
            sample.ib_bandwidth_gbps,
            sample.ecc_correctable_rate - prev_corr,   # ecc_corr_delta
            sample.ecc_uncorrectable_rate - prev_uncorr,  # ecc_uncorr_delta
        ]
        buf.append(row)
        if len(buf) > SEQ_LEN:
            del buf[0]

    def _should_publish(self, _gpu_id: str, window: np.ndarray) -> bool:
        _, p2h, _ = self._model.infer(window)
        return p2h > THRESHOLD

    async def run(self) -> None:
        async with grpc.aio.insecure_channel(self._addr) as channel:
            stub = telemetry_pb2_grpc.TelemetryServiceStub(channel)
            while True:
                try:
                    await self._poll_and_infer(stub)
                except grpc.aio.AioRpcError as exc:
                    log.warning("gRPC poll error: %s", exc)
                await asyncio.sleep(self._poll_interval)

    async def _poll_and_infer(self, stub: telemetry_pb2_grpc.TelemetryServiceStub) -> None:
        snapshot = await stub.GetClusterState(telemetry_pb2.Void())
        now_ns = time.time_ns()

        for sample in snapshot.latest:
            self._append_sample(sample)
            buf = self._windows.get(sample.gpu_id, [])
            if len(buf) < SEQ_LEN:
                continue

            window = np.array(buf[-SEQ_LEN:], dtype=np.float32)
            _, p2h, _ = self._model.infer(window)
            if p2h <= THRESHOLD:
                continue

            evidence = self._model.explain(window)
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
