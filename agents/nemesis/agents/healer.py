"""HealerAgent: subscribes to failure predictions, calls Claude, executes NCCL shrink."""
from __future__ import annotations

import logging
import time
from typing import Any

import anthropic
import grpc
import grpc.aio

from nemesis.grpc import (
    healer_pb2,
    healer_pb2_grpc,
    telemetry_pb2,
    telemetry_pb2_grpc,
)

log = logging.getLogger(__name__)

HEALER_TOOLS = [
    {
        "name": "get_active_jobs",
        "description": "List running jobs and their GPU/communicator assignments.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "shrink_communicator",
        "description": (
            "Exclude a failed GPU from a live NCCL communicator. "
            "Blocks training for <30s. Irreversible until expand."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "communicator_id": {"type": "string"},
                "job_id": {"type": "string"},
                "exclude_ranks": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["communicator_id", "job_id", "exclude_ranks"],
        },
    },
    {
        "name": "expand_communicator",
        "description": "Add replacement GPU to an active NCCL communicator.",
        "input_schema": {
            "type": "object",
            "properties": {
                "communicator_id": {"type": "string"},
                "job_id": {"type": "string"},
                "new_gpu_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["communicator_id", "job_id", "new_gpu_ids"],
        },
    },
    {
        "name": "execute_playbook",
        "description": "Run a named playbook from the library.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "parameters": {"type": "object"},
            },
            "required": ["name"],
        },
    },
]

SYSTEM_PROMPT = (
    "You are the NEMESIS Healer Agent. You respond to hardware failure predictions.\n\n"
    "Decision tiers (confidence = p2h from TelemetryAgent):\n"
    "- confidence >= 0.95 AND time_to_failure <= 30min: call shrink_communicator NOW\n"
    "- confidence >= 0.85 AND time_to_failure <= 2h: monitor; call execute_playbook to prepare\n"
    "- confidence < 0.85: log evidence, take no action\n\n"
    "shrink_communicator is irreversible until expand — reason carefully before calling it."
)

_MAX_LOOP_ITERATIONS = 10


class HealerAgent:
    def __init__(self, substrate_addr: str, model: str = "claude-sonnet-4-6") -> None:
        self._addr = substrate_addr
        self._model = model
        self._client = anthropic.Anthropic()
        self._jobs: dict[str, str] = {}  # job_id → communicator_id

    def register_job(self, job_id: str, rank: int, world_size: int) -> str:
        with grpc.insecure_channel(self._addr) as channel:
            stub = healer_pb2_grpc.HealerServiceStub(channel)
            resp = stub.RegisterJob(healer_pb2.RegisterJobRequest(
                job_id=job_id, rank=rank, world_size=world_size,
            ))
        self._jobs[job_id] = resp.communicator_id
        return resp.communicator_id

    async def subscribe_and_react(self) -> None:
        async with grpc.aio.insecure_channel(self._addr) as channel:
            tel_stub = telemetry_pb2_grpc.TelemetryServiceStub(channel)
            heal_stub = healer_pb2_grpc.HealerServiceStub(channel)
            filt = telemetry_pb2.EventFilter(
                kinds=[telemetry_pb2.HardwareEvent.HARDWARE_FAILURE_PREDICTED],
            )
            async for event in tel_stub.SubscribeEvents(filt):
                await self._react(event, heal_stub)

    async def _react(self, event: Any, heal_stub: Any) -> None:
        now_ns = time.time_ns()
        ttf_s = (event.predicted_failure_ns - now_ns) / 1e9
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    f"HARDWARE_FAILURE_PREDICTED for GPU {event.gpu_id}.\n"
                    f"confidence: {event.confidence:.3f}\n"
                    f"time_to_failure_seconds: {ttf_s:.0f}\n"
                    f"evidence: {dict(event.evidence)}\n"
                    f"active_jobs: {self._jobs}\n"
                    "Take appropriate action based on the decision tiers."
                ),
            }
        ]

        for _ in range(_MAX_LOOP_ITERATIONS):
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=HEALER_TOOLS,  # type: ignore[arg-type]
                messages=messages,  # type: ignore[arg-type]
            )

            if response.stop_reason in ("end_turn", "max_tokens"):
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = await self._dispatch_tool(block.name, block.input, heal_stub)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

    async def _dispatch_tool(self, name: str, inputs: dict[str, Any], heal_stub: Any) -> Any:
        if name == "get_active_jobs":
            return {"jobs": [{"job_id": j, "comm_id": c} for j, c in self._jobs.items()]}
        if name == "shrink_communicator":
            r = await heal_stub.ShrinkCommunicator(healer_pb2.ShrinkRequest(
                communicator_id=inputs["communicator_id"],
                job_id=inputs["job_id"],
                exclude_ranks=inputs["exclude_ranks"],
            ))
            return {"success": r.success, "duration_ns": r.duration_ns, "active": r.active_rank_count}
        if name == "expand_communicator":
            r = await heal_stub.ExpandCommunicator(healer_pb2.ExpandRequest(
                communicator_id=inputs["communicator_id"],
                job_id=inputs["job_id"],
                new_gpu_ids=inputs["new_gpu_ids"],
            ))
            return {"success": r.success, "active": r.active_rank_count}
        if name == "execute_playbook":
            r = await heal_stub.ExecutePlaybook(healer_pb2.PlaybookRequest(
                name=inputs["name"],
                parameters=inputs.get("parameters", {}),
            ))
            return {"success": r.success, "actions": list(r.actions_taken)}
        return {"error": f"unknown tool: {name}"}
