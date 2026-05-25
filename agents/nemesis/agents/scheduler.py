"""SchedulerAgent: Claude-driven topology-aware GPU placement."""
from __future__ import annotations

import logging
from typing import Any

import anthropic
import grpc

from nemesis.grpc import telemetry_pb2, topology_pb2, topology_pb2_grpc

log = logging.getLogger(__name__)

SCHEDULER_TOOLS = [
    {
        "name": "validate_topology",
        "description": (
            "Type-check a topology DSL string (e.g. 'TP8_NVL12') without reserving GPUs. "
            "Always call this before schedule_job."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "topology_dsl": {"type": "string"},
                "gpu_count": {"type": "integer"},
            },
            "required": ["job_id", "topology_dsl", "gpu_count"],
        },
    },
    {
        "name": "schedule_job",
        "description": "Reserve GPUs for a job according to its topology spec. Returns assigned gpu_ids.",
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "topology_dsl": {"type": "string"},
                "gpu_count": {"type": "integer"},
            },
            "required": ["job_id", "topology_dsl", "gpu_count"],
        },
    },
    {
        "name": "release_job",
        "description": "Release GPU reservation for a completed or failed job.",
        "input_schema": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
    },
    {
        "name": "get_topology",
        "description": "Return live cluster topology: healthy GPUs, link bandwidth, InfiniBand hop counts.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]

SYSTEM_PROMPT = (
    "You are the NEMESIS Scheduler Agent. Place distributed training jobs on GPU clusters "
    "with topology-aware optimization.\n\n"
    "Rules:\n"
    "1. Always call validate_topology before schedule_job.\n"
    "2. Prefer NVLink-connected placements for tensor-parallel dimensions.\n"
    "3. Prefer low-hop InfiniBand paths for pipeline-parallel dimensions.\n"
    "4. If validate returns errors, do not call schedule_job — report the errors instead."
)


class SchedulerAgent:
    def __init__(self, substrate_addr: str, model: str = "claude-sonnet-4-6") -> None:
        self._addr = substrate_addr
        self._model = model
        self._client = anthropic.Anthropic()

    def place(self, job_id: str, topology_dsl: str, gpu_count: int) -> list[str]:
        """Ask Claude to validate and schedule a job. Returns assigned gpu_ids."""
        channel = grpc.insecure_channel(self._addr)
        stub = topology_pb2_grpc.SchedulerServiceStub(channel)
        try:
            return self._agentic_loop(stub, job_id, topology_dsl, gpu_count)
        finally:
            channel.close()

    def _agentic_loop(self, stub: Any, job_id: str, topology_dsl: str, gpu_count: int) -> list[str]:
        messages: list[dict] = [
            {
                "role": "user",
                "content": (
                    f"Place job '{job_id}' with topology '{topology_dsl}' "
                    f"requiring {gpu_count} GPUs."
                ),
            }
        ]

        while True:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=SCHEDULER_TOOLS,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                log.info("Scheduler completed for %s", job_id)
                return []

            tool_results = []
            gpu_ids: list[str] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = self._dispatch_tool(stub, block.name, block.input)
                if block.name == "schedule_job" and isinstance(result, list):
                    gpu_ids = result
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })

            if gpu_ids:
                return gpu_ids

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

    def _dispatch_tool(self, stub: Any, name: str, inputs: dict) -> Any:
        if name == "validate_topology":
            r = stub.Validate(topology_pb2.JobSpec(**inputs))
            return {"valid": r.valid, "errors": list(r.errors)}
        if name == "schedule_job":
            r = stub.Schedule(topology_pb2.JobSpec(**inputs))
            return list(r.gpu_ids) if r.placed else {"error": r.rejection_reason}
        if name == "release_job":
            stub.Release(topology_pb2.ReleaseRequest(job_id=inputs["job_id"]))
            return {"released": True}
        if name == "get_topology":
            r = stub.GetTopology(telemetry_pb2.Void())
            return {
                "nodes": [{"gpu_id": n.gpu_id, "healthy": n.healthy} for n in r.nodes],
                "edge_count": len(r.edges),
            }
        return {"error": f"unknown tool: {name}"}
