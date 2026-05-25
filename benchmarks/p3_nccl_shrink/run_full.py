"""P3 benchmark: start nemesis-sim, register a job, trigger ShrinkCommunicator, measure resumption time."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import grpc

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / "agents"))

from nemesis.grpc import healer_pb2, healer_pb2_grpc, telemetry_pb2

_SIM_BIN = _ROOT / "sim" / "target" / "release" / "nemesis-sim"
_SCENARIO = _ROOT / "sim" / "scenarios" / "ecc_escalation.yaml"
_SIM_PORT = 50052
_SIM_ADDR = f"localhost:{_SIM_PORT}"


def _wait_for_sim(stub: healer_pb2_grpc.HealerServiceStub, deadline: float) -> None:
    while time.monotonic() < deadline:
        try:
            stub.ListPlaybooks(telemetry_pb2.Void(), timeout=1.0)
            return
        except grpc.RpcError:
            time.sleep(0.3)
    raise TimeoutError("nemesis-sim did not start within 30s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if not _SIM_BIN.exists():
        print(f"ERROR: sim binary not found at {_SIM_BIN}. Run: make build", file=sys.stderr)
        sys.exit(1)

    sim = subprocess.Popen(
        [
            str(_SIM_BIN),
            "--scenario", str(_SCENARIO),
            "--seed", str(args.seed),
            "--time-scale", "100.0",
            "--port", str(_SIM_PORT),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        channel = grpc.insecure_channel(_SIM_ADDR)
        healer_stub = healer_pb2_grpc.HealerServiceStub(channel)
        _wait_for_sim(healer_stub, time.monotonic() + 30.0)

        reg = healer_stub.RegisterJob(healer_pb2.RegisterJobRequest(
            job_id="bench-p3-001", rank=0, world_size=8,
        ))
        comm_id = reg.communicator_id

        # gpu-3 is the degraded GPU in ecc_escalation scenario — exclude rank 3
        shrink = healer_stub.ShrinkCommunicator(healer_pb2.ShrinkRequest(
            communicator_id=comm_id,
            job_id="bench-p3-001",
            exclude_ranks=[3],
        ))

        result = {
            "resumption_seconds": round(shrink.duration_ns / 1e9, 3),
            "job_restart_count": 0 if shrink.success else 1,
            "active_rank_count": shrink.active_rank_count,
            "success": shrink.success,
            "seed": args.seed,
        }
        channel.close()
    finally:
        sim.terminate()
        sim.wait(timeout=10)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
