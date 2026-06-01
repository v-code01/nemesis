"""P3 benchmark: start nemesis-sim, register a job, trigger ShrinkCommunicator, measure resumption time."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import grpc

_ROOT = Path(__file__).parent.parent.parent
# Monorepo: agents package lives at _ROOT/agents, not installed system-wide in CI
sys.path.insert(0, str(_ROOT / "agents"))

from nemesis.grpc import healer_pb2, healer_pb2_grpc, telemetry_pb2

_SIM_BIN = _ROOT / "sim" / "target" / "release" / "nemesis-sim"
_SIM_PORT = 50052
# Sim binds to [::1]:{port} — use explicit IPv6 loopback so gRPC doesn't try 127.0.0.1 first
_SIM_ADDR = f"[::1]:{_SIM_PORT}"
_SIM_STARTUP_TIMEOUT_S = 30.0
_SIM_POLL_INTERVAL_S = 0.3
_SIM_SHUTDOWN_TIMEOUT_S = 10


def _wait_for_sim(stub: healer_pb2_grpc.HealerServiceStub, deadline: float) -> None:
    while time.monotonic() < deadline:
        try:
            stub.ListPlaybooks(telemetry_pb2.Void(), timeout=1.0)
            return
        except grpc.RpcError:
            time.sleep(_SIM_POLL_INTERVAL_S)
    raise TimeoutError(f"nemesis-sim did not start within {_SIM_STARTUP_TIMEOUT_S}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if not _SIM_BIN.exists():
        print(f"ERROR: sim binary not found at {_SIM_BIN}. Run: make build", file=sys.stderr)
        sys.exit(1)

    # Sim only accepts --config <yaml>; individual flags are silently ignored.
    # time_scale=100 compresses 2h of scenario time into ~72s of wall time.
    sim_cfg = (
        f"seed: {args.seed}\n"
        f"time_scale: 100.0\n"
        f"port: {_SIM_PORT}\n"
        "cluster:\n"
        "  gpu_count: 8\n"
        "  topology: nvlink_full\n"
        "  nvlink_gbps: 600.0\n"
        "  ib_gbps: 200.0\n"
    )
    cfg_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    cfg_file.write(sim_cfg)
    cfg_file.close()

    sim = subprocess.Popen(
        [str(_SIM_BIN), "--config", cfg_file.name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        with grpc.insecure_channel(_SIM_ADDR) as channel:
            healer_stub = healer_pb2_grpc.HealerServiceStub(channel)
            _wait_for_sim(healer_stub, time.monotonic() + _SIM_STARTUP_TIMEOUT_S)

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

            # job_restart_count: 0 on clean shrink, 1 if gRPC reports failure
            result = {
                "resumption_seconds": round(shrink.duration_ns / 1e9, 3),
                "job_restart_count": 0 if shrink.success else 1,
                "active_rank_count": shrink.active_rank_count,
                "success": shrink.success,
                "seed": args.seed,
            }
    finally:
        sim.terminate()
        try:
            sim.wait(timeout=_SIM_SHUTDOWN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            sim.kill()
        os.unlink(cfg_file.name)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
