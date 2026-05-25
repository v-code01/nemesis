#!/usr/bin/env python3
"""Print Table 1 from the paper: all three benchmark results."""
import json
import sys
from pathlib import Path

_RESULTS_DIR = Path(__file__).parent.parent / "results"


def _load(name: str) -> dict[str, object]:
    p = _RESULTS_DIR / name
    if not p.exists():
        print(f"WARNING: result file not found: {p}", file=sys.stderr)
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        print(f"WARNING: malformed JSON in {p}: {exc}", file=sys.stderr)
        return {}


def main() -> None:
    ecc = _load("ecc.json")
    mfu = _load("mfu.json")
    nccl = _load("nccl.json")

    sep = "=" * 60
    print(sep)
    print("Table 1: NEMESIS Hard Gate Benchmark Results")
    print(sep)
    print("P1  ECC Prediction")
    print(f"    F1 @ 1h horizon : {ecc.get('f1_1h', 'N/A'):>8}   (gate: —)")
    print(f"    F1 @ 2h horizon : {ecc.get('f1_2h', 'N/A'):>8}   (gate: >= 0.90)")
    print(f"    F1 @ 3h horizon : {ecc.get('f1_3h', 'N/A'):>8}   (gate: —)")
    print()
    print("P2  Scheduler MFU")
    print(f"    MFU NEMESIS     : {mfu.get('mfu_nemesis', 'N/A'):>8}")
    print(f"    MFU k8s default : {mfu.get('mfu_k8s', 'N/A'):>8}")
    print(f"    MFU ratio       : {mfu.get('mfu_ratio', 'N/A'):>8}   (gate: >= 1.4x)")
    print()
    print("P3  NCCL Communicator Shrink")
    print(f"    Resumption (s)  : {nccl.get('resumption_seconds', 'N/A'):>8}   (gate: < 30s)")
    print(f"    Job restarts    : {nccl.get('job_restart_count', 'N/A'):>8}   (gate: = 0)")
    print(sep)


if __name__ == "__main__":
    main()
