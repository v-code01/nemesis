"""P2 benchmark: analytical MFU comparison NEMESIS (NVLink) vs k8s default (PCIe)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bandwidth_model import BandwidthModel, NEMESIS_BW_GBPS, K8S_BW_GBPS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    n_gpus = 8
    nemesis = BandwidthModel(bw_gbps=NEMESIS_BW_GBPS, n_gpus=n_gpus)
    k8s = BandwidthModel(bw_gbps=K8S_BW_GBPS, n_gpus=n_gpus)

    mfu_n = nemesis.mfu()
    mfu_k = k8s.mfu()

    result = {
        "mfu_nemesis": round(mfu_n, 4),
        "mfu_k8s": round(mfu_k, 4),
        "mfu_ratio": round(mfu_n / mfu_k, 4),
        "seed": args.seed,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    if result["mfu_ratio"] < 1.4:
        print(f"HARD GATE FAILED: mfu_ratio={result['mfu_ratio']} < 1.4", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
