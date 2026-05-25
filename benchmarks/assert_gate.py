#!/usr/bin/env python3
"""Hard-gate assertion for benchmark results. Exits nonzero on failure."""
import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--metric", required=True)
    ap.add_argument("--min", type=float, dest="min_", default=None)
    ap.add_argument("--max", type=float, dest="max_", default=None)
    args = ap.parse_args()

    if args.min_ is None and args.max_ is None:
        print("ERROR: must specify --min or --max", file=sys.stderr)
        sys.exit(2)

    data = json.loads(Path(args.result).read_text())
    actual = data[args.metric]

    if args.min_ is not None:
        passed = actual >= args.min_
        required = f">= {args.min_}"
    else:
        passed = actual <= args.max_
        required = f"<= {args.max_}"

    status = "PASS" if passed else "FAIL"
    print(
        f"AssertionError: HARD GATE {'PASSED' if passed else 'FAILED'}\n"
        f"  metric:   {args.metric}\n"
        f"  required: {required}\n"
        f"  actual:   {actual}\n"
        f"  status:   {status}"
    )
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
