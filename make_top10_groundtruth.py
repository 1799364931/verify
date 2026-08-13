#!/usr/bin/env python3
"""Validate a 100-neighbor ivecs file and write its first ten IDs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from cagra_utils import DEFAULT_CONFIG, apply_config, load_config, open_bvecs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="JSON configuration file")
    parser.add_argument("--input")
    parser.add_argument("--query")
    parser.add_argument("--output")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_config(args, load_config(args.config))
    if not args.input or not args.output:
        raise ValueError("missing groundtruth input/output paths in --config or command line")
    output = Path(args.output)
    if output.exists() and not args.force:
        raise ValueError(f"{output} already exists; pass --force to overwrite")
    query_count = open_bvecs(args.query).shape[0]
    words = np.memmap(args.input, dtype="<i4", mode="r")
    if words.size == 0 or int(words[0]) != 100 or words.size % 101:
        raise ValueError(f"{args.input}: expected fixed 100-neighbor ivecs records")
    records = words.reshape(-1, 101)
    if records.shape[0] != query_count or not np.all(records[:, 0] == 100):
        raise ValueError(f"{args.input}: headers or query count are invalid")
    result = np.empty((query_count, 11), dtype="<i4")
    result[:, 0] = 10
    result[:, 1:] = records[:, 1:11]
    output.parent.mkdir(parents=True, exist_ok=True)
    result.tofile(output)
    print(f"wrote {query_count} x top10 ivecs: {output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
