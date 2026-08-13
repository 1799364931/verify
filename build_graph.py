#!/usr/bin/env python3
"""Build/load a single-GPU CAGRA index and measure search QPS."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
from cuvs.common import Resources
from cuvs.neighbors import cagra

from cagra_utils import (
    DEFAULT_CONFIG,
    add_cagra_param_args,
    apply_config,
    index_param_kwargs,
    load_config,
    load_ivecs,
    open_bbin,
    open_bvecs,
    recall_at_k,
    search_param_kwargs,
    validate_common_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="JSON configuration file")
    parser.add_argument("--base")
    parser.add_argument("--query")
    parser.add_argument("--index", help="CAGRA index file")
    parser.add_argument("--groundtruth", help="optional ivecs")
    parser.add_argument("--results", help="optional .npz output")
    parser.add_argument("--topk", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--rebuild", action="store_true")
    add_cagra_param_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_config(args, load_config(args.config), index_kind="single")
    validate_common_args(args)
    base, queries = open_bbin(args.base), open_bvecs(args.query)
    if base.shape[1] != queries.shape[1]:
        raise ValueError(f"dimension mismatch: base={base.shape[1]}, query={queries.shape[1]}")
    if args.topk > base.shape[0]:
        raise ValueError("topk exceeds base vector count")

    index_path = Path(args.index)
    if args.rebuild and index_path.exists():
        index_path.unlink()
    resources = Resources()
    build_start = time.perf_counter()
    if index_path.exists():
        index = cagra.load(str(index_path), resources=resources)
        action = "load"
    else:
        dataset = cp.asarray(base)
        index = cagra.build(cagra.IndexParams(**index_param_kwargs(args)), dataset, resources=resources)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        cagra.save(str(index_path), index, include_dataset=True, resources=resources)
        action = "build"
    resources.sync()
    print(f"index_{action}_seconds={time.perf_counter() - build_start:.6f}")

    query_device = cp.asarray(queries)
    params = cagra.SearchParams(**search_param_kwargs(args))
    for _ in range(args.warmup):
        cagra.search(params, index, query_device, args.topk, resources=resources)
    resources.sync()
    start = time.perf_counter()
    distances = neighbors = None
    for _ in range(args.repetitions):
        distances, neighbors = cagra.search(params, index, query_device, args.topk, resources=resources)
    resources.sync()
    elapsed = time.perf_counter() - start
    qps = queries.shape[0] * args.repetitions / elapsed
    print(f"search_seconds={elapsed:.6f}")
    print(f"qps={qps:.2f}")

    host_neighbors = cp.asnumpy(neighbors)
    if args.groundtruth:
        gt = load_ivecs(args.groundtruth, expected_rows=queries.shape[0], topk=args.topk)
        print(f"recall_at_{args.topk}={recall_at_k(host_neighbors, gt):.6f}")
    if args.results:
        np.savez(args.results, neighbors=host_neighbors, distances=cp.asnumpy(distances))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
