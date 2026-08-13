#!/usr/bin/env python3
"""Benchmark same-index replication or native sharded multi-GPU CAGRA."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
from cuvs.common import MultiGpuResources
from cuvs.neighbors.mg import cagra as mg_cagra

from cagra_utils import (
    DEFAULT_CONFIG,
    add_cagra_param_args,
    apply_config,
    common_metadata,
    gpu_info,
    index_param_kwargs,
    load_config,
    load_ivecs,
    open_bbin,
    open_bvecs,
    recall_at_k,
    search_param_kwargs,
    validate_common_args,
    write_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["same-index", "sharded"], required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="JSON configuration file")
    parser.add_argument("--base")
    parser.add_argument("--query")
    parser.add_argument("--index", help="single index for same-index; sharded index otherwise")
    parser.add_argument("--groundtruth", help="pass an empty string to skip recall")
    parser.add_argument("--results", help="optional .npz output")
    parser.add_argument("--metrics", help="JSON output; defaults to cagra_<mode>_metrics.json")
    parser.add_argument("--topk", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--rebuild", action="store_true", help="rebuild a sharded index")
    parser.add_argument("--merge-mode", choices=["tree_merge", "merge_on_root_rank"])
    parser.add_argument("--n-rows-per-batch", type=int)
    add_cagra_param_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_config(args, load_config(args.config), index_kind="single" if args.mode == "same-index" else "sharded")
    validate_common_args(args)
    if args.n_rows_per_batch is not None and args.n_rows_per_batch <= 0:
        raise ValueError("n-rows-per-batch must be positive")
    if cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("no CUDA devices are visible; set CUDA_VISIBLE_DEVICES")
    base, queries = open_bbin(args.base), open_bvecs(args.query)
    if base.shape[1] != queries.shape[1]:
        raise ValueError(f"dimension mismatch: base={base.shape[1]}, query={queries.shape[1]}")
    if args.topk > base.shape[0]:
        raise ValueError("topk exceeds base vector count")

    resources = MultiGpuResources()
    index_path = Path(args.index)
    if args.mode == "same-index":
        if not index_path.exists():
            raise ValueError(f"same-index requires a single-GPU index: {index_path}")
        prepare_start = time.perf_counter()
        index = mg_cagra.distribute(str(index_path), resources=resources)
        prepare_kind = "distribute"
    else:
        if args.rebuild and index_path.exists():
            index_path.unlink()
        prepare_start = time.perf_counter()
        if index_path.exists():
            index = mg_cagra.load(str(index_path), resources=resources)
            prepare_kind = "load_sharded"
        else:
            index = mg_cagra.build(
                mg_cagra.IndexParams(distribution_mode="sharded", **index_param_kwargs(args)),
                base,
                resources=resources,
            )
            index_path.parent.mkdir(parents=True, exist_ok=True)
            mg_cagra.save(index, str(index_path), resources=resources)
            prepare_kind = "build_sharded"
    resources.sync()
    prepare_seconds = time.perf_counter() - prepare_start

    search_kwargs = search_param_kwargs(args)
    if args.merge_mode is not None:
        search_kwargs["merge_mode"] = args.merge_mode
    if args.n_rows_per_batch is not None:
        search_kwargs["n_rows_per_batch"] = args.n_rows_per_batch
    params = mg_cagra.SearchParams(**search_kwargs)
    warmup_start = time.perf_counter()
    for _ in range(args.warmup):
        mg_cagra.search(params, index, queries, args.topk, resources=resources)
    resources.sync()
    warmup_seconds = time.perf_counter() - warmup_start

    start = time.perf_counter()
    distances = neighbors = None
    for _ in range(args.repetitions):
        distances, neighbors = mg_cagra.search(params, index, queries, args.topk, resources=resources)
    resources.sync()
    search_seconds = time.perf_counter() - start
    qps = queries.shape[0] * args.repetitions / search_seconds

    metrics = common_metadata(args, base, queries)
    metrics.update(
        {
            "mode": args.mode,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
            "gpus": gpu_info(cp),
            "gpu_count": cp.cuda.runtime.getDeviceCount(),
            "index_prepare_kind": prepare_kind,
            "index_prepare_seconds": prepare_seconds,
            "warmup_seconds": warmup_seconds,
            "search_seconds": search_seconds,
            "qps": qps,
            "merge_mode": args.merge_mode,
            "n_rows_per_batch": args.n_rows_per_batch,
        }
    )
    if args.groundtruth:
        gt = load_ivecs(args.groundtruth, expected_rows=queries.shape[0], topk=args.topk)
        metrics[f"recall_at_{args.topk}"] = recall_at_k(neighbors, gt)
    if args.results:
        np.savez(args.results, neighbors=neighbors, distances=distances)
    metrics_path = args.metrics or f"cagra_{args.mode}_metrics.json"
    write_metrics(metrics_path, metrics)
    print(f"index_prepare_seconds={prepare_seconds:.6f}")
    print(f"warmup_seconds={warmup_seconds:.6f}")
    print(f"search_seconds={search_seconds:.6f}")
    print(f"qps={qps:.2f}")
    if f"recall_at_{args.topk}" in metrics:
        print(f"recall_at_{args.topk}={metrics[f'recall_at_{args.topk}']:.6f}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
