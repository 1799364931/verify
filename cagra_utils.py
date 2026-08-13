"""Small shared helpers for the CAGRA verification scripts."""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CONFIG = Path(__file__).with_name("cagra_config.json")


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as handle:
            config = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"{path}: top level must be a JSON object")
    return config


def apply_config(args: argparse.Namespace, config: dict[str, Any], index_kind: str | None = None) -> None:
    """Fill unset command-line values from the shared JSON configuration."""
    paths = config.get("paths", {})
    benchmark = config.get("benchmark", {})
    index_params = config.get("index_params", {})
    search_params = config.get("search_params", {})
    multi_gpu = config.get("multi_gpu", {})
    for name, section, key in (
        ("base", paths, "base"), ("query", paths, "query"),
        ("groundtruth", paths, "groundtruth_top10"),
        ("topk", benchmark, "topk"), ("warmup", benchmark, "warmup"),
        ("repetitions", benchmark, "repetitions"),
        ("metric", index_params, "metric"),
        ("graph_degree", index_params, "graph_degree"),
        ("intermediate_graph_degree", index_params, "intermediate_graph_degree"),
        ("build_algo", index_params, "build_algo"),
        ("itopk_size", search_params, "itopk_size"),
        ("merge_mode", multi_gpu, "merge_mode"),
        ("n_rows_per_batch", multi_gpu, "n_rows_per_batch"),
    ):
        if hasattr(args, name) and getattr(args, name) is None and key in section:
            setattr(args, name, section[key])
    if index_kind and getattr(args, "index", None) is None:
        key = "single_index" if index_kind == "single" else "sharded_index"
        args.index = paths.get(key)
    if hasattr(args, "input") and args.input is None:
        args.input = paths.get("groundtruth_source")
    if hasattr(args, "output") and args.output is None:
        args.output = paths.get("groundtruth_top10")
    required = ("base", "query", "topk", "warmup", "repetitions")
    missing = [name for name in required if hasattr(args, name) and getattr(args, name) is None]
    if missing:
        raise ValueError(f"missing configuration values: {', '.join(missing)}")
    if index_kind and not args.index:
        raise ValueError("missing index path in --index or config paths")


def open_bbin(path: str | Path) -> np.memmap:
    """Open a [uint32 rows, uint32 dim] + uint8 row-major matrix."""
    path = Path(path)
    with path.open("rb") as handle:
        header = handle.read(8)
    if len(header) != 8:
        raise ValueError(f"{path}: missing bbin header")
    rows, dim = struct.unpack("<II", header)
    if rows == 0 or dim == 0:
        raise ValueError(f"{path}: invalid shape ({rows}, {dim})")
    expected = 8 + rows * dim
    if path.stat().st_size != expected:
        raise ValueError(f"{path}: expected {expected} bytes, found {path.stat().st_size}")
    return np.memmap(path, dtype=np.uint8, mode="r", offset=8, shape=(rows, dim))


def open_bvecs(path: str | Path) -> np.ndarray:
    """Read standard repeated-int32-header bvecs into contiguous uint8 host memory."""
    path = Path(path)
    with path.open("rb") as handle:
        raw_dim = handle.read(4)
    if len(raw_dim) != 4:
        raise ValueError(f"{path}: missing bvecs header")
    dim = struct.unpack("<i", raw_dim)[0]
    if dim <= 0:
        raise ValueError(f"{path}: invalid bvecs dimension {dim}")
    stride = 4 + dim
    size = path.stat().st_size
    if size % stride:
        raise ValueError(f"{path}: size is not a whole number of bvecs records")
    rows = size // stride
    if rows == 0:
        raise ValueError(f"{path}: contains no queries")
    records = np.memmap(
        path, mode="r", dtype=np.dtype([("dim", "<i4"), ("vector", "u1", dim)]), shape=(rows,)
    )
    if not np.all(records["dim"] == dim):
        raise ValueError(f"{path}: inconsistent bvecs dimensions")
    return np.ascontiguousarray(records["vector"])


def load_ivecs(path: str | Path, expected_rows: int | None = None, topk: int = 10) -> np.ndarray:
    """Load fixed-width standard ivecs and return the first ``topk`` IDs."""
    path = Path(path)
    words = np.memmap(path, dtype="<i4", mode="r")
    if words.size == 0:
        raise ValueError(f"{path}: empty ivecs file")
    width = int(words[0])
    if width < topk or words.size % (width + 1):
        raise ValueError(f"{path}: invalid fixed-width ivecs records")
    records = words.reshape(-1, width + 1)
    if not np.all(records[:, 0] == width):
        raise ValueError(f"{path}: inconsistent ivecs headers")
    if expected_rows is not None and records.shape[0] != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, found {records.shape[0]}")
    return np.asarray(records[:, 1 : topk + 1], dtype=np.int64)


def recall_at_k(neighbors: np.ndarray, groundtruth: np.ndarray) -> float:
    if neighbors.shape != groundtruth.shape:
        raise ValueError(f"result shape {neighbors.shape} does not match ground truth {groundtruth.shape}")
    hits = sum(np.isin(neighbors[row], groundtruth[row]).sum() for row in range(neighbors.shape[0]))
    return float(hits / neighbors.size) if neighbors.size else 0.0


def add_cagra_param_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--metric", choices=["sqeuclidean", "inner_product", "cosine"])
    parser.add_argument("--graph-degree", type=int)
    parser.add_argument("--intermediate-graph-degree", type=int)
    parser.add_argument("--build-algo", choices=["ivf_pq", "nn_descent", "iterative_cagra_search", "ace"])
    parser.add_argument("--itopk-size", type=int)


def index_param_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in ("metric", "graph_degree", "intermediate_graph_degree", "build_algo"):
        value = getattr(args, name)
        if value is not None:
            values[name] = value
    return values


def search_param_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {} if args.itopk_size is None else {"itopk_size": args.itopk_size}


def validate_common_args(args: argparse.Namespace) -> None:
    if args.topk <= 0 or args.warmup < 0 or args.repetitions <= 0:
        raise ValueError("topk and repetitions must be positive; warmup must be non-negative")
    for name in ("graph_degree", "intermediate_graph_degree", "itopk_size"):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")


def write_metrics(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def gpu_info(cp: Any) -> list[dict[str, Any]]:
    result = []
    for device in range(cp.cuda.runtime.getDeviceCount()):
        props = cp.cuda.runtime.getDeviceProperties(device)
        name = props["name"]
        result.append({"id": device, "name": name.decode() if isinstance(name, bytes) else str(name)})
    return result


def common_metadata(args: argparse.Namespace, base: np.ndarray, queries: np.ndarray) -> dict[str, Any]:
    return {
        "config": os.fspath(args.config),
        "base": os.fspath(args.base),
        "base_shape": list(base.shape),
        "query": os.fspath(args.query),
        "query_shape": list(queries.shape),
        "topk": args.topk,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "index_params": index_param_kwargs(args),
        "search_params": search_param_kwargs(args),
    }
