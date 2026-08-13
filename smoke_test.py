#!/usr/bin/env python3
"""Tiny format and optional GPU smoke tests for the verification scripts.

Run with ``conda run -n sift-locality python smoke_test.py --with-gpu`` on a
GPU machine.  The test never accesses the SIFT100M files.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def write_bbin(path: Path, rows: np.ndarray) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<II", *rows.shape))
        rows.tofile(handle)


def write_bvecs(path: Path, rows: np.ndarray) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(struct.pack("<i", row.size))
            row.tofile(handle)


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=HERE, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-gpu", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(7)
    base = rng.integers(0, 256, size=(256, 8), dtype=np.uint8)
    queries = base[:16].copy()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        base_path, query_path = root / "base.bbin", root / "query.bvecs"
        source_gt, top10_gt = root / "source.ivecs", root / "top10.ivecs"
        write_bbin(base_path, base)
        write_bvecs(query_path, queries)
        source = np.empty((queries.shape[0], 101), dtype="<i4")
        source[:, 0] = 100
        source[:, 1:] = np.arange(100, dtype=np.int32)
        source.tofile(source_gt)
        run("make_top10_groundtruth.py", "--input", str(source_gt), "--query", str(query_path), "--output", str(top10_gt))
        top10 = np.fromfile(top10_gt, dtype="<i4").reshape(-1, 11)
        assert np.all(top10[:, 0] == 10) and np.array_equal(top10[:, 1:], source[:, 1:11])

        if not args.with_gpu:
            print("format smoke test passed (pass --with-gpu for CAGRA tests)")
            return
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError("--with-gpu requires CuPy/cuvs") from exc
        if cp.cuda.runtime.getDeviceCount() == 0:
            raise RuntimeError("--with-gpu requires a visible CUDA device")
        index_path = root / "single.cagra"
        common = [
            "--base", str(base_path), "--query", str(query_path), "--topk", "5", "--warmup", "0",
            "--groundtruth", str(top10_gt), "--build-algo", "nn_descent", "--graph-degree", "8",
            "--intermediate-graph-degree", "16",
        ]
        run("build_graph.py", "--index", str(index_path), *common)
        run("build_graph.py", "--index", str(index_path), *common)  # load path
        same_metrics = root / "same.json"
        run(
            "cagra_mg_benchmark.py", "--mode", "same-index", "--index", str(index_path),
            "--metrics", str(same_metrics), *common,
        )
        assert json.loads(same_metrics.read_text())["qps"] > 0
        if cp.cuda.runtime.getDeviceCount() >= 2:
            sharded_metrics = root / "sharded.json"
            run(
                "cagra_mg_benchmark.py", "--mode", "sharded", "--index", str(root / "sharded.cagra"),
                "--metrics", str(sharded_metrics), *common,
            )
            assert json.loads(sharded_metrics.read_text())["qps"] > 0
    print("GPU smoke test passed")


if __name__ == "__main__":
    main()
