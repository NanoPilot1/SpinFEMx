from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import numpy as np

from .builder import HMatrixBuildConfig, HMatrixBuilder, print_hmatrix_summary
from .backend import HMatrixGPU
from .backend_fused import HMatrixGPUPackedFused


class SyntheticKernelProvider:
    """
    Smooth nonsymmetric synthetic kernel for testing generic H-matrix code.

    This is not the Lindholm kernel. It tests the cluster tree, compression,
    persistence, CPU MatVec and optional CuPy MatVec independently of DOLFINx.
    """

    def __init__(self, points: np.ndarray):
        self.Xb = np.asarray(points, dtype=np.float64)

    @property
    def size(self):
        return int(self.Xb.shape[0])

    def fill_block(self, rows, cols):
        target = self.Xb[np.asarray(rows, dtype=np.int32)]
        source = self.Xb[np.asarray(cols, dtype=np.int32)]

        delta = target[:, None, :] - source[None, :, :]
        distance = np.sqrt(np.sum(delta * delta, axis=2) + 0.04)
        nonsymmetry = 1.0 + 0.07 * target[:, None, 0] - 0.03 * source[None, :, 1]
        A = nonsymmetry / distance

        for ii, gi in enumerate(rows):
            matches = np.flatnonzero(np.asarray(cols) == gi)
            A[ii, matches] = 0.0
        return A


def dense_reference(provider, diag_jump):
    indices = np.arange(provider.size, dtype=np.int32)
    A = provider.fill_block(indices, indices)
    A[np.arange(provider.size), np.arange(provider.size)] += diag_jump
    return A


def build_gpu_backend(name: str, data, threads_per_block: int):
    name = str(name).strip().lower()
    if name == "legacy":
        return HMatrixGPU(data)
    if name == "packed_fused":
        return HMatrixGPUPackedFused(
            data,
            threads_per_block=threads_per_block,
        )
    raise ValueError(f"Unsupported GPU backend: {name!r}")


def run(cpu_only: bool, n: int, seed: int, gpu_backend: str, threads_per_block: int):
    rng = np.random.default_rng(seed)
    points = rng.random((n, 3))
    provider = SyntheticKernelProvider(points)
    diag_jump = -0.5 + 0.05 * rng.standard_normal(n)

    config = HMatrixBuildConfig(
        epsilon=1e-7,
        eta=1.5,
        leaf_size=16,
        compressor="fullaca",
        max_temporary_block_bytes=64 * 1024 * 1024,
    )

    builder = HMatrixBuilder(
        Xb=points,
        diag_jump=diag_jump,
        provider=provider,
        config=config,
    )
    data = builder.build(extra_metadata={"kernel": "synthetic"})

    if len(data.far_blocks) == 0:
        raise RuntimeError(
            "Validation setup did not generate low-rank blocks. "
            "The compressed far-field path was not exercised."
        )

    print_hmatrix_summary(data, prefix="[synthetic]")


    print_hmatrix_summary(data, prefix="[synthetic]")

    A = dense_reference(provider, diag_jump)
    x = rng.standard_normal(n)
    y_ref = A @ x
    y_cpu = data.matvec_cpu(x)
    rel_cpu = np.linalg.norm(y_cpu - y_ref) / np.linalg.norm(y_ref)
    print(f"[synthetic] CPU relative MatVec error : {rel_cpu:.12e}")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "synthetic_hmatrix.npz"
        data.save(cache_path)
        reloaded = type(data).load(cache_path)
        y_reloaded = reloaded.matvec_cpu(x)
        rel_reload = np.linalg.norm(y_reloaded - y_cpu) / max(np.linalg.norm(y_cpu), 1e-300)
        print(f"[synthetic] cache round-trip error     : {rel_reload:.12e}")

    if rel_cpu > 5e-6:
        raise RuntimeError(f"CPU validation failed: relative error {rel_cpu:.3e}")
    if rel_reload > 1e-14:
        raise RuntimeError(f"Cache validation failed: relative error {rel_reload:.3e}")

    if cpu_only:
        print("[synthetic] CPU-only validation passed.")
        return

    gpu = build_gpu_backend(gpu_backend, data, threads_per_block)
    y_gpu = gpu.matvec_numpy(x)
    rel_gpu = np.linalg.norm(y_gpu - y_ref) / np.linalg.norm(y_ref)

    print(f"[synthetic] GPU backend                : {gpu_backend}")
    print(f"[synthetic] GPU relative MatVec error : {rel_gpu:.12e}")

    timing = gpu.benchmark(x, repeats=30, warmup=3)
    print(f"[synthetic] GPU timing method          : {timing.get('timer', 'unknown')}")
    print(f"[synthetic] GPU average MatVec         : {timing['average_seconds']:.6e} s")

    if hasattr(gpu, "print_memory_report"):
        gpu.print_memory_report(prefix="[synthetic packed]")

    if rel_gpu > 5e-6:
        raise RuntimeError(f"GPU validation failed: relative error {rel_gpu:.3e}")

    print("[synthetic] CPU and GPU validation passed.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--n", type=int, default=320)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--gpu-backend",
        choices=["legacy", "packed_fused"],
        default="packed_fused",
    )
    parser.add_argument("--threads-per-block", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    run(
        cpu_only=args.cpu_only,
        n=args.n,
        seed=args.seed,
        gpu_backend=args.gpu_backend,
        threads_per_block=args.threads_per_block,
    )


if __name__ == "__main__":
    main()
