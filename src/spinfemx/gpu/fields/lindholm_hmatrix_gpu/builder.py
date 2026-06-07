from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json
import numpy as np

from .compressors import compress_dense_block
from .storage import HMatrixCPUData, HMatrixStorageBuilder
from .tree import ClusterNode, admissible, build_cluster_tree


@dataclass
class HMatrixBuildConfig:
    epsilon: float = 1e-6
    eta: float = 2.0
    leaf_size: int = 64
    compressor: str = "fullaca"
    max_rank: int | None = None
    max_temporary_block_bytes: int = 256 * 1024 * 1024
    low_rank_storage_factor: float = 0.95

    def __post_init__(self):
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if self.eta <= 0.0:
            raise ValueError("eta must be positive.")
        if self.leaf_size < 1:
            raise ValueError("leaf_size must be positive.")
        if self.max_temporary_block_bytes < 8:
            raise ValueError("max_temporary_block_bytes is too small.")
        if not (0.0 < self.low_rank_storage_factor <= 1.0):
            raise ValueError("low_rank_storage_factor must be in (0, 1].")


def geometry_hash(Xb: np.ndarray, tri_lid: np.ndarray | None = None) -> str:
    digest = sha256()
    digest.update(np.ascontiguousarray(Xb, dtype=np.float64).view(np.uint8))
    if tri_lid is not None:
        digest.update(np.ascontiguousarray(tri_lid, dtype=np.int32).view(np.uint8))
    return digest.hexdigest()


class HMatrixBuilder:
    """Build an H-matrix in permuted coordinates using an arbitrary entry provider."""

    def __init__(
        self,
        Xb: np.ndarray,
        diag_jump: np.ndarray,
        provider,
        config: HMatrixBuildConfig | None = None,
    ):
        self.Xb = np.ascontiguousarray(Xb, dtype=np.float64)
        self.diag_jump = np.ascontiguousarray(diag_jump, dtype=np.float64)
        self.provider = provider
        self.config = config or HMatrixBuildConfig()

        if self.Xb.ndim != 2 or self.Xb.shape[1] != 3:
            raise ValueError("Xb must have shape (Nb, 3).")
        if self.diag_jump.shape != (self.Xb.shape[0],):
            raise ValueError("diag_jump must have shape (Nb,).")
        if int(provider.size) != int(self.Xb.shape[0]):
            raise ValueError("provider.size must match Xb.shape[0].")

        self.Nb = int(self.Xb.shape[0])
        self.tree = build_cluster_tree(self.Xb, leaf_size=self.config.leaf_size)
        self.storage = HMatrixStorageBuilder(self.Nb)

        self.stats = {
            "dense_blocks": 0,
            "low_rank_blocks": 0,
            "rejected_low_rank_blocks": 0,
            "forced_subdivisions_memory": 0,
            "block_entry_evaluations": 0,
            "max_observed_rank": 0,
            "sum_observed_rank": 0,
        }

    def _indices(self, node: ClusterNode):
        return np.ascontiguousarray(self.tree.perm[node.start:node.stop], dtype=np.int32)

    def _dense_block(self, row_node: ClusterNode, col_node: ClusterNode):
        rows = self._indices(row_node)
        cols = self._indices(col_node)
        A = self.provider.fill_block(rows, cols)
        self.stats["block_entry_evaluations"] += int(A.size)
        return A

    def _add_dense(self, row_node: ClusterNode, col_node: ClusterNode):
        A = self._dense_block(row_node, col_node)
        self.storage.add_dense(
            row_node.start,
            row_node.stop,
            col_node.start,
            col_node.stop,
            A,
        )
        self.stats["dense_blocks"] += 1

    @staticmethod
    def _can_split(node: ClusterNode) -> bool:
        return not node.is_leaf

    def _subdivide(self, row_node: ClusterNode, col_node: ClusterNode):
        row_can_split = self._can_split(row_node)
        col_can_split = self._can_split(col_node)

        if not row_can_split and not col_can_split:
            self._add_dense(row_node, col_node)
            return

        if row_can_split and (not col_can_split or row_node.diameter >= col_node.diameter):
            self._build_block(row_node.left, col_node)
            self._build_block(row_node.right, col_node)
        else:
            self._build_block(row_node, col_node.left)
            self._build_block(row_node, col_node.right)

    def _build_block(self, row_node: ClusterNode, col_node: ClusterNode):
        config = self.config
        dense_bytes = row_node.size * col_node.size * np.dtype(np.float64).itemsize
        is_admissible = admissible(row_node, col_node, eta=config.eta)

        if is_admissible and dense_bytes > config.max_temporary_block_bytes:
            self.stats["forced_subdivisions_memory"] += 1
            self._subdivide(row_node, col_node)
            return

        if is_admissible:
            A = self._dense_block(row_node, col_node)
            result = compress_dense_block(
                A,
                method=config.compressor,
                rtol=config.epsilon,
                max_rank=config.max_rank,
            )

            dense_entries = int(A.size)
            low_rank_entries = int(result.storage_entries)

            if (
                result.rank > 0
                and low_rank_entries < config.low_rank_storage_factor * dense_entries
            ):
                self.storage.add_low_rank(
                    row_node.start,
                    row_node.stop,
                    col_node.start,
                    col_node.stop,
                    result.U,
                    result.V,
                    relative_residual=result.relative_residual,
                )
                self.stats["low_rank_blocks"] += 1
                self.stats["max_observed_rank"] = max(
                    self.stats["max_observed_rank"], result.rank
                )
                self.stats["sum_observed_rank"] += result.rank
            else:
                self.storage.add_dense(
                    row_node.start,
                    row_node.stop,
                    col_node.start,
                    col_node.stop,
                    A,
                )
                self.stats["dense_blocks"] += 1
                self.stats["rejected_low_rank_blocks"] += 1
            return

        if row_node.is_leaf and col_node.is_leaf:
            self._add_dense(row_node, col_node)
            return

        self._subdivide(row_node, col_node)

    def build(self, extra_metadata: dict | None = None) -> HMatrixCPUData:
        t0 = perf_counter()
        self._build_block(self.tree.root, self.tree.root)
        elapsed = perf_counter() - t0

        low_rank_blocks = self.stats["low_rank_blocks"]
        avg_rank = (
            self.stats["sum_observed_rank"] / low_rank_blocks
            if low_rank_blocks
            else 0.0
        )

        metadata = {
            "Nb": self.Nb,
            "geometry_hash": geometry_hash(
                self.Xb,
                getattr(self.provider, "tri_lid", None),
            ),
            "build_config": asdict(self.config),
            "build_time_seconds": elapsed,
            "average_far_rank": avg_rank,
            **self.stats,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        data = self.storage.finalize(
            perm=self.tree.perm,
            diag_jump=self.diag_jump,
            metadata=metadata,
        )
        data.metadata["near_nnz"] = int(data.near_csr.nnz)
        data.metadata["far_storage_entries"] = int(
            sum(block.storage_entries for block in data.far_blocks)
        )
        data.metadata["stored_entries_total"] = int(data.stored_entries)
        data.metadata["compression_ratio_entries"] = float(
            data.compression_ratio_entries
        )
        return data


def build_or_load_hmatrix(
    cache_path: str | Path,
    Xb: np.ndarray,
    diag_jump: np.ndarray,
    provider,
    config: HMatrixBuildConfig | None = None,
    force_rebuild: bool = False,
    verbose: bool = True,
) -> HMatrixCPUData:
    cache_path = Path(cache_path)
    config = config or HMatrixBuildConfig()
    expected_hash = geometry_hash(Xb, getattr(provider, "tri_lid", None))

    if cache_path.exists() and not force_rebuild:
        data = HMatrixCPUData.load(cache_path)
        cached_hash = data.metadata.get("geometry_hash")
        cached_config = data.metadata.get("build_config")

        if cached_hash == expected_hash and cached_config == asdict(config):
            if verbose:
                print(f"[HMatrix] loaded cache: {cache_path}", flush=True)
            return data

        if verbose:
            print("[HMatrix] cache metadata changed; rebuilding.", flush=True)

    builder = HMatrixBuilder(
        Xb=Xb,
        diag_jump=diag_jump,
        provider=provider,
        config=config,
    )
    data = builder.build()
    data.save(cache_path)

    if verbose:
        print_hmatrix_summary(data, prefix="[HMatrix build]")
        print(f"[HMatrix] saved cache: {cache_path}", flush=True)

    return data


def print_hmatrix_summary(data: HMatrixCPUData, prefix: str = "[HMatrix]"):
    meta = data.metadata
    print(f"{prefix} Nb                         : {data.size}")
    print(f"{prefix} near CSR nnz               : {data.near_csr.nnz}")
    print(f"{prefix} far low-rank blocks        : {len(data.far_blocks)}")
    print(f"{prefix} average far rank           : {meta.get('average_far_rank', 0.0):.3f}")
    print(f"{prefix} maximum far rank           : {meta.get('max_observed_rank', 0)}")
    print(f"{prefix} stored entries             : {data.stored_entries}")
    print(f"{prefix} compression ratio entries  : {data.compression_ratio_entries:.3f}")
    print(f"{prefix} build time                 : {meta.get('build_time_seconds', 0.0):.6f} s", flush=True)
