from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CompressionResult:
    U: np.ndarray
    V: np.ndarray
    rank: int
    relative_residual: float
    method: str

    @property
    def storage_entries(self) -> int:
        return int(self.U.size + self.V.size)


def fullaca_dense(
    A: np.ndarray,
    rtol: float = 1e-6,
    max_rank: int | None = None,
) -> CompressionResult:
    """
    Simple full-pivot ACA reference implementation.

    It intentionally materializes the current dense residual. This is suitable
    for validating the architecture. It is not the final GPU construction path.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be a matrix.")

    m, n = A.shape
    if max_rank is None:
        max_rank = min(m, n)
    max_rank = min(int(max_rank), m, n)

    norm_A = float(np.linalg.norm(A, ord="fro"))
    if norm_A == 0.0 or max_rank == 0:
        return CompressionResult(
            U=np.zeros((m, 0), dtype=np.float64),
            V=np.zeros((0, n), dtype=np.float64),
            rank=0,
            relative_residual=0.0,
            method="fullaca",
        )

    residual = A.copy()
    U_cols = []
    V_rows = []
    relative_residual = 1.0

    for _ in range(max_rank):
        pivot_flat = int(np.argmax(np.abs(residual)))
        i, j = np.unravel_index(pivot_flat, residual.shape)
        pivot = residual[i, j]

        # This protects against exhausted numerical rank.
        if abs(pivot) <= np.finfo(np.float64).eps * norm_A:
            break

        u = residual[:, j].copy()
        v = residual[i, :].copy() / pivot

        U_cols.append(u)
        V_rows.append(v)

        residual -= np.outer(u, v)
        relative_residual = float(np.linalg.norm(residual, ord="fro") / norm_A)

        if relative_residual <= rtol:
            break

    if not U_cols:
        U = np.zeros((m, 0), dtype=np.float64)
        V = np.zeros((0, n), dtype=np.float64)
    else:
        U = np.ascontiguousarray(np.column_stack(U_cols), dtype=np.float64)
        V = np.ascontiguousarray(np.row_stack(V_rows), dtype=np.float64)

    return CompressionResult(
        U=U,
        V=V,
        rank=int(U.shape[1]),
        relative_residual=float(relative_residual),
        method="fullaca",
    )


def truncated_svd(
    A: np.ndarray,
    rtol: float = 1e-6,
    max_rank: int | None = None,
) -> CompressionResult:
    """Reference truncated SVD compressor for debugging and validation."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be a matrix.")

    m, n = A.shape
    if max_rank is None:
        max_rank = min(m, n)
    max_rank = min(int(max_rank), m, n)

    norm_A = float(np.linalg.norm(A, ord="fro"))
    if norm_A == 0.0 or max_rank == 0:
        return CompressionResult(
            U=np.zeros((m, 0), dtype=np.float64),
            V=np.zeros((0, n), dtype=np.float64),
            rank=0,
            relative_residual=0.0,
            method="svd",
        )

    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    squared_tail = np.cumsum((s[::-1] ** 2))[::-1]

    rank = min(max_rank, len(s))
    for candidate in range(0, min(max_rank, len(s)) + 1):
        residual_norm = 0.0 if candidate == len(s) else float(np.sqrt(squared_tail[candidate]))
        if residual_norm / norm_A <= rtol:
            rank = candidate
            break

    if rank == 0:
        U_scaled = np.zeros((m, 0), dtype=np.float64)
        V = np.zeros((0, n), dtype=np.float64)
    else:
        U_scaled = np.ascontiguousarray(U[:, :rank] * s[:rank][None, :])
        V = np.ascontiguousarray(Vh[:rank, :])

    residual = A - U_scaled @ V
    rel = float(np.linalg.norm(residual, ord="fro") / norm_A)

    return CompressionResult(
        U=U_scaled,
        V=V,
        rank=int(rank),
        relative_residual=rel,
        method="svd",
    )


def compress_dense_block(
    A: np.ndarray,
    method: str,
    rtol: float,
    max_rank: int | None,
) -> CompressionResult:
    method_normalized = str(method).strip().lower()
    if method_normalized == "fullaca":
        return fullaca_dense(A, rtol=rtol, max_rank=max_rank)
    if method_normalized == "svd":
        return truncated_svd(A, rtol=rtol, max_rank=max_rank)
    raise ValueError(f"Unsupported compressor: {method!r}. Use 'fullaca' or 'svd'.")
