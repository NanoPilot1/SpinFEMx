"""
Command-line mesh diagnostics for DOLFINx simplex meshes.

This utility provides simplex-shape and edge-size statistics for finite-element
meshes. Its terminal-oriented workflow was inspired by mesh inspection tools
such as Nmag/nmeshpp. The implementation was written independently for
DOLFINx and MPI and does not reuse Nmag source code.

The simplex-quality metric is:

    q = d * r_in / r_circum

where d is the topological dimension, r_in is the inradius and r_circum is the
circumradius. A regular simplex has q = 1, while a degenerate simplex approaches
q = 0.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import numpy as np
from mpi4py import MPI

import dolfinx
from dolfinx import io


@dataclass(frozen=True)
class ScalarStats:
    count: int
    mean: float
    std: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class Histogram:
    edges: list[float]
    counts: list[int]
    probabilities: list[float]


@dataclass(frozen=True)
class MeshStats:
    mesh_path: str
    mesh_name: str
    dimension: int
    cell_type: str
    num_cells: int
    num_vertices: int
    num_edges: int
    quality_definition: str
    quality: ScalarStats
    quality_histogram: Histogram
    edge_length: ScalarStats
    edge_length_histogram: Histogram
    cell_measure: ScalarStats


def _detect_grid_name(xdmf_path: Path) -> str:
    """
    Return the first XDMF uniform-grid name.

    DOLFINx commonly writes Grid or mesh. Detecting the first uniform
    grid avoids forcing the user to remember the internal XDMF name.
    """
    root = ET.parse(xdmf_path).getroot()

    for grid in root.iter():
        tag = grid.tag.split("}")[-1]
        if tag != "Grid":
            continue

        grid_type = grid.attrib.get("GridType", "Uniform")
        name = grid.attrib.get("Name")

        if name and grid_type.lower() == "uniform":
            return name

    raise RuntimeError(
        f"Could not detect a uniform XDMF Grid name inside {xdmf_path}. "
        "Pass --name explicitly."
    )


def _read_mesh(
    mesh_path: Path,
    *,
    comm: MPI.Comm,
    mesh_name: str | None,
) -> tuple[dolfinx.mesh.Mesh, str]:
    if mesh_path.suffix.lower() != ".xdmf":
        raise ValueError("This command currently supports XDMF meshes only.")

    name = mesh_name or _detect_grid_name(mesh_path)

    with io.XDMFFile(comm, str(mesh_path), "r") as xdmf:
        mesh = xdmf.read_mesh(name=name)

    return mesh, name


def _global_stats(values: np.ndarray, comm: MPI.Comm) -> ScalarStats:
    values = np.asarray(values, dtype=np.float64).reshape(-1)

    local_count = int(values.size)
    count = int(comm.allreduce(local_count, op=MPI.SUM))

    if count == 0:
        nan = float("nan")
        return ScalarStats(0, nan, nan, nan, nan)

    local_sum = float(np.sum(values))
    local_sq_sum = float(np.dot(values, values))
    local_min = float(np.min(values)) if local_count else float("inf")
    local_max = float(np.max(values)) if local_count else -float("inf")

    total_sum = float(comm.allreduce(local_sum, op=MPI.SUM))
    total_sq_sum = float(comm.allreduce(local_sq_sum, op=MPI.SUM))
    minimum = float(comm.allreduce(local_min, op=MPI.MIN))
    maximum = float(comm.allreduce(local_max, op=MPI.MAX))

    mean = total_sum / count
    variance = max(total_sq_sum / count - mean * mean, 0.0)

    return ScalarStats(
        count=count,
        mean=mean,
        std=float(np.sqrt(variance)),
        minimum=minimum,
        maximum=maximum,
    )


def _global_histogram(
    values: np.ndarray,
    *,
    bins: int,
    value_range: tuple[float, float],
    comm: MPI.Comm,
) -> Histogram:
    lo, hi = map(float, value_range)

    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Histogram limits must be finite.")

    if hi <= lo:
        # Preserve a visible interval even for constant edge lengths.
        delta = max(abs(lo), 1.0) * 0.5
        lo -= delta
        hi += delta

    local_counts, edges = np.histogram(
        np.asarray(values, dtype=np.float64),
        bins=int(bins),
        range=(lo, hi),
    )

    global_counts = np.zeros_like(local_counts)
    comm.Allreduce(local_counts, global_counts, op=MPI.SUM)

    total = int(np.sum(global_counts))
    if total:
        probabilities = global_counts.astype(np.float64) / total
    else:
        probabilities = np.zeros_like(global_counts, dtype=np.float64)

    return Histogram(
        edges=[float(v) for v in edges],
        counts=[int(v) for v in global_counts],
        probabilities=[float(v) for v in probabilities],
    )


def _triangle_area(points: np.ndarray) -> float:
    p0, p1, p2 = points
    return 0.5 * float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)))


def _triangle_quality(points: np.ndarray) -> tuple[float, float]:
    p0, p1, p2 = points

    a = float(np.linalg.norm(p1 - p2))
    b = float(np.linalg.norm(p2 - p0))
    c = float(np.linalg.norm(p0 - p1))

    area = _triangle_area(points)
    semiperimeter = 0.5 * (a + b + c)

    if area <= 0.0 or semiperimeter <= 0.0:
        return 0.0, area

    r_in = area / semiperimeter
    r_circum = a * b * c / (4.0 * area)

    if r_circum <= 0.0:
        return 0.0, area

    q = 2.0 * r_in / r_circum
    return float(np.clip(q, 0.0, 1.0)), area


def _tetra_volume(points: np.ndarray) -> float:
    p0, p1, p2, p3 = points
    return abs(float(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0)))) / 6.0


def _tetra_quality(points: np.ndarray) -> tuple[float, float]:
    volume = _tetra_volume(points)

    if volume <= 0.0:
        return 0.0, volume

    p0, p1, p2, p3 = points

    surface_area = (
        _triangle_area(np.asarray([p0, p1, p2]))
        + _triangle_area(np.asarray([p0, p1, p3]))
        + _triangle_area(np.asarray([p0, p2, p3]))
        + _triangle_area(np.asarray([p1, p2, p3]))
    )

    if surface_area <= 0.0:
        return 0.0, volume

    r_in = 3.0 * volume / surface_area

    matrix = 2.0 * (points[1:] - p0)
    rhs = np.einsum("ij,ij->i", points[1:], points[1:]) - float(
        np.dot(p0, p0)
    )

    try:
        center = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return 0.0, volume

    r_circum = float(np.linalg.norm(center - p0))

    if not np.isfinite(r_circum) or r_circum <= 0.0:
        return 0.0, volume

    q = 3.0 * r_in / r_circum
    return float(np.clip(q, 0.0, 1.0)), volume


def _owned_cell_quality_and_measure(
    mesh: dolfinx.mesh.Mesh,
) -> tuple[np.ndarray, np.ndarray]:
    tdim = mesh.topology.dim

    if tdim not in (2, 3):
        raise NotImplementedError(
            "Only triangular 2D and tetrahedral 3D meshes are supported."
        )

    cell_map = mesh.topology.index_map(tdim)
    if cell_map is None:
        raise RuntimeError("Missing cell index map.")

    n_owned = int(cell_map.size_local)
    coordinates = np.asarray(mesh.geometry.x, dtype=np.float64)
    geometry_dofmap = np.asarray(mesh.geometry.dofmap)

    expected_nodes = tdim + 1

    quality = np.empty(n_owned, dtype=np.float64)
    measure = np.empty(n_owned, dtype=np.float64)

    for cell in range(n_owned):
        gdofs = np.asarray(geometry_dofmap[cell], dtype=np.int32)

        if gdofs.size != expected_nodes:
            raise NotImplementedError(
                "Only first-order simplex geometry is supported. "
                f"Cell {cell} has {gdofs.size} geometry nodes; "
                f"expected {expected_nodes}."
            )

        points = coordinates[gdofs]

        if tdim == 3:
            quality[cell], measure[cell] = _tetra_quality(points)
        else:
            quality[cell], measure[cell] = _triangle_quality(points)

    return quality, measure


def _vertex_to_geometry_dof(mesh: dolfinx.mesh.Mesh) -> np.ndarray:
    topology = mesh.topology
    tdim = topology.dim

    topology.create_connectivity(tdim, 0)
    cells_to_vertices = topology.connectivity(tdim, 0)

    if cells_to_vertices is None:
        raise RuntimeError("Could not create cell-to-vertex connectivity.")

    cell_map = topology.index_map(tdim)
    vertex_map = topology.index_map(0)

    if cell_map is None or vertex_map is None:
        raise RuntimeError("Missing topology index map.")

    n_cells = int(cell_map.size_local + cell_map.num_ghosts)
    n_vertices = int(vertex_map.size_local + vertex_map.num_ghosts)

    geometry_dofmap = np.asarray(mesh.geometry.dofmap)
    mapping = np.full(n_vertices, -1, dtype=np.int32)

    for cell in range(n_cells):
        vertices = np.asarray(cells_to_vertices.links(cell), dtype=np.int32)
        gdofs = np.asarray(geometry_dofmap[cell], dtype=np.int32)

        if vertices.size != gdofs.size:
            raise NotImplementedError(
                "Only first-order simplex geometry is supported."
            )

        mapping[vertices] = gdofs

    if np.any(mapping < 0):
        raise RuntimeError("Could not map all topology vertices to geometry dofs.")

    return mapping


def _owned_edge_lengths(mesh: dolfinx.mesh.Mesh) -> np.ndarray:
    topology = mesh.topology

    topology.create_entities(1)
    topology.create_connectivity(1, 0)

    edge_map = topology.index_map(1)
    edges_to_vertices = topology.connectivity(1, 0)

    if edge_map is None or edges_to_vertices is None:
        raise RuntimeError("Could not build edge topology.")

    vertex_to_geometry = _vertex_to_geometry_dof(mesh)
    coordinates = np.asarray(mesh.geometry.x, dtype=np.float64)

    n_owned_edges = int(edge_map.size_local)
    lengths = np.empty(n_owned_edges, dtype=np.float64)

    for edge in range(n_owned_edges):
        vertices = np.asarray(edges_to_vertices.links(edge), dtype=np.int32)

        if vertices.size != 2:
            raise RuntimeError("An edge must contain exactly two vertices.")

        gdofs = vertex_to_geometry[vertices]
        lengths[edge] = float(
            np.linalg.norm(coordinates[gdofs[1]] - coordinates[gdofs[0]])
        )

    return lengths


def analyze_mesh(
    mesh: dolfinx.mesh.Mesh,
    *,
    quality_bins: int = 10,
    edge_bins: int = 10,
) -> MeshStats:
    comm = mesh.comm
    tdim = int(mesh.topology.dim)

    quality_values, cell_measures = _owned_cell_quality_and_measure(mesh)
    edge_lengths = _owned_edge_lengths(mesh)

    quality_stats = _global_stats(quality_values, comm)
    edge_stats = _global_stats(edge_lengths, comm)
    cell_stats = _global_stats(cell_measures, comm)

    quality_hist = _global_histogram(
        quality_values,
        bins=quality_bins,
        value_range=(0.0, 1.0),
        comm=comm,
    )

    edge_hist = _global_histogram(
        edge_lengths,
        bins=edge_bins,
        value_range=(edge_stats.minimum, edge_stats.maximum),
        comm=comm,
    )

    cell_map = mesh.topology.index_map(tdim)
    vertex_map = mesh.topology.index_map(0)
    edge_map = mesh.topology.index_map(1)

    if cell_map is None or vertex_map is None or edge_map is None:
        raise RuntimeError("Missing topology index maps.")

    return MeshStats(
        mesh_path="",
        mesh_name="",
        dimension=tdim,
        cell_type=str(mesh.topology.cell_type),
        num_cells=int(cell_map.size_global),
        num_vertices=int(vertex_map.size_global),
        num_edges=int(edge_map.size_global),
        quality_definition="q = d * r_in / r_circum",
        quality=quality_stats,
        quality_histogram=quality_hist,
        edge_length=edge_stats,
        edge_length_histogram=edge_hist,
        cell_measure=cell_stats,
    )


def _histogram_lines(
    histogram: Histogram,
    *,
    digits: int,
    bar_width: int = 36,
) -> Iterable[str]:
    """Format a compact terminal histogram using a project-specific layout."""
    probabilities = np.asarray(histogram.probabilities, dtype=np.float64)
    peak = float(np.max(probabilities)) if probabilities.size else 0.0

    for idx, count in enumerate(histogram.counts):
        lo = histogram.edges[idx]
        hi = histogram.edges[idx + 1]
        probability = histogram.probabilities[idx]

        n_blocks = (
            int(round(bar_width * probability / peak))
            if peak > 0.0
            else 0
        )

        yield (
            f"{lo: .{digits}f} .. {hi: .{digits}f} | "
            f"{count:8d} | "
            f"{100.0 * probability:6.2f}% | "
            f"{'#' * n_blocks}"
        )


def format_report(stats: MeshStats) -> str:
    """Return a terminal report with a project-specific layout."""
    lines = [
        "FEniCSx-Micromagnetics :: mesh diagnostic report",
        "-----------------------------------------------------------------------",
        f"source file        : {stats.mesh_path}",
        f"grid identifier    : {stats.mesh_name}",
        f"topological dim.   : {stats.dimension}",
        f"element type       : {stats.cell_type}",
        f"cells              : {stats.num_cells}",
        f"vertices           : {stats.num_vertices}",
        f"unique edges       : {stats.num_edges}",
        "",
        "Element shape score",
        "-----------------------------------------------------------------------",
        f"definition         : {stats.quality_definition}",
        "meaning            : 1 -> regular simplex; 0 -> collapsed simplex",
        (
            "summary            : "
            f"avg={stats.quality.mean:.6f}, "
            f"std={stats.quality.std:.6f}, "
            f"lowest={stats.quality.minimum:.6f}, "
            f"highest={stats.quality.maximum:.6f}"
        ),
        "",
        "score range              | elements | share   | relative bar",
    ]

    lines.extend(_histogram_lines(stats.quality_histogram, digits=3))

    lines.extend(
        [
            "",
            "Edge-size distribution [mesh coordinate units]",
            "-----------------------------------------------------------------------",
            (
                "summary            : "
                f"avg={stats.edge_length.mean:.6f}, "
                f"std={stats.edge_length.std:.6f}, "
                f"smallest={stats.edge_length.minimum:.6f}, "
                f"largest={stats.edge_length.maximum:.6f}"
            ),
            "",
            "edge-size range          | edges    | share   | relative bar",
        ]
    )

    lines.extend(_histogram_lines(stats.edge_length_histogram, digits=6))

    lines.extend(
        [
            "",
            f"Cell measure summary [mesh units^{stats.dimension}]",
            "-----------------------------------------------------------------------",
            (
                "summary            : "
                f"avg={stats.cell_measure.mean:.6e}, "
                f"std={stats.cell_measure.std:.6e}, "
                f"smallest={stats.cell_measure.minimum:.6e}, "
                f"largest={stats.cell_measure.maximum:.6e}"
            ),
        ]
    )

    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="print_statistic",
        description=(
            "Print simplex-shape and edge-size diagnostics for a "
            "DOLFINx triangular or tetrahedral XDMF mesh."
        ),
    )

    parser.add_argument(
        "mesh",
        type=Path,
        help="Input XDMF mesh.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help=(
            "Grid name inside the XDMF file. If omitted, the first uniform "
            "XDMF Grid is detected automatically."
        ),
    )
    parser.add_argument(
        "--quality-bins",
        type=int,
        default=10,
        help="Number of simplex-quality histogram bins. Default: 10.",
    )
    parser.add_argument(
        "--edge-bins",
        type=int,
        default=10,
        help="Number of edge-length histogram bins. Default: 10.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )

    return parser


def main() -> int:
    args = _parser().parse_args()
    comm = MPI.COMM_WORLD

    mesh_path = args.mesh.expanduser().resolve()

    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)

    mesh, mesh_name = _read_mesh(
        mesh_path,
        comm=comm,
        mesh_name=args.name,
    )

    stats = analyze_mesh(
        mesh,
        quality_bins=args.quality_bins,
        edge_bins=args.edge_bins,
    )

    stats = replace(
        stats,
        mesh_path=str(mesh_path),
        mesh_name=mesh_name,
    )

    if comm.rank == 0:
        print(format_report(stats), flush=True)

        if args.json is not None:
            json_path = args.json.expanduser().resolve()
            json_path.parent.mkdir(parents=True, exist_ok=True)

            with json_path.open("w", encoding="utf-8") as handle:
                json.dump(asdict(stats), handle, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
