"""
CPU solvers for FEniCSx-Micromagnetics.
"""

from __future__ import annotations

__all__ = [
    "LLG",
    "LLG_STT",
    "EnergyMinimizer",
    "ZhangLi",
    "LLG_SOT",
]


def __getattr__(name: str):
    if name == "LLG":
        from .llg_module import LLG
        return LLG

    if name == "LLG_STT":
        from .llg_stt_module import LLG_STT
        return LLG_STT

    if name == "EnergyMinimizer":
        from .Minimizer import EnergyMinimizer
        return EnergyMinimizer

    if name == "ZhangLi":
        from .Zhang_Li import ZhangLi
        return ZhangLi

    if name == "LLG_SOT":
        from .llg_SOT_module import LLG_SOT
        return LLG_SOT

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
