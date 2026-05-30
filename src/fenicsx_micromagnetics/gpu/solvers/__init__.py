"""
GPU solvers for FEniCSx-Micromagnetics.
"""

from __future__ import annotations

__all__ = [
    "LLG_GPU",
    "LLG_STT_GPU",
    "EnergyMinimizerGPU",
]


def __getattr__(name: str):
    if name == "LLG_GPU":
        from .llg_module_GPU import LLG_GPU
        return LLG_GPU

    if name == "LLG_STT_GPU":
        from .llg_stt_module_GPU import LLG_STT_GPU
        return LLG_STT_GPU

    if name == "EnergyMinimizerGPU":
        from .Minimizer_GPU import EnergyMinimizerGPU
        return EnergyMinimizerGPU
    
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")