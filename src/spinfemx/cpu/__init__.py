"""
CPU/MPI backend for SpinFEMx.
"""

from __future__ import annotations
from .io_utils import load_mesh_xdmf

__all__ = [
    "LLG",
    "LLG_STT",
    "EnergyMinimizer",
    "ZhangLi",
    "ExchangeField",
    "AnisotropyField",
    "CubicAnisotropyField",
    "DMIBULK",
    "DMIInterfacial",
    "DemagField",
    "make_demag_field",
    "available_demag_methods",
    "LLG_SOT",
]


def __getattr__(name: str):
    if name == "LLG":
        from .solvers.llg_module import LLG
        return LLG

    if name == "LLG_STT":
        from .solvers.llg_stt_module import LLG_STT
        return LLG_STT

    if name == "LLG_SOT":
        from .solvers.llg_SOT_module import LLG_SOT
        return LLG_SOT

    if name == "EnergyMinimizer":
        from .solvers.Minimizer import EnergyMinimizer
        return EnergyMinimizer

    if name == "ZhangLi":
        from .solvers.Zhang_Li import ZhangLi
        return ZhangLi

    if name == "ExchangeField":
        from .fields.Exchange import ExchangeField
        return ExchangeField

    if name == "AnisotropyField":
        from .fields.Anisotropy import AnisotropyField
        return AnisotropyField


    if name == "CubicAnisotropyField":
        from .fields.Cubic_Anisotropy import CubicAnisotropyField
        return CubicAnisotropyField

    if name == "DMIBULK":
        from .fields.DMI_Bulk import DMIBULK
        return DMIBULK

    if name == "DMIInterfacial":
        from .fields.DMI_Interfacial import DMIInterfacial
        return DMIInterfacial

    if name == "DemagField":
        from .fields.Demag import DemagField
        return DemagField

    if name == "make_demag_field":
        from .fields.Demag import make_demag_field
        return make_demag_field

    if name == "available_demag_methods":
        from .fields.Demag import available_demag_methods
        return available_demag_methods

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
