"""
GPU field terms for SpinFEMx.
"""

from __future__ import annotations

__all__ = [
    "ExchangeField",
    "AnisotropyField",
    "CubicAnisotropyField",
    "DMIBULK",
    "DMIInterfacial",
    "DemagFieldFMMJAXGPU",
    "DemagFieldLindholmGPU",
]


def __getattr__(name: str):
    if name == "ExchangeField":
        from .Exchange_GPU import ExchangeField
        return ExchangeField

    if name == "AnisotropyField":
        from .Anisotropy_GPU import AnisotropyField
        return AnisotropyField

    if name == "CubicAnisotropyField":
        from .Cubic_Anisotropy_GPU import CubicAnisotropyFieldGPU
        return CubicAnisotropyFieldGPU

    if name == "DMIBULK":
        from .DMI_Bulk_GPU import DMIBULK
        return DMIBULK

    if name == "DMIInterfacial":
        from .DMI_Interfacial_GPU import DMIInterfacial
        return DMIInterfacial

    if name == "DemagFieldFMMJAXGPU":
        from .Demag_FMM_GPU import DemagFieldFMMJAXGPU
        return DemagFieldFMMJAXGPU

    if name == "DemagFieldLindholmGPU":
        from .Demag_Lindholm_GPU import DemagFieldLindholmGPU
        return DemagFieldLindholmGPU

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")