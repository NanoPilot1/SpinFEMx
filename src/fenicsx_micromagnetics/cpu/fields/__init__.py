"""
CPU field terms for FEniCSx-Micromagnetics.
"""

from __future__ import annotations

__all__ = [
    "ExchangeField",
    "AnisotropyField",
    "CubicAnisotropyField",
    "DMIBULK",
    "DMIInterfacial",
    "DemagField",
    "make_demag_field",
    "available_demag_methods",
]


def __getattr__(name: str):
    if name == "ExchangeField":
        from .Exchange import ExchangeField
        return ExchangeField

    if name == "AnisotropyField":
        from .Anisotropy import AnisotropyField
        return AnisotropyField

    if name == "CubicAnisotropyField":
        from .Cubic_Anisotropy import CubicAnisotropyField
        return CubicAnisotropyField

    if name == "DMIBULK":
        from .DMI_Bulk import DMIBULK
        return DMIBULK

    if name == "DMIInterfacial":
        from .DMI_Interfacial import DMIInterfacial
        return DMIInterfacial

    if name == "DemagField":
        from .Demag import DemagField
        return DemagField

    if name == "make_demag_field":
        from .Demag import make_demag_field
        return make_demag_field

    if name == "available_demag_methods":
        from .Demag import available_demag_methods
        return available_demag_methods

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")