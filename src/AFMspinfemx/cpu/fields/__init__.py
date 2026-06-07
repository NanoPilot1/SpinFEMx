"""
CPU field terms for two-sublattice AFM simulations
"""

from __future__ import annotations

__all__ = [
    "ExchangeField",
    "AnisotropyField",
    "DMIInterfacialField",
    "DMIBulkField",
]


def __getattr__(name: str):

    if name == "ExchangeField":
        from .Exchange_AFM import ExchangeField
        return ExchangeField

    if name == "AnisotropyField":
        from .Anisotropy_AFM import AnisotropyField
        return AnisotropyField

    
    if name == "DMIBulkField":
        from .DMI_Bulk_AFM import DMIBulkField
        return DMIBulkField
    
    if name == "DMIInterfacialField":
        from .DMI_Interfacial_AFM import DMIInterfacialField
        return DMIInterfacialField

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
