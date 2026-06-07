# Demag/__init__.py

from __future__ import annotations

import importlib
import importlib.util


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _check_fmm_dependencies():
    missing = []

    if not _has_module("jax"):
        missing.append("jax")

    if not _has_module("jaxfmm"):
        missing.append("jaxfmm")

    if missing:
        raise ImportError(
            "demag_method='fmm' requires optional dependencies that are not installed: "
            + ", ".join(missing)
            + ". Install them or choose another demag_method, e.g. 'lindholm', 'bempp' or 'htool'."
        )


def _check_htool_dependencies():
    if not _has_module("petsc4py"):
        raise ImportError(
            "demag_method='htool' requires petsc4py, but petsc4py is not installed."
        )

    try:
        from petsc4py import PETSc
    except Exception as e:
        raise ImportError(
            "demag_method='htool' requires petsc4py/PETSc, but PETSc could not be imported."
        ) from e

    # Best available check: PETSc must be compiled with Htool.
    try:
        has_htool = PETSc.Sys.hasExternalPackage("htool")
    except Exception:
        has_htool = False

    if not has_htool:
        raise ImportError(
            "demag_method='htool' requires PETSc compiled with Htool support. "
            "Your current PETSc installation does not report Htool support. "
            "Build PETSc from source with Htool enabled, then rebuild petsc4py against that PETSc."
        )


_BACKENDS = {
    "lindholm": {
        "module": ".Demag_Lindholm",
        "class": "DemagField_Lindholm",
        "check": None,
    },
    "dense": {
        "module": ".Demag_Lindholm",
        "class": "DemagField_Lindholm",
        "check": None,
    },
    "bempp": {
        "module": ".Demag_Bempp",
        "class": "DemagField_bempp",
        "check": None,
    },
    "fmm": {
        "module": ".Demag_FMM",
        "class": "DemagField_FMM",
        "check": _check_fmm_dependencies,
    },
    "jax_fmm": {
        "module": ".Demag_FMM",
        "class": "DemagField_FMM",
        "check": _check_fmm_dependencies,
    },
    "htool": {
        "module": ".Demag_Lindholm_HTool_MPI",
        "class": "DemagField_Lindholm_HTool_MPI",
        "check": _check_htool_dependencies,
    },
    "lindholm_htool": {
        "module": ".Demag_Lindholm_HTool_MPI",
        "class": "DemagField_Lindholm_HTool_MPI",
        "check": _check_htool_dependencies,
    },
    "lindholm_htool_mpi": {
        "module": ".Demag_Lindholm_HTool_MPI",
        "class": "DemagField_Lindholm_HTool_MPI",
        "check": _check_htool_dependencies,
    },
}


def available_demag_methods():
    """
    Returns the registered demag methods. This does not import heavy optional backends.
    """
    return list(_BACKENDS.keys())


def _load_demag_class(method: str):
    key = (method or "lindholm").strip().lower()

    if key not in _BACKENDS:
        raise ValueError(
            f"demag_method='{method}' is not supported. "
            f"Available options are: {list(_BACKENDS.keys())}"
        )

    spec = _BACKENDS[key]

    check = spec["check"]
    if check is not None:
        check()

    try:
        mod = importlib.import_module(spec["module"], package=__name__)
    except Exception as e:
        raise ImportError(
            f"Could not import backend for demag_method='{method}'. "
            f"Backend module: {spec['module']}."
        ) from e

    try:
        return getattr(mod, spec["class"])
    except AttributeError as e:
        raise ImportError(
            f"Backend module '{spec['module']}' was imported, but class "
            f"'{spec['class']}' was not found."
        ) from e


def make_demag_field(method, mesh, V, V1, Ms, **kwargs):
    cls = _load_demag_class(method)
    return cls(mesh, V, V1, Ms, **kwargs)


# Backward compatibility:
# Avoid importing the Lindholm class immediately unless someone explicitly asks for DemagField.
def __getattr__(name):
    if name == "DemagField":
        return _load_demag_class("lindholm")

    if name == "DemagField_Lindholm":
        return _load_demag_class("lindholm")

    if name == "DemagField_bempp":
        return _load_demag_class("bempp")

    if name == "DemagField_FMM":
        return _load_demag_class("fmm")

    if name == "DemagField_Lindholm_HTool_MPI":
        return _load_demag_class("htool")

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")