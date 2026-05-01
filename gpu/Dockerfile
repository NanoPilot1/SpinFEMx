FROM benpachev/cudolfinx:v0.10.0-cuda12.6

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1
ARG JAXFMM_REF=strayfield

# Basic system dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Base Python packaging tools.
RUN python3 -m pip install --upgrade \
    pip \
    setuptools \
    wheel

# Main Python dependencies.
RUN python3 -m pip install \
    pyvista \
    cupy-cuda12x \
    adios4dolfinx \
    "jax[cuda12]"

# CUDA runtime headers may be required by CuPy/NVRTC on CUDA 12.x.
RUN python3 -m pip install \
    "nvidia-cuda-runtime-cu12==12.6.*"

# JAXFMM strayfield branch/ref.
RUN python3 -m pip install --no-deps \
    "jaxFMM @ git+https://gitlab.com/jaxfmm/jaxfmm.git@${JAXFMM_REF}"

# Wrapper for controlling the runtime library path used by Python/JAX/DOLFINx.
RUN cat > /usr/local/bin/run_clean_gpu_env <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")

export LD_LIBRARY_PATH="/usr/local/dolfinx-real/lib:/usr/local/lib:\
$SITE/nvidia/nvjitlink/lib:\
$SITE/nvidia/cuda_runtime/lib:\
$SITE/nvidia/cublas/lib:\
$SITE/nvidia/cusparse/lib:\
$SITE/nvidia/cusolver/lib:\
$SITE/nvidia/cudnn/lib:\
$SITE/nvidia/cufft/lib:\
$SITE/nvidia/nccl/lib"

exec "$@"
EOF

# Normalize line endings in case the Dockerfile was edited on Windows.
RUN sed -i 's/\r$//' /usr/local/bin/run_clean_gpu_env && \
    chmod +x /usr/local/bin/run_clean_gpu_env

# Basic import check.
# This does not require GPU access during build, since docker build usually
# does not expose the host GPU.
RUN run_clean_gpu_env python3 - <<'PY'
import jax
import cupy
import jaxfmm

print("JAX version:", jax.__version__)
print("JAX backend during build:", jax.default_backend())
print("CuPy version:", cupy.__version__)
print("jaxfmm:", jaxfmm.__file__)
PY

# Copy and install the local package.
COPY pyproject.toml /opt/micromag_project/pyproject.toml
COPY src /opt/micromag_project/src

RUN run_clean_gpu_env python3 -m pip install -e /opt/micromag_project

# Check Python dependency consistency.
RUN run_clean_gpu_env python3 -m pip check

# Safe package metadata check.
RUN run_clean_gpu_env python3 - <<'PY'
import importlib.metadata as md

print("micromagnetic-gpu installed:", md.version("micromagnetic-gpu"))
PY

WORKDIR /workspace
