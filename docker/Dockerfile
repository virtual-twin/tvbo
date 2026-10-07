# syntax=docker/dockerfile:1.7
# Copyright © 2025 Charité Universitätsmedizin Berlin. This software is licensed under the terms of the European Union Public Licence (EUPL) version 1.2 or later.

FROM python:3.13-slim

LABEL maintainer="Leon Martin <leon.martin@bih-charite.de>"

# Bring in uv (single static binary, no Python bootstrap needed).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# System deps: compilers (f2py/PyRates Fortran), cairo (bsplot SVG), curl/wget (Quarto install),
#              default-jre-headless (jNeuroML LEMS runner)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ gfortran make git perl libcairo2-dev curl wget \
        default-jre-headless ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    # NEURON PyPI wheels are built on RHEL and hardcode the GCC toolset path.
    # Create symlinks so nrnmech_makefile finds its compiler on Debian.
    && mkdir -p /opt/rh/gcc-toolset-10/root/usr/bin \
    && ln -s /usr/bin/gcc /opt/rh/gcc-toolset-10/root/usr/bin/gcc \
    && ln -s /usr/bin/g++ /opt/rh/gcc-toolset-10/root/usr/bin/g++

# Build AUTO-07p
RUN git clone --depth 1 https://github.com/auto-07p/auto-07p /auto-07p \
    && cd /auto-07p && ./configure && make && uv pip install --system . \
    && rm -rf /auto-07p/.git /auto-07p/doc /auto-07p/demos

# Install Quarto (multi-arch: detect amd64 vs arm64). Pinned rather than resolved from the releases API, which is unauthenticated and rate-limited: when it answers with nothing the version is empty, the URL it builds is a 404, and the failure reads as a broken download rather than a throttled lookup.
ARG QUARTO_VERSION=1.10.18
RUN ARCH=$(dpkg --print-architecture) \
    && wget -qO quarto.deb "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-${ARCH}.deb" \
    && dpkg -i quarto.deb && rm quarto.deb

# ─── Layered install for maximum cache reuse ──────────────────────────────
# Slow steps (deps + Julia precompile) only depend on pyproject.toml and
# tvbo/juliapkg.json. Source-only changes skip these layers.

WORKDIR /app
ENV UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# 1. Install external deps only (no project build). Cached on pyproject.toml.
#    `uv pip install -r pyproject.toml` reads [project.dependencies] +
#    selected extras directly without invoking the build backend.
#    NOTE: linkml/linkml-runtime are pinned to v1.11.0-rc1 from GitHub (the
#    pip-compat layer of uv does not honor [tool.uv.sources]). Required for
#    SimpleDict-shorthand fixes; remove once 1.11.0 is on PyPI.
COPY pyproject.toml /app/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r pyproject.toml --extra julia --extra pyrates --extra neuroml --extra neuron --extra tvb --extra docs --extra tvboptim \
        jupyterlab uvicorn fastapi \
        "tvboptim @ git+https://github.com/virtual-twin/tvboptim.git@main"

# 2. Pre-warm Julia. juliacall just needs juliapkg.json on sys.path.
ENV JULIA_DEPOT_PATH="/opt/julia-depot" \
    PYTHONPATH="/tmp/jl-stub"
RUN mkdir -p /tmp/jl-stub/tvbo /opt/julia-depot
COPY tvbo/juliapkg.json /tmp/jl-stub/tvbo/juliapkg.json
RUN echo "" > /tmp/jl-stub/tvbo/__init__.py \
    && python3 -c "from juliacall import Main" \
    && chmod -R a+rw /opt/julia-depot
ENV PYTHONPATH=""

# 3. Pre-warm the surface templates a `template:` figure panel draws, which bsplot fetches from templateflow as it renders — a docs run has no network to fetch them over, and a render failure does not fail the run, so the page fails later reading back a figure that was never drawn.
ENV TEMPLATEFLOW_HOME="/opt/templateflow"
RUN python3 -c "import bsplot; from bsplot.data.surface import get_surface_geometry; [(get_surface_geometry(template='fsaverage', hemi=h, density='164k'), bsplot.load_atlas_labels(atlas='Desikan2006', hemi=h, template='fsaverage', density='164k')) for h in ('lh', 'rh')]" \
    && chmod -R a+rw /opt/templateflow

# 4. Copy real source and install tvbo only (deps already satisfied → fast).
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps .

ENV AUTO_DIR=/auto-07p
EXPOSE 8000 8888

RUN cp docker/entrypoint.sh /entrypoint.sh && chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
