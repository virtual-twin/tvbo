# Copyright © 2025 Charité Universitätsmedizin Berlin. This software is licensed under the terms of the European Union Public Licence (EUPL) version 1.2 or later.

FROM python:3.13-slim

LABEL maintainer="Leon Martin <leon.martin@bih-charite.de>"

# System deps: compilers (f2py/PyRates Fortran), cairo (bsplot SVG), curl/wget (Quarto install)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ gfortran make git perl libcairo2-dev curl wget \
    && rm -rf /var/lib/apt/lists/*

# Build AUTO-07p
RUN git clone --depth 1 https://github.com/auto-07p/auto-07p /auto-07p \
    && cd /auto-07p && ./configure && make && pip install . \
    && rm -rf /auto-07p/.git /auto-07p/doc /auto-07p/demos

# Install Quarto (multi-arch: detect amd64 vs arm64)
RUN QUARTO_VERSION=$(curl -s https://api.github.com/repos/quarto-dev/quarto-cli/releases/latest | grep -oP '"tag_name": "v\K[^"]+') \
    && ARCH=$(dpkg --print-architecture) \
    && wget -qO quarto.deb "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-${ARCH}.deb" \
    && dpkg -i quarto.deb && rm quarto.deb

# ─── Layered install for maximum cache reuse ──────────────────────────────
# The slow steps (pip install of all deps + Julia precompile) only depend on
# pyproject.toml and tvbo/juliapkg.json. By copying those FIRST, source-only
# changes (Python edits, docs, tests) skip rebuilding these layers entirely.

WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip

# 1. Stub package skeleton: enough for hatchling to read __version__ and for
#    juliacall to find tvbo/juliapkg.json on sys.path.
COPY pyproject.toml /app/pyproject.toml
COPY tvbo/juliapkg.json /app/tvbo/juliapkg.json
RUN echo '__version__ = "0.0.0"' > /app/tvbo/__init__.py \
    && touch /app/README.md

# 2. Install all deps (heavy layer, cached on pyproject.toml hash).
RUN pip install --no-cache-dir jupyterlab uvicorn fastapi ".[julia,pyrates]" tvboptim

# 3. Pre-warm Julia. Cached on tvbo/juliapkg.json hash, NOT on Python source.
ENV JULIA_DEPOT_PATH="/opt/julia-depot" \
    JULIA_NUM_PRECOMPILE_TASKS="auto"
RUN mkdir -p /opt/julia-depot \
    && python3 -c "from juliacall import Main" \
    && chmod -R a+rw /opt/julia-depot

# 4. Now copy the real source and reinstall tvbo only (no deps → instant).
COPY . /app
RUN pip install --no-cache-dir --no-deps --force-reinstall .

ENV AUTO_DIR=/auto-07p
EXPOSE 8000 8888

RUN cp docker/entrypoint.sh /entrypoint.sh && chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
