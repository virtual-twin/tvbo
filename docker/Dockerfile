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

# Install Python packages (juliacall is included via tvbo[julia] and downloads
# its own compatible Julia via juliapkg — no system Julia installation needed)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir jupyterlab uvicorn fastapi tvbo[julia,pyrates] tvboptim

# Julia depot in a fixed, world-readable location so packages pre-compiled
# here are found regardless of runtime HOME or user.
ENV JULIA_DEPOT_PATH="/opt/julia-depot"

# Pre-warm juliacall: triggers juliapkg to download a compatible Julia binary
# and pre-installs all required Julia packages so CI/runtime has zero download cost.
COPY prewarm_julia.py /tmp/prewarm_julia.py
RUN mkdir -p /opt/julia-depot \
    && python3 /tmp/prewarm_julia.py \
    && chmod -R a+rw /opt/julia-depot \
    && rm /tmp/prewarm_julia.py

ENV AUTO_DIR=/auto-07p
WORKDIR /app
EXPOSE 8000 8888

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
