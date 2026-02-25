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

# Install tvbo from source so tvbo/juliapkg.json is available for juliacall.
# juliacall reads juliapkg.json to know which Julia packages to resolve.
COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir jupyterlab uvicorn fastapi ".[julia,pyrates]" tvboptim

# Julia depot in a fixed, world-readable location so packages pre-compiled
# here are found regardless of runtime HOME or user.
ENV JULIA_DEPOT_PATH="/opt/julia-depot"

# Pre-warm juliacall: importing Main triggers juliapkg to download a compatible
# Julia binary and resolve all packages declared in tvbo/juliapkg.json.
RUN mkdir -p /opt/julia-depot \
    && python3 -c "from juliacall import Main" \
    && chmod -R a+rw /opt/julia-depot

ENV AUTO_DIR=/auto-07p
EXPOSE 8000 8888

# entrypoint.sh was already copied with COPY . /app above
RUN cp docker/entrypoint.sh /entrypoint.sh && chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
