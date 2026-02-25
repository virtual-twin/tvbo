# Copyright © 2025 Charité Universitätsmedizin Berlin. This software is licensed under the terms of the European Union Public Licence (EUPL) version 1.2 or later.

FROM julia:latest AS julia
FROM python:3.13-slim

LABEL maintainer="Leon Martin <leon.martin@bih-charite.de>"

# Copy Julia from official image
COPY --from=julia /usr/local/julia /usr/local/julia
ENV PATH="/usr/local/julia/bin:$PATH"

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

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir jupyterlab uvicorn fastapi tvbo[julia,pyrates] tvboptim

# Pre-install Julia packages for NetworkDynamics, BifurcationKit, ModelingToolkit docs
RUN julia -e '\
    using Pkg; \
    pkgs = ["Graphs","NetworkDynamics","OrdinaryDiffEqTsit5", \
            "OrdinaryDiffEqSDIRK","SimpleWeightedGraphs","StochasticDiffEq", \
            "BifurcationKit","ModelingToolkit","Plots"]; \
    for p in pkgs; Pkg.add(p); end; \
    Pkg.precompile()'

ENV AUTO_DIR=/auto-07p
WORKDIR /app
EXPOSE 8000 8888

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
