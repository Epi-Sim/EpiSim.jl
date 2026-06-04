# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.12-slim-bookworm
ARG JULIA_VERSION=1.11.4
ARG JULIA_SHA256=fb3d3c5fccef82158a70677c0044ac5ae40410eceb0604cdc8e643eeff21df8d

FROM ${PYTHON_IMAGE} AS julia-runtime
ARG JULIA_VERSION
ARG JULIA_SHA256

ENV JULIA_HOME=/opt/julia
ENV PATH="${JULIA_HOME}/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libatomic1 \
    libgomp1 \
    libhdf5-103-1 \
    libnetcdf19 \
    tar \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    curl -fsSL "https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" -o /tmp/julia.tar.gz; \
    echo "${JULIA_SHA256}  /tmp/julia.tar.gz" | sha256sum -c -; \
    mkdir -p "${JULIA_HOME}"; \
    tar -xzf /tmp/julia.tar.gz -C "${JULIA_HOME}" --strip-components=1; \
    rm /tmp/julia.tar.gz; \
    julia --version

FROM julia-runtime AS episim-build
ARG SHOULD_COMPILE=false

ENV JULIA_PROJECT=/opt/episim
ENV JULIA_DEPOT_PATH=/opt/julia-depot
ENV JULIA_PKG_PRECOMPILE_AUTO=0
ENV UV_PROJECT_ENVIRONMENT=/usr/local
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libhdf5-dev \
    libnetcdf-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /opt/episim

COPY Project.toml Manifest.toml install.jl README.md LICENSE ./
COPY src ./src

RUN julia -e 'using Pkg; Pkg.Registry.add("General"); Pkg.instantiate(); Pkg.precompile()'

COPY python/pyproject.toml python/uv.lock python/README.md ./python/
COPY python/episim_python ./python/episim_python

RUN cd python && uv sync --locked --no-dev --no-editable --no-cache

COPY docker/episim-batch docker/episim-entrypoint /usr/local/bin/
RUN chmod +x /usr/local/bin/episim-batch /usr/local/bin/episim-entrypoint

COPY . .

RUN mkdir -p build && julia -e 'using Pkg; Pkg.precompile()'
RUN if [ "${SHOULD_COMPILE}" = "true" ]; then \
        julia install.jl --compile --incremental --target /usr/local/bin; \
    else \
        julia install.jl --incremental --target /usr/local/bin; \
    fi

FROM julia-runtime AS episim-runtime
ARG OCI_SOURCE="https://github.com/Epi-Sim/EpiSim.jl"
ARG OCI_REVISION="unknown"
ARG OCI_VERSION="0.1.1"

LABEL org.opencontainers.image.title="EpiSim.jl" \
      org.opencontainers.image.description="Runtime image for EpiSim.jl epidemic simulations" \
      org.opencontainers.image.source="${OCI_SOURCE}" \
      org.opencontainers.image.revision="${OCI_REVISION}" \
      org.opencontainers.image.version="${OCI_VERSION}" \
      org.opencontainers.image.licenses="MIT"

ENV JULIA_PROJECT=/opt/episim
ENV EPISIM_EXECUTABLE_PATH=/usr/local/bin/episim
ENV JULIA_DEPOT_PATH=/opt/julia-depot
ENV JULIA_PKG_PRECOMPILE_AUTO=0
ENV EPISIM_INSTANTIATE_ON_STARTUP=0
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/episim

COPY --from=episim-build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
RUN rm -rf /usr/local/lib/python3.12/site-packages/uv /usr/local/lib/python3.12/site-packages/uv-*.dist-info
COPY --from=episim-build /usr/local/bin/episim /usr/local/bin/episim
COPY --from=episim-build /usr/local/bin/episim-batch /usr/local/bin/episim-batch
COPY --from=episim-build /usr/local/bin/episim-entrypoint /usr/local/bin/episim-entrypoint
COPY --from=episim-build /opt/julia-depot /opt/julia-depot
RUN mkdir -p /opt/julia-depot/logs \
    && chmod -R a+rwX /opt/julia-depot
COPY --from=episim-build /opt/episim/Project.toml /opt/episim/Manifest.toml /opt/episim/install.jl /opt/episim/README.md /opt/episim/LICENSE ./
COPY --from=episim-build /opt/episim/src ./src
COPY --from=episim-build /opt/episim/build ./build
COPY --from=episim-build /opt/episim/python/pyproject.toml /opt/episim/python/uv.lock /opt/episim/python/README.md ./python/
COPY --from=episim-build /opt/episim/python/episim_python ./python/episim_python

ENTRYPOINT ["episim-entrypoint"]
CMD ["--help"]

FROM episim-runtime AS episim-test

ENV UV_PROJECT_ENVIRONMENT=/usr/local

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libhdf5-dev \
    libnetcdf-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY --from=episim-build /usr/local/bin/uv /usr/local/bin/uv

COPY --from=episim-build /opt/episim/models ./models
COPY --from=episim-build /opt/episim/test ./test
COPY --from=episim-build /opt/episim/python ./python

RUN cd python && uv sync --locked --all-extras --group dev --no-editable --no-cache
