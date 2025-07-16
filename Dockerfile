# Single-stage build with Python base
FROM python:3.12

# Install system dependencies for NetCDF, HDF5, and Julia
RUN apt-get update && apt-get install -y \
    libnetcdf-dev \
    libhdf5-dev \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Julia runtime
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.4-linux-x86_64.tar.gz && \
    tar -xvf julia-1.11.4-linux-x86_64.tar.gz && \
    mv julia-1.11.4 /opt/julia && \
    ln -s /opt/julia/bin/julia /usr/local/bin/julia && \
    rm julia-1.11.4-linux-x86_64.tar.gz

# Set working directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY Project.toml Manifest.toml ./
COPY python/pyproject.toml python/

# Set Julia environment and install dependencies
ENV JULIA_PROJECT=/app

# Initialize Julia registries and install Julia dependencies (but not EpiSim itself yet)
RUN julia -e "using Pkg; Pkg.Registry.add(\"General\")"
RUN julia -e "using Pkg; Pkg.instantiate()"

# Install uv for Python package management
RUN pip install uv

# Copy rest of the source code
COPY . .

# Install Python dependencies in editable mode after source code is copied
RUN cd python && uv pip install --system -e .

# Now precompile everything including EpiSim
RUN julia -e "using Pkg; Pkg.precompile()"

# Compile the Julia application
RUN julia install.jl -i -t /usr/local/bin

# Set environment variables for executable discovery
ENV EPISIM_EXECUTABLE_PATH=/usr/local/bin/episim
ENV EPISIM_JULIA_PROJECT=/app
ENV JULIA_PROJECT=/app

# Default command shows help
CMD ["julia", "--project", "src/run.jl", "--help"] 
