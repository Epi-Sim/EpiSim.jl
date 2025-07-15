# Use Julia official image as base
FROM julia:1.11.4

# Install system dependencies for NetCDF and HDF5
RUN apt-get update && apt-get install -y \
    libnetcdf-dev \
    libhdf5-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Set Julia environment and install dependencies
ENV JULIA_PROJECT=/app

# Initialize Julia registries and install dependencies step by step
RUN julia -e "using Pkg; Pkg.Registry.add(\"General\")"
RUN julia -e "using Pkg; Pkg.instantiate()"
RUN julia -e "using Pkg; Pkg.precompile()"

# Now run the install script to compile the application
RUN julia install.jl -i -t /usr/local/bin

# Default command shows help
CMD ["julia", "--project", "src/run.jl", "--help"] 
