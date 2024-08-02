#!/bin/bash

# script args
# $1: hpc account name: for srun to compile the project
ACCOUNT=$1
PROJDIR=$(dirname $0)

source hpc_utils.sh

MMCACovid19Vac="https://github.com/Epi-Sim/MMCACovid19Vac.jl"

if ! in_hpc_bsc || in_hpc_wifi; then
    echo "Installing MMCACovid19Vac package..."
    if julia +1.10.0 --project=${PROJDIR} -e "using Pkg; Pkg.add(url=\"${MMCACovid19Vac}\"); Pkg.instantiate(); Pkg.precompile()"; then
        echo "Engine installed successfully."
    else
        echo "Engine installation failed. Please check the error messages above."
        exit 1
    fi
else
    echo "Skipping engine installation in this environment."
    echo "Make sure the engines are already installed !!!"
fi

echo "Compiling the package..."
if in_hpc_bsc; then
    if [ -z "$ACCOUNT" ]; then
        echo "Error: Please pass an HPC account name as the first argument."
        exit 1
    fi
    srun --unbuffered \
        -t 00:30:00 \
        -A $ACCOUNT \
        --qos gp_bscls \
        -c 4 -n 1 \
        --output=install_log.txt \
        --error=install_log.txt \
        julia install.jl -c

    # Wait for the log file to be created
    while [ ! -f install_log.txt ]; do
        sleep 1
    done

    # Read from the log file
    tail -f install_log.txt
else
    julia +1.10.0 install.jl -c
fi