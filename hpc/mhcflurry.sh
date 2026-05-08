#!/usr/bin/env bash
#PBS -l walltime=4:00:00
#PBS -l select=1:ncpus=1:mem=50gb
#PBS -N aav_mhcflurry
#PBS -o hpc/logs/01_mhcflurry.o
#PBS -e hpc/logs/01_mhcflurry.e

cd "$PBS_O_WORKDIR"

mkdir -p hpc/logs data/output/fragmentation data/output/mhcflurry

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate AAV

export MHCFLURRY_DATA_DIR="/rds/general/project/hda_25-26/live/TDS/rowzang/final_project/mhcflurry"
mkdir -p "$MHCFLURRY_DATA_DIR"

echo "PWD=$(pwd)"
echo "MHCFLURRY_DATA_DIR=$MHCFLURRY_DATA_DIR"
mhcflurry-downloads path || true


python scripts/run_mhcflurry.py