#!/usr/bin/env bash
#PBS -l walltime=4:00:00
#PBS -l select=1:ncpus=1:mem=50gb
#PBS -N aav_fragmentation
#PBS -o hpc/logs/01_fragment.o
#PBS -e hpc/logs/01_fragment.e

cd /general/user/rma25/projects/hda_25-26/live/TDS/rowzang/final_project/aav-tcr-epitope-pipeline

mkdir -p hpc/logs data/output/fragmentation

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate AAV

python scripts/run_fragmentation.py