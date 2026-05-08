#!/usr/bin/env bash
#PBS -l walltime=4:00:00
#PBS -l select=1:ncpus=1:mem=50gb
#PBS -N aav_mhcflurry
#PBS -o logs/01_mhcflurry.o
#PBS -e logs/01_mhcflurry.e

cd /general/user/rma25/projects/hda_25-26/live/TDS/rowzang/final_project/aav-tcr-epitope-pipeline

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate AAV

export MHCFLURRY_DATA_DIR="/general/user/rma25/projects/hda_25-26/live/TDS/rowzang/final_project/mhcflurry_data"

python scripts/run_mhcflurry.py