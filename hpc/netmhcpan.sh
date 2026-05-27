#!/usr/bin/env bash
#PBS -l walltime=8:00:00
#PBS -l select=1:ncpus=1:mem=50gb
#PBS -N aav_netmhcpan
#PBS -o hpc/logs/02_netmhcpan.o
#PBS -e hpc/logs/02_netmhcpan.e

cd "$PBS_O_WORKDIR"

mkdir -p hpc/logs data/output/fragmentation data/output/netmhcpan

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate AAV

export TMPDIR="/rds/general/user/rma25/projects/hda_25-26/live/TDS/rowzang/final_project/netMHCpan-4.2/tmp"
mkdir -p "$TMPDIR"

chmod u+x /rds/general/user/rma25/projects/hda_25-26/live/TDS/rowzang/final_project/netMHCpan-4.2/Linux_x86_64/bin/*

python scripts/run_netmhc.py