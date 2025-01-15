#!/bin/bash

#SBATCH -J mlp_hw
#SBATCH -n16
#SBATCH --mem=40G
#SBATCH --qos=lowest
#SBATCH -p batch

echo 'Your job is running on node(s):'
echo $SLURM_JOB_NODELIST
echo 'Cores per node:'
echo $SLURM_TASKS_PER_NODE

rm -rf /scratch/$USER/mlp

module load xilinx/vivado/2021.2 tapa
rsync -r ~/projects/Llama2Lite/benchmark/mlp/spatial /scratch/$USER/mlp/ --exclude ".git"
cd /scratch/$USER/mlp/spatial/
make build_hw
rsync -r /scratch/$USER/mlp/spatial/work.out/ ~/projects/Llama2Lite/benchmark/mlp/spatial/work.out/ --exclude ".git" --remove-source-files
rsync -r /scratch/$USER/mlp/spatial/bitstreams/ ~/projects/Llama2Lite/benchmark/mlp/spatial/bitstreams/ --exclude ".git" --remove-source-files
