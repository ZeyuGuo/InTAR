#!/bin/bash

#SBATCH -J vae_hw_emu
#SBATCH -n32
#SBATCH --mem=80G
#SBATCH --qos=lowest
#SBATCH -p batch
#SBATCH --nodelist=c01

echo 'Your job is running on node(s):'
echo $SLURM_JOB_NODELIST
echo 'Cores per node:'
echo $SLURM_TASKS_PER_NODE

rm -rf /scratch/$USER/vae

module load xilinx/vivado/2021.2 tapa
rsync -r ~/projects/Llama2Lite/benchmark/vae/sequential /scratch/$USER/vae/ --exclude ".git"
cd /scratch/$USER/vae/sequential/
make build_hw_emu
rsync -r /scratch/$USER/vae/sequential/work.out/ ~/projects/Llama2Lite/benchmark/vae/sequential/work.out/ --exclude ".git" --remove-source-files
rsync -r /scratch/$USER/vae/sequential/bitstreams/ ~/projects/Llama2Lite/benchmark/vae/sequential/bitstreams/ --exclude ".git" --remove-source-files
