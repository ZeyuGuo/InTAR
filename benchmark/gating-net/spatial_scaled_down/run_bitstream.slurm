#!/bin/bash

#SBATCH --output bitstream_run.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --nodes=1
#SBATCH --ntasks=1  ## number of tasks (analyses) to run
#SBATCH -wc02
#SBATCH --cpus-per-task=10  ## the number of threads allocated to each task
#SBATCH --mem=50G   # memory per CPU core
#SBATCH --partition=batch  ## the partitions to run in (comma seperated)
#SBATCH --gpus=u280:1
#SBATCH --qos=quarter

module load tapa
apptainer instance start /opt/public/apptainer/tapa-2024-05-18/tapa.sif tapa
