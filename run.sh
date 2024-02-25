#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=16
#SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --time=00:30:00
#SBATCH --mem=128GB
#SBATCH -o byop-%j
#SBATCH --export=ALL

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/uufs/chpc.utah.edu/common/home/u1472614/miniconda3/lib
export WORKDIR="$HOME/WORK/NLP-with-Deep-Learning/BYOP"
export seed=128
export lr=0.00001
export dr=0.3
export ep=2
export md="visual"
#export SCRDIR="/scratch/general/vast/$USER/byop_data_{$SLURM_JOBID}_bs_16"
export SCRDIR="/scratch/general/vast/$USER/byop_{$lr}_dr{$dr}_{$SLURM_JOBID}_text"
export models="Models"
export it=1


mkdir -p $SCRDIR
cp -r $WORKDIR/* $SCRDIR
cd $SCRDIR
mkdir $models


source ~/miniconda3/etc/profile.d/conda.sh
conda activate envKD
python ./new_proj.py --learning_rate $lr --dropout $dr --epochs $ep --model $md > my_out
#python ./new_proj.py --iteration $it > my_out
