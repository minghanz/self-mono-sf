#!/bin/bash
#SBATCH --job-name CPU_FLOW_C3D
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --account=hpeng1
###SBATCH --partition=gpu
#SBATCH --mail-type=NONE
###SBATCH --gpus-per-node=1
###SBATCH --gpus=1
###SBATCH --cpus-per-gpu=18
###SBATCH --mem-per-gpu=16g
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6g
#SBATCH --get-user-env

### # SBATCH --cpus-per-task=1
# conda init bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tp36dup
# conda activate pytorch3d
# CUDA_VISIBLE_DEVICES=0 
python ../datasets/toy_test_flow_disp_pose.py