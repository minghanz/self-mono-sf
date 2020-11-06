#!/bin/bash
#SBATCH --job-name EVAL_FLOW_C3D
#SBATCH --nodes=1
#SBATCH --time=200:00:00
#SBATCH --account=hpeng1
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAILNE
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=12
###SBATCH --mem-per-gpu=16g
#SBATCH --mem-per-cpu=2g
#SBATCH --get-user-env

### # SBATCH --cpus-per-task=1
# conda init bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tp36dup
# CUDA_VISIBLE_DEVICES=0 
./eval_monosf_selfsup_kitti_train.sh
./eval_monodepth_selfsup_kitti_train.sh
# ./eval_monodepth_selfsup_eigen_test.sh
# python bts_test_dataloader.py arguments_train_eigen_c3d.txt