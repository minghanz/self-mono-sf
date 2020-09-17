#!/bin/bash
#SBATCH --job-name TB_FLOW_C3D
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --account=hpeng1
###SBATCH --partition=gpu
#SBATCH --mail-type=END,FAILNE
###SBATCH --gpus-per-node=1
###SBATCH --gpus=1
###SBATCH --cpus-per-gpu=18
###SBATCH --mem-per-gpu=16g
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6g
#SBATCH --get-user-env

### # SBATCH --cpus-per-task=1
### https://gist.github.com/taylorpaul/250ee3ed2524e8c28ee7c58ed656a5b9
# conda init bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tp36dup

### This is to set the port and print it
let ipnport=($UID-6025)%65274
echo ipnport=$ipnport

### This is to print the host name
ipnip=$(hostname -i)
echo ipnip=$ipnip

tensorboard --logdir /scratch/hpeng_root/hpeng1/minghanz/self-mono-sf --port=$ipnport
# ./train_monosf_selfsup_c3d_eigen_train.sh
# python bts_test_dataloader.py arguments_train_eigen_c3d.txt