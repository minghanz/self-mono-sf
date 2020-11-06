#!/bin/bash

# experiments and datasets meta
KITTI_RAW_HOME="/scratch/hpeng_root/hpeng1/minghanz/kitti_data_all"
EXPERIMENTS_HOME="/scratch/hpeng_root/hpeng1/minghanz/self-mono-sf"

# model
MODEL=MonoSceneFlow_fullmodel

# save path
ALIAS="-eigen-"
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$EXPERIMENTS_HOME/$MODEL$ALIAS$TIME"
CHECKPOINT=None
# CHECKPOINT="/scratch/hpeng_root/hpeng1/minghanz/self-mono-sf/MonoSceneFlow_fullmodel-eigen-20200821-160210/checkpoint_best.ckpt"
# CHECKPOINT="/scratch/hpeng_root/hpeng1/minghanz/self-mono-sf/MonoSceneFlow_fullmodel-eigen-20200821-160210/checkpoint_best.ckpt"
# CHECKPOINT="/scratch/hpeng_root/hpeng1/minghanz/self-mono-sf/MonoSceneFlow_fullmodel-eigen-20201012-123240/checkpoint_epoch_20.ckpt"
# CHECKPOINT="/scratch/hpeng_root/hpeng1/minghanz/self-mono-sf/MonoSceneFlow_fullmodel-eigen-20201024-183030/checkpoint_epoch_60_copy.ckpt"

# Loss and Augmentation
Train_Dataset=KITTI_Raw_EigenSplit_Train_mnsf
Train_Augmentation=Augmentation_SceneFlow_C3D
Train_Loss_Function=Loss_SceneFlow_SelfSup_C3D

Valid_Dataset=KITTI_Raw_EigenSplit_Valid_mnsf
Valid_Augmentation=Augmentation_Resize_Only
Valid_Loss_Function=Loss_SceneFlow_SelfSup

# training configuration
# CUDA_VISIBLE_DEVICES=0 \
CUDA_LAUNCH_BLOCKING=0 \
python ../main.py \
--batch_size_val=1 \
--checkpoint=$CHECKPOINT \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[23, 39, 47, 54]" \
--model=$MODEL \
--num_workers=10 \
--save=$SAVE_PATH \
--total_epochs=62 \
--training_augmentation=$Train_Augmentation \
--training_augmentation_photometric=True \
--training_dataset=$Train_Dataset \
--training_dataset_root=$KITTI_RAW_HOME \
--training_dataset_flip_augmentations=True \
--training_dataset_preprocessing_crop=True \
--training_dataset_num_examples=-1 \
--training_key=t_loss_w_c3d \
--training_loss=$Train_Loss_Function \
--validation_augmentation=$Valid_Augmentation \
--validation_dataset=$Valid_Dataset \
--validation_dataset_root=$KITTI_RAW_HOME \
--validation_dataset_preprocessing_crop=False \
--validation_key=total_loss \
--validation_loss=$Valid_Loss_Function \
--inbalance_to_closer=1.1 \
--save_per_epoch=20 \
--save_pic_tboard \
--disable_c3d \
--dep_warp_loss \
--optimizer=Adam \
--optimizer_lr=2e-4 \
--batch_size=4 \
--pred_xyz_add_uvd \
--strict_flow_acc \
--strict_flow_acc_ramp=10
# --strict_flow_acc_uvd \
# --use_posenet
# --rigid \
# --rigid_pred=full \
# --rigid_warp=res \
# --rigid_pass=warp \
# --rigid_flow_loss \
# --shared_posehead \
# --pose_not_separable
# --c3d_pose_weight=0.1 \
# --hourglass_decoder \
# --optimizer_group="{'params': '*hourglass*', 'lr': 4e-5}" \
# --highres_loss
# --reg_depth
# --c3d_weight 1e-4
# --c3d_knn_mode
# --c3d_config_file="/home/minghanz/repos/self-mono-sf/scripts/c3d_config.txt"
