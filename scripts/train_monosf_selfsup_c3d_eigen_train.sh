#!/bin/bash

# experiments and datasets meta
KITTI_RAW_HOME="/mnt/storage8t/minghanz/Datasets/KITTI_data/kitti_data"
EXPERIMENTS_HOME="/mnt/storage8t/minghanz/self_mono_sf_tmp"

# model
MODEL=MonoSceneFlow_fullmodel

# save path
ALIAS="-eigen-"
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$EXPERIMENTS_HOME/$MODEL$ALIAS$TIME"
CHECKPOINT=None

# Loss and Augmentation
Train_Dataset=KITTI_Raw_EigenSplit_Train_mnsf
Train_Augmentation=Augmentation_SceneFlow_C3D
Train_Loss_Function=Loss_SceneFlow_SelfSup_C3D

Valid_Dataset=KITTI_Raw_EigenSplit_Valid_mnsf
Valid_Augmentation=Augmentation_Resize_Only
Valid_Loss_Function=Loss_SceneFlow_SelfSup

# training configuration
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=0 \
python ../main.py \
--batch_size=2 \
--batch_size_val=1 \
--checkpoint=$CHECKPOINT \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[23, 39, 47, 54]" \
--model=$MODEL \
--num_workers=8 \
--optimizer=Adam \
--optimizer_lr=2e-4 \
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