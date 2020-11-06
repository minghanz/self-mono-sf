from __future__ import absolute_import, division, print_function

import os
import logging
import subprocess
import torch
from core import commandline, runtime, logger, tools, configuration as config

class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        return getattr(self.module, name)

def main():
    
    # Change working directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Parse commandline arguments
    args = commandline.setup_logging_and_parse_arguments(blocktitle="Commandline Arguments")

    # Set random seed, possibly on Cuda
    config.configure_random_seed(args)
    
    ################ Modified to improve speed: data augmentation in dataset
    # Configure data augmentation
    training_augmentation, validation_augmentation = config.configure_runtime_augmentations(args)

    # DataLoader
    train_loader, validation_loader, inference_loader = config.configure_data_loaders(args, training_augmentation, validation_augmentation)
    success = any(loader is not None for loader in [train_loader, validation_loader, inference_loader])
    if not success:
        logging.info("No dataset could be loaded successfully. Please check dataset paths!")
        quit()
    ### If do the original augmentation part after dataloader (in the training loop), use these lines.
    ### (This is preferred because the original augmentation involves big array manipulation, which is faster on GPU (in our experiment, 0.05s vs 0.003s), 
    ###  while the dataset __getitem__ cannot involve GPU processing, because it will cause error in multiprocessing (when num_workers>0)
    ### But the other option is still possible, because multiprocessing can mitigate the delay. )
    if args.cuda:
        if training_augmentation is not None:
            training_augmentation = training_augmentation.cuda()
        if validation_augmentation is not None:
            validation_augmentation = validation_augmentation.cuda()  
    ### If do the original augmentation part in the dataset, use these lines.
    # training_augmentation = None
    # validation_augmentation = None

    ################

    # ################# Original: data augmentation after dataloader
    # # DataLoader
    # train_loader, validation_loader, inference_loader = config.configure_data_loaders(args)
    # success = any(loader is not None for loader in [train_loader, validation_loader, inference_loader])
    # if not success:
    #     logging.info("No dataset could be loaded successfully. Please check dataset paths!")
    #     quit()

    # # Configure data augmentation
    # training_augmentation, validation_augmentation = config.configure_runtime_augmentations(args)
    # #################

    # Configure model and loss
    model_and_loss = config.configure_model_and_loss(args)

    # Resume from checkpoint if available 
    checkpoint_saver, checkpoint_stats = config.configure_checkpoint_saver(args, model_and_loss)
    
    if checkpoint_stats is not None:
        # Set checkpoint stats
        if args.checkpoint_mode in ["resume_from_best", "resume_from_latest"]:
            args.start_epoch = checkpoint_stats["epoch"] + 1

    # Multi-GPU automation
    with logger.LoggingBlock("Multi GPU", emph=True):
        if torch.cuda.device_count() > 1:
            logging.info("Let's use %d GPUs!" % torch.cuda.device_count())
            model_and_loss._model = torch.nn.DataParallel(model_and_loss._model)
            # model_and_loss._training_loss = torch.nn.DataParallel(model_and_loss._training_loss)
            # model_and_loss = MyDataParallel(model_and_loss)
            # model_and_loss._model = torch.nn.parallel.DistributedDataParallel(model_and_loss._model)
        else:
            logging.info("Let's use %d GPU!" % torch.cuda.device_count())


    # Checkpoint and save directory
    with logger.LoggingBlock("Save Directory", emph=True):
        logging.info("Save directory: %s" % args.save)
        if not os.path.exists(args.save):
            os.makedirs(args.save)

    # Configure optimizer
    optimizer = config.configure_optimizer(args, model_and_loss)

    # Configure learning rate
    lr_scheduler = config.configure_lr_scheduler(args, optimizer)

    # If this is just an evaluation: overwrite savers and epochs
    if args.evaluation:
        args.start_epoch = 1
        args.total_epochs = 1
        train_loader = None
        checkpoint_saver = None
        optimizer = None
        lr_scheduler = None

    # Cuda optimization
    if args.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Kickoff training, validation and/or testing
    # torch.autograd.set_detect_anomaly(True)       ### Remember to comment this line when training: affect speed more than 20x (0.3s vs 7s) !!!

    return runtime.exec_runtime(
        args,
        checkpoint_saver=checkpoint_saver,
        model_and_loss=model_and_loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        validation_loader=validation_loader,
        inference_loader=inference_loader,
        training_augmentation=training_augmentation,
        validation_augmentation=validation_augmentation)

if __name__ == "__main__":
    main()
