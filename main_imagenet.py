#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
import os
import time
import argparse

import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torchvision.models as models
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboard_logger import log_value

import utils
from dataset.imagenet_dataset import ImageFolderWithIdx
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training With Data Parameters')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--job_name', default='temp', help='Job name used to create save directories for '
                                                       'checkpoints and logs')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--restart', default=False, const=True, action='store_const',
                    help='Erase log and saved checkpoints and restart training')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate for model parameters', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--learn_class_parameters', default=False, const=True, action='store_const',
                    help='Learn temperature per class')
parser.add_argument('--learn_inst_parameters', default=False, const=True, action='store_const',
                    help='Learn temperature per instance')
parser.add_argument('--skip_clamp_data_param', default=False, const=True, action='store_const',
                    help='Do not clamp data parameters during optimization')
parser.add_argument('--lr_class_param', default=0.1, type=float, help='Learning rate for class parameters')
parser.add_argument('--lr_inst_param', default=0.1, type=float, help='Learning rate for instance parameters')
parser.add_argument('--wd_class_param', default=0.0, type=float, help='Weight decay for class parameters')
parser.add_argument('--wd_inst_param', default=0.0, type=float, help='Weight decay for instance parameters')
parser.add_argument('--init_class_param', default=1.0, type=float, help='Initial value for class parameters')
parser.add_argument('--init_inst_param', default=1.0, type=float, help='Initial value for instance parameters')
parser.add_argument('--lr_drop_epoch_step', default=30, type=int, help='Nr epochs upon which model parameters '
                                                                       'lr should drop by 0.1')


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial learning rate decayed by 10 every few epochs.

    Args:
        optimizer (class derived under torch.optim): torch optimizer.
        epoch (int): current epoch count.
        args (argparse.Namespace):
    """
    lr = args.lr * (0.1 ** (epoch // args.lr_drop_epoch_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_train_and_val_loader(args):
    """"Constructs data loaders for train and validation on ImageNet.

    Args:
        args (argparse.Namespace):

    Returns:
        train_loader (torch.utils.data.DataLoader): data loader for train data.
        val_loader (torch.utils.data.DataLoader): data loader for validation data.
    """
    traindir = os.path.join(args.data, 'training')
    valdir = os.path.join(args.data, 'validation')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Instead of ImageFolder dataset class, we use ImageFolderIdx which is derived
    # from the former and returns the index of items in the minibatch.
    train_dataset = ImageFolderWithIdx(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        ImageFolderWithIdx(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)
    return train_loader, val_loader


def get_model_and_loss_criterion(args):
    """Initializes DNN model and loss function on a single GPU or multiple GPU's for data parallelism.

    Args:
        args (argparse.Namespace):

    Returns:
        model (torch.nn.Module): DNN model.
        criterion (torch.nn.modules.loss): cross entropy loss
    """
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    if args.gpu is not None:
        print("Using GPU: {} for training".format(args.gpu))
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    else:
        print('Splitting model across all GPUs with data parallelism')
        criterion = nn.CrossEntropyLoss().cuda()
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    return model, criterion


def validate(args, val_loader, model, criterion, epoch):
    """Evaluates model on validation set and logs score on tensorboard.

    Args:
        args (argparse.Namespace):
        val_loader (torch.utils.data.dataloader): dataloader for validation set.
        model (torch.nn.Module): DNN model.
        criterion (torch.nn.modules.loss): cross entropy loss
        epoch (int): current epoch
    """
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, target, _) in enumerate(val_loader):
            if args.gpu is not None:
                inputs = inputs.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            else:
                inputs = inputs.cuda()
                target = target.cuda()

            # compute output
            logits = model(inputs)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            acc1, acc5 = utils.compute_topk_accuracy(logits, target, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

        print(' * Validation Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    # Logging results on tensorboard
    log_value('val/accuracy_top1', top1.avg, step=epoch)
    log_value('val/accuracy_top5', top5.avg, step=epoch)
    log_value('val/loss', losses.avg, step=epoch)


def train_for_one_epoch(args,
                        train_loader,
                        model,
                        criterion,
                        optimizer,
                        epoch,
                        global_iter,
                        optimizer_data_parameters,
                        data_parameters,
                        config):
    """Train model for one epoch on the train set.

    Args:
        args (argparse.Namespace):
        train_loader (torch.utils.data.dataloader): dataloader for train set.
        model (torch.nn.Module): DNN model.
        criterion (torch.nn.modules.loss): cross entropy loss.
        optimizer (torch.optim.SGD): optimizer for model parameters.
        epoch (int): current epoch.
        global_iter (int): current iteration count.
        optimizer_data_parameters (tuple SparseSGD): SparseSGD optimizer for class and instance data parameters.
        data_parameters (tuple of torch.Tensor): class and instance level data parameters.
        config (dict): config file for the experiment.

    Returns:
        global iter (int): updated iteration count after 1 epoch.
    """

    # Initialize counters
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    # Unpack data parameters
    optimizer_class_param, optimizer_inst_param = optimizer_data_parameters
    class_parameters, inst_parameters = data_parameters

    # Switch to train mode
    model.train()
    start_epoch_time = time.time()

    for i, (inputs, target, index_dataset) in enumerate(train_loader):
        global_iter = global_iter + 1

        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
        else:
            inputs = inputs.cuda()
            target = target.cuda()

        # Flush the gradient buffer for model and data-parameters
        optimizer.zero_grad()
        if args.learn_class_parameters:
            optimizer_class_param.zero_grad()
        if args.learn_inst_parameters:
            optimizer_inst_param.zero_grad()

        logits = model(inputs)

        if args.learn_class_parameters or args.learn_inst_parameters:
            # Compute data parameters for instances in the minibatch
            class_parameter_minibatch = class_parameters[target]
            inst_parameter_minibatch = inst_parameters[index_dataset]
            data_parameter_minibatch = utils.get_data_param_for_minibatch(
                                            args,
                                            class_param_minibatch=class_parameter_minibatch,
                                            inst_param_minibatch=inst_parameter_minibatch)

            # Compute logits scaled by data parameters
            logits = logits / data_parameter_minibatch

        loss = criterion(logits, target)

        # Apply weight decay on data parameters
        if args.learn_class_parameters or args.learn_inst_parameters:
            loss = utils.apply_weight_decay_data_parameters(args, loss,
                                                            class_parameter_minibatch=class_parameter_minibatch,
                                                            inst_parameter_minibatch=inst_parameter_minibatch)

        # Compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        if args.learn_class_parameters:
            optimizer_class_param.step()
        if args.learn_inst_parameters:
            optimizer_inst_param.step()

        # Clamp class and instance level parameters within certain bounds
        if args.learn_class_parameters or args.learn_inst_parameters:
            utils.clamp_data_parameters(args, class_parameters, config, inst_parameters)

        # Measure accuracy and record loss
        acc1, acc5 = utils.compute_topk_accuracy(logits, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

        # Log stats for data parameters and loss every few iterations
        if i % args.print_freq == 0:
            utils.log_intermediate_iteration_stats(args, class_parameters, epoch,
                                                   global_iter, inst_parameters,
                                                   losses, top1, top5)
    # Print and log stats for the epoch
    print('Train-Epoch-{}: Acc-5:{}, Acc-1:{}, Loss:{}'.format(epoch, top5.avg, top1.avg, losses.avg))
    print('Time for 1 epoch: {}'.format(time.time() - start_epoch_time))
    log_value('train/accuracy_top5', top5.avg, step=epoch)
    log_value('train/accuracy_top1', top1.avg, step=epoch)
    log_value('train/loss', losses.avg, step=epoch)

    return global_iter


def main_worker(args, config):
    """Trains model on ImageNet using data parameters

    Args:
        args (argparse.Namespace):
        config (dict): config file for the experiment.
    """
    global_iter = 0

    # Create model
    model, loss_criterion = get_model_and_loss_criterion(args)

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Get train and validation dataset loader
    train_loader, val_loader = get_train_and_val_loader(args)

    # Initialize class and instance based temperature
    (class_parameters, inst_parameters,
     optimizer_class_param, optimizer_inst_param) = utils.get_class_inst_data_params_n_optimizer(
                                                                        args=args,
                                                                        nr_classes=1000,
                                                                        nr_instances=len(train_loader.dataset),
                                                                        device='cuda'
                                                                        )
    # Training loop
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # Train for one epoch
        global_iter = train_for_one_epoch(
                            args=args,
                            train_loader=train_loader,
                            model=model,
                            criterion=loss_criterion,
                            optimizer=optimizer,
                            epoch=epoch,
                            global_iter=global_iter,
                            optimizer_data_parameters=(optimizer_class_param, optimizer_inst_param),
                            data_parameters=(class_parameters, inst_parameters),
                            config=config)

        # Evaluate on validation set
        validate(args, val_loader, model, loss_criterion, epoch)

        # Save artifacts
        utils.save_artifacts(args, epoch, model, class_parameters, inst_parameters)

        # Log temperature stats over epochs
        if args.learn_class_parameters:
            utils.log_stats(data=torch.exp(class_parameters),
                            name='epochs_stats_class_parameter',
                            step=epoch)
        if args.learn_inst_parameters:
            utils.log_stats(data=torch.exp(inst_parameters),
                            name='epochs_stats_inst_parameter',
                            step=epoch)


def main():
    args = parser.parse_args()
    args.log_dir = './logs_CL_imagenet'
    args.save_dir = './weights_CL_imagenet'
    utils.generate_log_dir(args)
    utils.generate_save_dir(args)

    config = {}
    config['clamp_inst_sigma'] = {}
    config['clamp_inst_sigma']['min'] = np.log(1/20)
    config['clamp_inst_sigma']['max'] = np.log(20)
    config['clamp_cls_sigma'] = {}
    config['clamp_cls_sigma']['min'] = np.log(1/20)
    config['clamp_cls_sigma']['max'] = np.log(20)
    utils.save_config(args.save_dir, config)

    # Set seed for reproducibility
    utils.set_seed(args)

    # Simply call main_worker function
    main_worker(args, config)


if __name__ == '__main__':
    main()
