#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
""" Utility functions for training DNNs with data parameters"""
import os
import json
import shutil
import random

import torch
import numpy as np
from tensorboard_logger import configure, log_value, log_histogram

from optimizer.sparse_sgd import SparseSGD


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def compute_topk_accuracy(prediction, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        prediction (torch.Tensor): N*C tensor, contains logits for N samples over C classes.
        target (torch.Tensor):  labels for each row in prediction.
        topk (tuple of int): different values of k for which top-k accuracy should be computed.

    Returns:
        result (tuple of float): accuracy at different top-k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = prediction.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        result = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            result.append(correct_k.mul_(100.0 / batch_size))
        return result


def save_artifacts(args, epoch, model, class_parameters, inst_parameters):
    """Saves model and data parameters.

    Args:
        args (argparse.Namespace):
        epoch (int): current epoch
        model (torch.nn.Module): DNN model.
        class_parameters (torch.Tensor): class level data parameters.
        inst_parameters (torch.Tensor): instance level data parameters.
    """
    artifacts = {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'class_parameters': class_parameters.cpu().detach().numpy(),
            'inst_parameters': inst_parameters.cpu().detach().numpy()
             }

    file_path = args.save_dir + '/epoch_{}.pth.tar'.format(epoch)
    torch.save(obj=artifacts, f=file_path)


def save_config(save_dir, cfg):
    """Save config file to disk at save_dir.

    Args:
        save_dir (str): path to directory.
        cfg (dict): config file.
    """
    save_path = save_dir + '/config.json'
    if os.path.isfile(save_path):
        raise Exception("Expected an empty folder but found an existing config file.., aborting")
    with open(save_path,  'w') as outfile:
        json.dump(cfg, outfile)


def generate_save_dir(args):
    """Generate directory to save artifacts and tensorboard log files."""

    print('\nModel artifacts (checkpoints and config) are going to be saved in: {}'.format(args.save_dir))
    if os.path.exists(args.save_dir):
        if args.restart:
            print('Deleting old model artifacts found in: {}'.format(args.save_dir))
            shutil.rmtree(args.save_dir)
            os.makedirs(args.save_dir)
        else:
            error='Old artifacts found; pass --restart flag to erase'.format(args.save_dir)
            raise Exception(error)
    else:
        os.makedirs(args.save_dir)


def generate_log_dir(args):
    """Generate directory to save artifacts and tensorboard log files."""

    print('\nLog is going to be saved in: {}'.format(args.log_dir))

    if os.path.exists(args.log_dir):
        if args.restart:
            print('Deleting old log found in: {}'.format(args.log_dir))
            shutil.rmtree(args.log_dir)
            configure(args.log_dir, flush_secs=10)
        else:
            error='Old log found; pass --restart flag to erase'.format(args.log_dir)
            raise Exception(error)
    else:
        configure(args.log_dir, flush_secs=10)


def set_seed(args):
    """Set seed to ensure deterministic runs.

    Note: Setting torch to be deterministic can lead to slow down in training.
    """
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_class_inst_data_params_n_optimizer(args,
                                           nr_classes,
                                           nr_instances,
                                           device):
    """Returns class and instance level data parameters and their corresponding optimizers.

    Args:
        args (argparse.Namespace):
        nr_classes (int):  number of classes in dataset.
        nr_instances (int): number of instances in dataset.
        device (str): device on which data parameters should be placed.

    Returns:
        class_parameters (torch.Tensor): class level data parameters.
        inst_parameters (torch.Tensor): instance level data parameters
        optimizer_class_param (SparseSGD): Sparse SGD optimizer for class parameters
        optimizer_inst_param (SparseSGD): Sparse SGD optimizer for instance parameters
    """
    # class-parameter
    class_parameters = torch.tensor(np.ones(nr_classes) * np.log(args.init_class_param),
                                    dtype=torch.float32,
                                    requires_grad=args.learn_class_parameters,
                                    device=device)
    optimizer_class_param = SparseSGD([class_parameters],
                                      lr=args.lr_class_param,
                                      momentum=0.9,
                                      skip_update_zero_grad=True)
    if args.learn_class_parameters:
        print('Initialized class_parameters with: {}'.format(args.init_class_param))
        print('optimizer_class_param:')
        print(optimizer_class_param)

    # instance-parameter
    inst_parameters = torch.tensor(np.ones(nr_instances) * np.log(args.init_inst_param),
                                   dtype=torch.float32,
                                   requires_grad=args.learn_inst_parameters,
                                   device=device)
    optimizer_inst_param = SparseSGD([inst_parameters],
                                     lr=args.lr_inst_param,
                                     momentum=0.9,
                                     skip_update_zero_grad=True)
    if args.learn_inst_parameters:
        print('Initialized inst_parameters with: {}'.format(args.init_inst_param))
        print('optimizer_inst_param:')
        print(optimizer_inst_param)

    return class_parameters, inst_parameters, optimizer_class_param, optimizer_inst_param


def get_data_param_for_minibatch(args,
                                 class_param_minibatch,
                                 inst_param_minibatch):
    """Returns the effective data parameter for instances in a minibatch as per the specified curriculum.

    Args:
        args (argparse.Namespace):
        class_param_minibatch (torch.Tensor): class level parameters for samples in minibatch.
        inst_param_minibatch (torch.Tensor): instance level parameters for samples in minibatch.

    Returns:
        effective_data_param_minibatch (torch.Tensor): data parameter for samples in the minibatch.
    """
    sigma_class_minibatch = torch.exp(class_param_minibatch).view(-1, 1)
    sigma_inst_minibatch = torch.exp(inst_param_minibatch).view(-1, 1)

    if args.learn_class_parameters and args.learn_inst_parameters:
        # Joint curriculum
        effective_data_param_minibatch = sigma_class_minibatch + sigma_inst_minibatch
    elif args.learn_class_parameters:
        # Class level curriculum
        effective_data_param_minibatch = sigma_class_minibatch
    elif args.learn_inst_parameters:
        # Instance level curriculum
        effective_data_param_minibatch = sigma_inst_minibatch
    else:
        # This corresponds to the baseline case without data parameters
        effective_data_param_minibatch = 1.0

    return effective_data_param_minibatch


def apply_weight_decay_data_parameters(args, loss, class_parameter_minibatch, inst_parameter_minibatch):
    """Applies weight decay on class and instance level data parameters.

    We apply weight decay on only those data parameters which participate in a mini-batch.
    To apply weight-decay on a subset of data parameters, we explicitly include l2 penalty on these data
    parameters in the computational graph. Note, l2 penalty is applied in log domain. This encourages
    data parameters to stay close to value 1, and prevents data parameters from obtaining very high or
    low values.

    Args:
        args (argparse.Namespace):
        loss (torch.Tensor): loss of DNN model during forward.
        class_parameter_minibatch (torch.Tensor): class level parameters for samples in minibatch.
        inst_parameter_minibatch (torch.Tensor): instance level parameters for samples in minibatch.

    Returns:
        loss (torch.Tensor): loss augmented with l2 penalty on data parameters.
    """

    # Loss due to weight decay on instance-parameters
    if args.learn_inst_parameters and args.wd_inst_param > 0.0:
        loss = loss + 0.5 * args.wd_inst_param * (inst_parameter_minibatch ** 2).sum()

    # Loss due to weight decay on class-parameters
    if args.learn_class_parameters and args.wd_class_param > 0.0:
        # (We apply weight-decay to only those classes which are present in the mini-batch)
        loss = loss + 0.5 * args.wd_class_param * (class_parameter_minibatch ** 2).sum()

    return loss


def clamp_data_parameters(args, class_parameters, config, inst_parameters):
    """Clamps class and instance level parameters within specified range.

    Args:
        args (argparse.Namespace):
        class_parameters (torch.Tensor): class level parameters.
        inst_parameters (torch.Tensor): instance level parameters.
        config (dict): config file for the experiment.
    """
    if args.skip_clamp_data_param is False:
        if args.learn_inst_parameters:
            # Project the sigma's to be within certain range
            inst_parameters.data = inst_parameters.data.clamp_(
                min=config['clamp_inst_sigma']['min'],
                max=config['clamp_inst_sigma']['max'])
        if args.learn_class_parameters:
            # Project the sigma's to be within certain range
            class_parameters.data = class_parameters.data.clamp_(
                min=config['clamp_cls_sigma']['min'],
                max=config['clamp_cls_sigma']['max'])


def log_stats(data, name, step):
    """Logs statistics on tensorboard for data tensor.

    Args:
        data (torch.Tensor): torch tensor.
        name (str): name under which stats for the tensor should be logged.
        step (int): step used for logging
    """
    log_value('{}/highest'.format(name), torch.max(data).item(), step=step)
    log_value('{}/lowest'.format(name), torch.min(data).item(),  step=step)
    log_value('{}/mean'.format(name), torch.mean(data).item(),   step=step)
    log_value('{}/std'.format(name), torch.std(data).item(),     step=step)
    log_histogram('{}'.format(name), data.data.cpu().numpy(),    step=step)


def log_intermediate_iteration_stats(args, class_parameters, epoch, global_iter,
                                     inst_parameters, losses, top1=None, top5=None):
    """Log stats for data parameters and loss on tensorboard."""
    if top5 is not None:
        log_value('train_iteration_stats/accuracy_top5', top5.avg, step=global_iter)
    if top1 is not None:
        log_value('train_iteration_stats/accuracy_top1', top1.avg, step=global_iter)
    log_value('train_iteration_stats/loss', losses.avg, step=global_iter)
    log_value('train_iteration_stats/epoch', epoch, step=global_iter)

    # Log temperature stats
    if args.learn_class_parameters:
        log_stats(data=torch.exp(class_parameters),
                  name='iter_stats_class_parameter',
                  step=global_iter)
    if args.learn_inst_parameters:
        log_stats(data=torch.exp(inst_parameters),
                  name='iter_stats_inst_parameter',
                  step=global_iter)


