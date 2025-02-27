import os
import torch


def build_criterion(config):
    loss_type = config.train.criterion.loss_type
    criterion = None
    if loss_type == 'l1':
        criterion = torch.nn.L1Loss().cuda()
    else:
        raise NotImplementedError('Loss type [{:s}] is not found.'.format(loss_type))
    return criterion
