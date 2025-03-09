import os
import torch
import torch.nn as nn


def build_criterion(config):
    loss_type = config.type
    criterion = None
    if loss_type == 'l1':
        criterion = torch.nn.L1Loss().cuda()
    elif loss_type == 'charbonnier':
        criterion = CharbonnierLoss(config.eps).cuda()
    else:
        raise NotImplementedError('Loss type [{:s}] is not found.'.format(loss_type))
    return criterion


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-9):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff ** 2 + self.eps ** 2))
