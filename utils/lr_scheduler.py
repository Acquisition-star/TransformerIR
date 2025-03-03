import os
import torch


def build_scheduler(config, optimizer):
    lr_scheduler = None
    if config.type == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            config.milestones,
            config.gamma
        )
    else:
        raise NotImplementedError
    return lr_scheduler
