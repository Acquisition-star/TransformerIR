import os
import torch


def build_scheduler(config, optimizer):
    lr_scheduler = None
    if config.train.scheduler.type == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            config.train.scheduler.milestones,
            config.train.scheduler.gamma
        )
    else:
        raise NotImplementedError
    return lr_scheduler
