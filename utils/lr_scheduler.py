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
    elif config.type == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step,
            gamma=config.gamma
        )
    elif config.type == 'CosineAnnealingLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.T_max,
            eta_min=config.eta_min
        )
    else:
        raise NotImplementedError
    return lr_scheduler
