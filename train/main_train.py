import os
import numpy as np
import torch

from data.build import build_loader
from model.build import build_model


def main(config, logger):
    # 数据准备
    dataset_train, dataset_val, data_loader_train, data_loader_val = build_loader(config)

    # 模型定义
    logger.info(f"Building model: {config.net.type} --> {config.task}")
    model = build_model(config.net)
    # logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of params: {n_parameters}")

    model.cuda()
