import math
import os
import torch
import time
from collections import OrderedDict
import random
import argparse
import numpy as np
import torch.backends.cudnn as cudnn

from pathlib import Path

from utils.logger import create_logger
from utils.config import get_config

from data.build import build_loader
from model.build import build_model
from utils.optimizer import build_optimizer
from utils.criterion import build_criterion
from utils.lr_scheduler import build_scheduler
from utils.checkpoint import load_checkpoint, save_checkpoint, auto_resume_helper
from utils.util import calculate_psnr, tensor2uint


def parse_option():
    parser = argparse.ArgumentParser('TransformerIR training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, default='configs/Denoising/swinir_denoising_color.yaml',
                        help='path to config file')
    parser.add_argument('--output', type=str, default='Info/', help='path to output folder')
    parser.add_argument('--env', type=str, default='default', help='experiment name')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config, logger):
    dataset_train, dataset_val, data_loader_train, data_loader_val = build_loader(config)

    # 模型定义
    logger.info(f"Building model: {config.net.type} --> {config.task}")
    model = build_model(config.net)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of params: {n_parameters}")
    model.cuda()

    # 优化器设置
    optimizer = build_optimizer(config.optimizer, model, logger)

    # 损失函数设置
    criterion = build_criterion(config.criterion)

    # 学习率调度器构建
    lr_scheduler = build_scheduler(config.scheduler, optimizer)

    # 检查点恢复
    resume_file = auto_resume_helper(config.path.checkpoint_path)
    config.defrost()
    config.resume = resume_file
    config.freeze()
    if resume_file:
        logger.info(f"Resume from {resume_file}")
    else:
        logger.info(f"No checkpoint found in {config.path.checkpoint_path}. Start training from scratch!")

    if config.resume:
        max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler, criterion, logger)
        psnr = validate(model, data_loader_val, logger)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {psnr:.1f}db")

    logger.info(f"Start training...")
    start_time = time.time()
    max_accuracy = 0.0
    max_psnr = 0.0

    current_step = 0
    for epoch in range(config.train.start_epoch, config.train.num_epochs):
        # 开始训练
        for iter, train_data in enumerate(data_loader_train):
            # 参数优化
            optimizer.zero_grad()
            L_img, H_img = train_data['L'].cuda(), train_data['H'].cuda()
            outputs = model(L_img)
            loss = config.train.lossfn_weight * criterion(outputs, H_img)
            loss.backward()
            optimizer.step()
            if epoch % config.train.checkpoint_print == 0:
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, loss:{:.3e}> '.format(epoch, iter,
                                                                                       lr_scheduler.get_last_lr()[0],
                                                                                       loss.item())
                logger.info(message)
        # 学习率更新
        lr_scheduler.step()
        # 测试
        if epoch % config.train.checkpoint_val == 0:
            avg_psnr = validate(model, data_loader_val, logger)
            logger.info('<epoch:{:3d}, Average PSNR : {:<.2f}dB\n'.format(epoch, avg_psnr))
            max_psnr = max(max_psnr, avg_psnr)
            # 模型保存
            save_checkpoint(config, epoch, model, max_psnr, optimizer, lr_scheduler, criterion, logger)
    logger.info('Finished Training')


@torch.no_grad()
def validate(model, data_loader, logger):
    avg_psnr = 0.0
    model.eval()
    logger.info('Start validation...')

    for iter, val_data in enumerate(data_loader):
        L_img, H_img = val_data['L'].cuda(), val_data['H'].cuda()
        image_name = os.path.basename(val_data['H_path'][0])

        outputs = model(L_img)
        visuals = OrderedDict()
        visuals['H'] = val_data['H'].detach()[0].float().cpu()  # 原始图像
        visuals['G'] = outputs.detach()[0].float().cpu()  # 生成图像
        H_img = tensor2uint(visuals['H'])
        G_img = tensor2uint(visuals['G'])
        current_psnr = calculate_psnr(G_img, H_img)
        logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(iter, image_name, current_psnr))
        avg_psnr += current_psnr
    avg_psnr /= len(data_loader.dataset)
    return avg_psnr


if __name__ == "__main__":
    args, config = parse_option()

    # 输出文件保存地址
    root_path = Path(args.output) / args.env / config.task  # 根目录
    checkpoint_path = root_path / "checkpoints"  # checkpoints目录
    os.makedirs(root_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    config.defrost()
    config.path.root_path = str(root_path)
    config.path.checkpoint_path = str(checkpoint_path)
    config.path.config_path = str(root_path / "config.json")
    config.freeze()

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    cudnn.benchmark = True

    logger = create_logger(output_dir=config.path.root_path, name=config.task)

    # 保存配置信息
    with open(config.path.config_path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {config.path.config_path}")

    # 显示config信息
    # logger.info(config.dump())
    # logger.info(json.dumps(vars(args)))

    main(config, logger)
