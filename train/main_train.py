import os
from collections import OrderedDict

import numpy as np
import torch
import time

from data.build import build_loader
from model.build import build_model
from utils.optimizer import build_optimizer
from utils.lr_scheduler import build_scheduler
from utils.criterion import build_criterion
from utils.checkpoint import load_checkpoint, save_checkpoint, auto_resume_helper
from utils.util import NativeScalerWithGradNormCount, tensor2uint, img_save, calculate_psnr


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

    # 优化器设置
    optimizer = build_optimizer(config.optimizer, model, logger)

    loss_scaler = NativeScalerWithGradNormCount()

    # 学习率调度器构建
    lr_scheduler = build_scheduler(config, optimizer)

    # 损失函数构建
    criterion = build_criterion(config)

    max_accuracy = 0.0

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
        max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger)
        psnr = validate(config, model, data_loader_val)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {psnr:.1f}db")

    logger.info('Start training!')

    start_time = time.time()
    max_psnr = 0.0
    for epoch in range(config.train.start_epoch, config.train.num_epochs):
        # 开始训练
        for iter, train_data in enumerate(data_loader_train):
            # 学习率更新
            lr_scheduler.step(epoch)
            # 参数优化
            optimizer.zero_grad()
            output = model(train_data['L'].cuda())
            loss = config.train.criterion.weight * criterion(output, train_data['H'].cuda())
            loss.backward()
            optimizer.step()
            # 训练信息
            if epoch % config.train.checkpoint_print == 0:
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, loss:{:.3e}> '.format(epoch, iter, lr_scheduler.get_lr()[0], loss.item())
                logger.info(message)
        # 测试
        if epoch % config.train.checkpoint_val == 0:
            avg_psnr = 0.0
            for test_data in data_loader_val:
                image_name_ext = os.path.basename(test_data['L_path'][0])
                img_name, ext = os.path.splitext(image_name_ext)

                img_dir = os.path.join(config.path.root_path, "images", img_name)
                os.makedirs(img_dir, exist_ok=True)
                model.eval()
                output = model(test_data['L'].cuda())
                model.train()
                E_img = tensor2uint(output)
                H_img = tensor2uint(test_data['H'].cuda())
                save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, epoch))
                img_save(E_img, save_img_path)
                cur_psnr = calculate_psnr(E_img, H_img, border=config.scale)
                logger.info('{:>10s} | {:<4.2f}dB'.format(image_name_ext, cur_psnr))
                avg_psnr += cur_psnr
            avg_psnr /= len(data_loader_val)
            logger.info('<epoch:{:3d}, Average PSNR : {:<.2f}dB\n'.format(epoch, avg_psnr))
            max_psnr = max(max_psnr, avg_psnr)
            # 模型保存
            if epoch % config.train.checkpoint_save == 0:
                save_checkpoint(config, epoch, model, max_psnr, optimizer, lr_scheduler, loss_scaler, logger)


@torch.no_grad()
def validate(config, model, data_loader):
    avg_psnr = 0.0
    model.eval()

    for test_data in data_loader:
        output = model(test_data.L)
        visuals = OrderedDict()
        visuals['L'] = test_data.L.detach()[0].float().cpu()  # 低分辨率图像
        visuals['E'] = model(test_data.L).detach()[0].float().cpu()  # 生成图像
        visuals['H'] = test_data.H.detach()[0].float().cpu()  # 高分辨率图像
        E_img = tensor2uint(visuals['E'])
        H_img = tensor2uint(visuals['H'])
        cur_psnr = calculate_psnr(E_img, H_img, border=config.scale)
        avg_psnr += cur_psnr
    avg_psnr /= len(data_loader.dataset)
    return avg_psnr
