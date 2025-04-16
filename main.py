import os
import torch
from collections import OrderedDict
import random
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

from pathlib import Path

from timm.utils import AverageMeter

from utils.logger import create_logger
from utils.config import get_config

from data.build import build_loader
from model.build import build_model
from utils.optimizer import build_optimizer
from utils.criterion import build_criterion
from utils.lr_scheduler import build_scheduler
from utils.checkpoint import load_checkpoint, save_checkpoint, auto_resume_helper, load_checkpoint_model
from utils.util import calculate_psnr, tensor2uint


def parse_option():
    parser = argparse.ArgumentParser('TransformerIR training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, default=None, help='path to config file')
    parser.add_argument("--dataloader_workers", type=int, default=1, help="number of dataloader workers")
    parser.add_argument("--batch_size", type=int, default=6, help='batch size')
    # parser.add_argument("--epochs", type=int, default=None, help='number of epochs')
    parser.add_argument('--output', type=str, default='Info/', help='path to output folder')
    parser.add_argument('--env', type=str, default='default', help='experiment name')
    parser.add_argument('--autodl', action='store_true', default=False, help='whether to use autodl machine to train')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config, logger):
    dataset_train, dataset_val, data_loader_train, data_loader_val = build_loader(config)

    # 模型定义
    logger.info(f"Building model: {config.net.type} --> {config.task}")
    model = build_model(config.net)
    logger.info(model)
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

    record = None

    if config.resume:
        if os.path.basename(config.resume) == 'ckpt_epoch_0.pth':
            load_checkpoint_model(model, config.resume, logger)
            logger.info(f"Pretrained model loaded from {config.resume}")
        else:
            max_accuracy, record = load_checkpoint(config, model, optimizer, lr_scheduler, criterion, logger)
            # psnr = validate(model, data_loader_val, logger)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {max_accuracy:.1f}db")

    logger.info(f"Start training...")

    if record is None:
        record = {'iter': [], 'psnr': [], 'lr': [], 'loss': []}

    iter = config.train.start_epoch
    while iter <= config.train.num_epochs:
        # 开始训练
        for _, train_data in enumerate(data_loader_train):
            # 参数优化
            optimizer.zero_grad()
            L_img, H_img = train_data['L'].cuda(), train_data['H'].cuda()
            outputs = model(L_img)

            if config.criterion.type in ['stripformer_loss']:
                loss = criterion(outputs, H_img, L_img)
            else:
                loss = config.train.lossfn_weight * criterion(outputs, H_img)
            loss.backward()
            optimizer.step()
            if iter % config.train.checkpoint_print == 0:
                message = ('<iter:{:8,d}, lr:{:.3e}, loss:{:.3e}> '
                           .format(iter, lr_scheduler.get_last_lr()[0], loss.item()))
                logger.info(message)
            # 测试
            if iter % config.train.checkpoint_val == 0:
                avg_psnr = validate(model, data_loader_val, logger)
                # 数据记录
                record['iter'].append(int(iter))
                record['psnr'].append(avg_psnr)
                record['lr'].append(round(lr_scheduler.get_last_lr()[0], 10))
                record['loss'].append(round(loss.item(), 6))
                logger.info('\n')
                logger.info('<iter:{:3d}, Average PSNR : {:<.2f}dB \n'.format(iter, avg_psnr))
                # 模型保存
                save_checkpoint(config, iter, model, avg_psnr, optimizer, lr_scheduler, criterion, logger, record)
                # 记录保存
                save_record(record, config.path.root_path)

            # 学习率更新
            lr_scheduler.step()
            iter += 1
            if iter > config.train.num_epochs:
                break

    save_record(record, config.path.root_path)

    logger.info('Finished Training')


@torch.no_grad()
def validate(model, data_loader, logger):
    model.eval()
    logger.info('Start validation...')

    avg_psnr = AverageMeter()

    # for iter, val_data in enumerate(data_loader):
    #     L_img, H_img = val_data['L'].cuda(), val_data['H'].cuda()
    #     image_name = os.path.basename(val_data['H_path'][0])
    #
    #     outputs = model(L_img)
    #     visuals = OrderedDict()
    #     visuals['H'] = val_data['H'].detach()[0].float().cpu()  # 原始图像
    #     visuals['G'] = outputs.detach()[0].float().cpu()  # 生成图像
    #     H_img = tensor2uint(visuals['H'])
    #     G_img = tensor2uint(visuals['G'])
    #     current_psnr = calculate_psnr(G_img, H_img)
    #     # logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(iter, image_name, current_psnr))
    #     avg_psnr.update(current_psnr)

    with tqdm(total=len(data_loader), desc="Validation Progress") as pbar:
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
            avg_psnr.update(current_psnr)

            # 更新进度条
            pbar.update(1)
            # 在进度条后显示当前 PSNR 和图像名称
            pbar.set_postfix({"Image": image_name, "PSNR": f"{current_psnr:.2f}dB"})

    model.train()
    logger.info('Average PSNR: {:<4.2f}dB'.format(avg_psnr.avg))
    return avg_psnr.avg


def save_record(record, path):
    df = pd.DataFrame(record)
    df_plot(df, path)
    df.to_csv(f'{path}/training_records.csv', index=False, encoding='utf-8-sig')


def df_plot(df, path):
    # 创建画布和垂直排列的双子图（更适应长序列）
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 8), sharex=True)

    # 通用样式设置
    plot_config = {
        "linewidth": 1.5,
        "alpha": 0.8,
        "marker": "",  # 移除数据点标记
        "markersize": 0
    }

    # 上：学习率曲线（对数坐标）
    sns.lineplot(
        data=df, x='iter', y='lr',
        ax=ax1, color='royalblue',
        **plot_config
    )
    ax1.set_yscale('log')  # 对数坐标转换
    ax1.grid(True, which='both', linestyle=':', alpha=0.5)
    ax1.set_ylabel('Learning Rate', labelpad=10)

    # 中：损失曲线
    sns.lineplot(
        data=df, x='iter', y='loss',
        ax=ax2, color='crimson',
        **plot_config
    )
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_xlabel('Iter', labelpad=10)
    ax2.set_ylabel('Loss', labelpad=10)

    # 下：PSNR曲线
    sns.lineplot(
        data=df, x='iter', y='psnr',
        ax=ax3, color='teal',
        **plot_config
    )
    ax3.set_ylabel('PSNR (dB)', labelpad=10)  # 修复标签错误
    ax3.grid(True, linestyle='--', alpha=0.5)  # 调整网格样式
    ax3.set_yscale('linear')  # PSNR通常不需要对数坐标
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.1f}"))  # 格式化Y轴标签

    # X轴优化
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(MaxNLocator(10))  # 自动间隔
        ax.tick_params(axis='x', rotation=45)  # 旋转标签

    # 紧凑布局
    plt.tight_layout(h_pad=3.0)  # 控制子图垂直间距
    plt.subplots_adjust(top=0.92)  # 顶部留出标题空间
    plt.suptitle('Training Process Monitoring', y=0.97)

    plt.savefig(f'{path}/training_plot.png', dpi=300, bbox_inches='tight')  # 保存图像


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
    config.path.config_path = str(root_path / "config.yaml")
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
