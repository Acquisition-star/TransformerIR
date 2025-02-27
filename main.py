import os
import json
import torch
import random
import argparse
import numpy as np
import torch.backends.cudnn as cudnn

from pathlib import Path

from utils.logger import create_logger
from utils.config import get_config
from train.main_train import main


def parse_option():
    parser = argparse.ArgumentParser('TransformerIR training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, default='configs/SR/X2/train_swinir_sr_classical_x2.yaml', help='path to config file')
    parser.add_argument('--output', type=str, default='Info/', help='path to output folder')
    parser.add_argument('--env', type=str, default='default', help='experiment name')
    parser.add_argument('--dist', action='store_true', help='use distributed training')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


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
