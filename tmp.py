import torch
from utils.config import get_config
import argparse


def parse_option():
    parser = argparse.ArgumentParser('TransformerIR training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, default='Info/demo/MB-TaylorFormer V2/config.yaml',
                        help='path to config file')
    parser.add_argument("--dataloader_workers", type=int, default=1, help="number of dataloader workers")
    parser.add_argument("--batch_size", type=int, default=24, help='batch size')
    parser.add_argument("--epochs", type=int, default=10, help='number of epochs')
    parser.add_argument('--output', type=str, default='Info/', help='path to output folder')
    parser.add_argument('--env', type=str, default='default', help='experiment name')
    parser.add_argument('--autodl', action='store_true', default=False, help='whether to use autodl machine to train')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


if __name__ == "__main__":
    args, config = parse_option()
    print(args)
    print('hello')
