import argparse

from utils.logger import create_logger
from utils.config import get_config


def parse_option():
    parser = argparse.ArgumentParser('TransformerIR training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, default=None, help='path to config file')
    parser.add_argument('--output', type=str, default='Info/', help='path to output folder')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


if __name__ == "__main__":
    logger = create_logger()
    logger.info("hello")
    args, config = parse_option()
