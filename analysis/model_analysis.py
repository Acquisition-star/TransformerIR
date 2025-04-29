import os

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from ptflops import get_model_complexity_info

from utils.config import get_config
from utils.checkpoint import load_checkpoint_model
from model.build import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser('TransformerIR evaluation script', add_help=False)
# parser.add_argument('--cfg', type=str, default=r'F:\GraduationThesis\Project\TransformerIR\configs\Denoising\MB-TaylorFormerV2-B.yaml', help='model name')
parser.add_argument("--pth", type=str, default=None, help="path to pretrained model")

args = parser.parse_known_args()[0]
# config = get_config(args)
config = torch.load(args.pth, map_location='cpu', weights_only=False)['config']


def define_model(config, args):
    model = build_model(config.net)
    if args.pth:
        load_checkpoint_model(model, args.pth)
    return model


test_results = {'模型': config.net.type}

model = define_model(config, args)
model.to(device)
model.eval()

# 模型速度、内存、计算复杂度
macs, params = get_model_complexity_info(model, (3, 128, 128), print_per_layer_stat=True)

print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
test_results['计算复杂度'] = macs
test_results['参数量'] = params
