import os
import torch
import argparse
import pandas as pd
import numpy as np
from ptflops import get_model_complexity_info

from utils.logger import create_logger

# 模型引入
from model.nets.SwinIR import SwinIR
from model.nets.Uformer import Uformer
from model.nets.NAFNet import NAFNet
from model.nets.Stripformer import Stripformer
from model.TransformerIR import TransformerIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser('TransformerIR evaluation script', add_help=False)
parser.add_argument('--model_name', type=str, default='TransformerIR', help='model name')

args = parser.parse_known_args()[0]


def define_only_model(model_name):
    model = None
    if model_name == 'SwinIR':
        model = SwinIR(
            upscale=1,
            in_chans=3,
            img_size=128,
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='',
            resi_connection='1conv'
        )
    elif model_name == 'Uformer-T':
        model = Uformer(
            img_size=128,
            embed_dim=16,
            win_size=8,
            token_projection='linear',
            token_mlp='leff',
            depths=[2, 2, 2, 2, 2, 2, 2, 2, 2],
            modulator=True,
            dd_in=3
        )
    elif model_name == 'Uformer-S':
        model = Uformer(
            img_size=128,
            embed_dim=32,
            win_size=8,
            token_projection='linear',
            token_mlp='leff',
            depths=[2, 2, 2, 2, 2, 2, 2, 2, 2],
            modulator=True,
            dd_in=3
        )
    elif model_name == 'Uformer-B':
        model = Uformer(
            img_size=128,
            embed_dim=32,
            win_size=8,
            token_projection='linear',
            token_mlp='leff',
            depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],
            modulator=True,
            dd_in=3
        )
    elif model_name == 'NAFNet-32':
        model = NAFNet(
            img_channel=3,
            width=32,
            middle_blk_num=12,
            enc_blk_nums=[2, 2, 4, 8],
            dec_blk_nums=[2, 2, 2, 2],
        )
    elif model_name == 'NAFNet-64':
        model = NAFNet(
            img_channel=3,
            width=64,
            middle_blk_num=12,
            enc_blk_nums=[2, 2, 4, 8],
            dec_blk_nums=[2, 2, 2, 2],
        )
    elif model_name == 'Stripformer':
        model = Stripformer()
    elif model_name == 'TransformerIR':
        model = TransformerIR()
    else:
        raise Exception("Model error!")
    return model


test_results = {'模型': args.model_name}

model = define_only_model(model_name=args.model_name)
model.to(device)
model.eval()

# 模型速度、内存、计算复杂度
macs, params = get_model_complexity_info(model, (3, 128, 128), print_per_layer_stat=True)

print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
test_results['计算复杂度'] = macs
test_results['参数量'] = params
