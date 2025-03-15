import os
import torch
from collections import OrderedDict

from utils.checkpoint import load_checkpoint_model

# 模型引入
from model.nets.SwinIR import SwinIR
from model.nets.Uformer import Uformer
from model.nets.NAFNet import NAFNet
from model.nets.Stripformer import Stripformer


def define_model(model_info, logger):
    model = None
    if 'SwinIR' in model_info['name']:
        if os.path.exists(model_info['pth']):
            logger.info(f"Loading model from {model_info['pth']}")
        model = SwinIR(
            upscale=1,
            in_chans=3,
            img_size=96,
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='',
            resi_connection='1conv'
        )
        if model_info['is_cpk']:
            load_checkpoint_model(model, model_info['pth'], logger)
        else:
            param_key_g = 'params'
            pretrained_model = torch.load(model_info['pth'])
            model.load_state_dict(
                pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
                strict=True)
    elif model_info['name'] == 'Uformer-T':
        if os.path.exists(model_info['pth']):
            logger.info(f"Loading model from {model_info['pth']}")
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
        if model_info['is_cpk']:
            load_checkpoint_model(model, model_info['pth'], logger)
        else:
            cpk = torch.load(model_info['pth'])
            try:
                model.load_state_dict(cpk["state_dict"])
            except:
                state_dict = cpk["state_dict"]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if 'module.' in k else k
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
    elif model_info['name'] == 'Uformer-S':
        if os.path.exists(model_info['pth']):
            logger.info(f"Loading model from {model_info['pth']}")
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
        if model_info['is_cpk']:
            load_checkpoint_model(model, model_info['pth'], logger)
        else:
            cpk = torch.load(model_info['pth'])
            try:
                model.load_state_dict(cpk["state_dict"])
            except:
                state_dict = cpk["state_dict"]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if 'module.' in k else k
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
    elif model_info['name'] == 'Uformer-B':
        if os.path.exists(model_info['pth']):
            logger.info(f"Loading model from {model_info['pth']}")
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
        if model_info['is_cpk']:
            load_checkpoint_model(model, model_info['pth'], logger)
        else:
            cpk = torch.load(model_info['pth'])
            try:
                model.load_state_dict(cpk["state_dict"])
            except:
                state_dict = cpk["state_dict"]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if 'module.' in k else k
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
    elif model_info['name'] == 'NAFNet_32':
        if os.path.exists(model_info['pth']):
            logger.info(f"Loading model from {model_info['pth']}")
        model = NAFNet(
            img_channel=3,
            width=32,
            middle_blk_num=12,
            enc_blk_nums=[2, 2, 4, 8],
            dec_blk_nums=[2, 2, 2, 2],
        )
        load_net = torch.load(
            model_info['pth'], map_location=lambda storage, loc: storage)
        load_net = load_net['params']
        model.load_state_dict(load_net, strict=True)
    elif model_info['name'] == 'NAFNet_64':
        if os.path.exists(model_info['pth']):
            logger.info(f"Loading model from {model_info['pth']}")
        model = NAFNet(
            img_channel=3,
            width=64,
            middle_blk_num=12,
            enc_blk_nums=[2, 2, 4, 8],
            dec_blk_nums=[2, 2, 2, 2],
        )
        load_net = torch.load(
            model_info['pth'], map_location=lambda storage, loc: storage)
        load_net = load_net['params']
        model.load_state_dict(load_net, strict=True)
    elif model_info['name'] == 'Stripformer':
        if os.path.exists(model_info['pth']):
            logger.info(f"Loading model from {model_info['pth']}")
        model = Stripformer()
        model.load_state_dict(torch.load(model_info['pth']))
    else:
        raise Exception("Model error!")
    return model
