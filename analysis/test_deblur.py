import os
import torch
import glob
import cv2
import pandas as pd
import numpy as np
import argparse
import json
import time
from torch.utils.data import Dataset
from ptflops import get_model_complexity_info
from timm.utils import AverageMeter
import lpips

# 工具函数
from utils.util import calculate_psnr, calculate_ssim, bgr2ycbcr, calculate_lpips
from utils.logger import create_logger
from utils.config import get_config
from utils.checkpoint import load_checkpoint_model
from data.load_images import read_images
from model.build import build_model
from data.dataset_deblur import Dataset_deblur_val

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lpips_fn = lpips.LPIPS(net='alex', verbose=False)

parser = argparse.ArgumentParser('TransformerIR evaluation script', add_help=False)
parser.add_argument('--task', type=str, default='deblur', help='task type')
parser.add_argument('--output', type=str, default='results/', help='path to output folder')
parser.add_argument('--env', type=str, default='demo', help='experiment name')
parser.add_argument('--cfg', type=str, default=None, help='model name')
parser.add_argument("--pth", type=str, default=None, help="path to pretrained model")
parser.add_argument("--imgH", type=int, default=None, help="image size")
parser.add_argument("--imgW", type=int, default=None, help="image size")

args = parser.parse_known_args()[0]
config = get_config(args)

root_path = f'{args.output}{args.task}/{args.env}'

os.makedirs(root_path, exist_ok=True)

logger = create_logger(root_path, name=f"{config.net.type}_{args.env}")

data_list = [
    {
        'name': 'GoPro',
        'H_path': r'D:\Data\Deblur\GoPro\test\target',
        'L_path': r'D:\Data\Deblur\GoPro\test\input',
    },
    {
        'name': 'HIDE',
        'H_path': r'D:\Data\Deblur\HIDE\target',
        'L_path': r'D:\Data\Deblur\HIDE\input',
    },
    # {
    #     'name': 'RealBlur_J',
    #     'H_path': r'D:\Data\Deblur\RealBlur_J\target',
    #     'L_path': r'D:\Data\Deblur\RealBlur_J\input',
    # },
    # {
    #     'name': 'RealBlur_R',
    #     'H_path': r'D:\Data\Deblur\RealBlur_R\target',
    #     'L_path': r'D:\Data\Deblur\RealBlur_R\input',
    # },
]


def trans_img(img):
    """
    输入图片CHW-RGB转化为HWC-BGR图片类型
    :param img: range(0, 1) CHW-RGB tensor
    :return: range(0, 255) HWC-BGR nparray cpu
    """
    img = img.float().squeeze(0).clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HWC-BGR
    return (img * 255.0).round().astype(np.uint8)


def deal_list():
    data_lists = []
    for data_info in data_list:
        data_lists.append(data_info)
    return data_lists


def define_model(config, args):
    model = build_model(config.net)
    if os.path.exists(args.pth):
        logger.info(f"Loading model from {args.pth}")
    load_checkpoint_model(model, args.pth, logger)
    return model


def main():
    test_results = {'模型': config.net.type, '模型文件': args.pth, '配置文件': args.cfg}

    # 数据信息处理
    data_lists = deal_list()

    model = define_model(config, args)
    model.to(device)
    model.eval()

    img_size = None
    if args.imgH is not None and args.imgW is not None:
        img_size = (args.imgH, args.imgW)

    for data_info in data_lists:
        data_set = Dataset_deblur_val(input_channels=3, H_path=[data_info['H_path']], L_path=[data_info['L_path']],
                                      patch_size=img_size)

        img_save_path = os.path.join(root_path, data_info['name'])
        os.makedirs(img_save_path, exist_ok=True)

        if hasattr(config.net, 'attn_type'):
            logger.info(f"{config.net.attn_type} start to test on {data_info['name']}!")
        else:
            logger.info(f"{config.net.type} start to test on {data_info['name']}!")
        avg_psnr, avg_ssim, avg_lpips, avg_psnr_y, avg_ssim_y = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        start_time = time.time()
        for index in range(0, len(data_set)):
            data_loader = data_set[index]
            L_img, H_img, L_path, H_path = data_loader['L'], data_loader['H'], data_loader['L_path'], data_loader[
                'H_path']  # CHW-RGB
            image_name = os.path.basename(L_path)
            # image_name = data_set[index]['img_name']

            L_img = L_img.unsqueeze(0).to(device)  # CHW-RGB --> NCHW-RGB

            with torch.no_grad():
                output = model(L_img)

            visual_H = trans_img(H_img)
            visual_O = trans_img(output.squeeze(0).cpu())

            cv2.imwrite(f'{img_save_path}/{image_name}', visual_O)

            psnr = calculate_psnr(visual_O, visual_H)
            ssim = calculate_ssim(visual_O, visual_H)
            lpips = calculate_lpips(visual_O, visual_H, lpips_fn)

            # visual_H = bgr2ycbcr(visual_H.astype(np.float32) / 255.) * 255.
            # visual_O = bgr2ycbcr(visual_O.astype(np.float32) / 255.) * 255.
            # psnr_y = calculate_psnr(visual_O, visual_H)
            # ssim_y = calculate_ssim(visual_O, visual_H)

            avg_psnr.update(psnr)
            avg_ssim.update(ssim)
            avg_lpips.update(lpips)

            logger.info('Testing {:d} \t {:20s} \t - PSNR: {:.2f} dB; SSIM: {:.4f}; LPIPS: {:.4f};'.
                        format(index, image_name, psnr, ssim, lpips))
        end_time = time.time()
        total_time = end_time - start_time  # 计算总时间
        avg_time = total_time / len(data_set)

        test_results[f"{data_info['name']}_PSNR"] = round(avg_psnr.avg, 2)
        test_results[f"{data_info['name']}_SSIM"] = round(avg_ssim.avg, 4)
        test_results[f"{data_info['name']}_LPIPS"] = round(avg_lpips.avg, 4)
        test_results[f"{data_info['name']}_time"] = round(avg_time, 2)

        logger.info(
            '{} -- Average PSNR: {:.2f} dB  -- Average SSIM(RGB): {:.4f}  -- Average LPIPS: {:.4f}  -- Average Time: {:.4f} s'.format(
                data_info['name'], avg_psnr.avg, avg_ssim.avg, avg_lpips.avg, avg_time))

        logger.info(f"{data_info['name']} finish to test!!!\n\n")

    # 保存结果到cvs文件中
    df = pd.DataFrame([test_results])
    # 保存到 CSV 文件，不保存行索引，并且使用 utf-8-sig 编码以保证在 Excel 中显示中文
    df.to_csv(f'{root_path}/test_results.csv', index=False, encoding='utf-8-sig')


main()
