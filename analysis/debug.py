import os
import torch
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from yacs.config import CfgNode

from model.nets.SwinIR import SwinIR
from utils.util import calculate_psnr, calculate_ssim, bgr2ycbcr, tensor2uint
from utils.logger import create_logger
from utils.checkpoint import load_checkpoint_model
from train.SwinIR_train import validate
from data.load_images import read_images
from data.dataset_denoising import Dataset_denoising

# 数据设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
noise = 15
border = 0
window_size = 8
env = 'default'
root_path = f'results/color_dn/{env}/noise_{noise}'
data_paths = [r'E:\Data\Test\CBSD68', r'E:\Data\Test\Kodak24', r'E:\Data\Test\McMaster']
os.makedirs(root_path, exist_ok=True)
logger = create_logger(root_path, name=f'color_dn_{noise}')

model_list = {
    'SwinIR': 'model_zoo/SwinIR/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth',
}


def main():
    # 模型定义
    test_results = {}

    model_name = 'SwinIR'
    path = r'F:\GraduationThesis\Project\TransformerIR\Info\default\swinir_denoising_color_15\checkpoints\ckpt_epoch_9.pth'

    model = define_model_ckp(model_name, path)
    model.eval()
    model.to(device)
    os.makedirs(os.path.join(root_path, model_name), exist_ok=True)

    H_path = r'E:\Data\Test\McMaster\HI\1.tif'
    L_path = r'E:\Data\Test\McMaster\Noise15\1_noise15.png'

    (data_name, suffix) = os.path.splitext(os.path.basename(path))
    logger.info(f'{data_name} start to test')

    np.random.seed(seed=0)

    config = CfgNode()
    config.defrost()
    config.state = 'Val'
    config.H_path = [r'E:\Data\Test\McMaster\HI']
    config.L_path = [r'E:\Data\Test\McMaster\Noise15']
    config.sigma = noise
    config.n_channels = 3
    config.freeze()
    dataset = Dataset_denoising(config)

    img_H, img_L = dataset[0]['H'].to(device), dataset[0]['L'].to(device)
    # img_gt = np.transpose(img_H[[2, 1, 0], :, :].detach().cpu().numpy(), (1, 2, 0))  # CHW-RGB to HWC-BGR
    # img_lq = img_L.unsqueeze(0)
    img_gt = cv2.imread(dataset[0]['H_path'], cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img_lq = cv2.imread(dataset[0]['L_path'], cv2.IMREAD_COLOR).astype(np.float32) / 255.

    img_lq = np.transpose(img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # CHW-RGB
    img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # NCHW-RGB

    with torch.no_grad():
        _, _, h_old, w_old = img_lq.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
        output = model(img_lq)
        output = output[..., :h_old, :w_old]

    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HWC-BGR
    output = (output * 255.0).round().astype(np.uint8)

    # 计算psnr
    img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
    img_gt = img_gt[:h_old, :w_old, ...]
    img_gt = np.squeeze(img_gt)

    psnr = calculate_psnr(output, img_gt, border=border)

    print(psnr)


def define_model(model_name, pth):
    model = None
    if model_name == 'SwinIR':
        if os.path.exists(pth):
            logger.info(f'Loading model from {pth}')
        model = SwinIR(upscale=1, in_chans=3, img_size=128, window_size=8,
                       img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                       mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'
        pretrained_model = torch.load(pth)
        model.load_state_dict(
            pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
            strict=True)
    return model


def define_model_ckp(model_name, pth):
    model = None
    if model_name == 'SwinIR':
        if os.path.exists(pth):
            logger.info(f'Loading model from {pth}')
        model = SwinIR(upscale=1, in_chans=3, img_size=24, window_size=8,
                       img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                       mlp_ratio=2, upsampler='', resi_connection='1conv')
        load_checkpoint_model(model, pth, logger)
    return model


def get_image_pair(path, noise):
    (filename, suffix) = os.path.splitext(os.path.basename(path))
    img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    np.random.seed(seed=0)
    img_lq = img_gt + np.random.normal(0, noise / 255., img_gt.shape)
    return filename, img_lq, img_gt


if __name__ == '__main__':
    main()
