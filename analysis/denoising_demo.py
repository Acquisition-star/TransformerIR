import os
import torch
import glob
import cv2
import numpy as np
from collections import OrderedDict
from torch.utils.data import Dataset

from model.nets.SwinIR import SwinIR
from utils.util import calculate_psnr, calculate_ssim, bgr2ycbcr
from utils.logger import create_logger
from utils.checkpoint import load_checkpoint_model
from data.load_images import read_images

# 参数设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
task_type = 'color_dn'
root_path = f'results/{task_type}'
os.makedirs(root_path, exist_ok=True)

sigma = 15  # 噪声等级
boarder = 0
input_channel = 3  # 图像通道数

data_paths = [
    {
        'name': 'CBSD68',
        'H_path': r'E:\Data\Test\CBSD68\HI',
        'L_path': 'None',
    },
    {
        'name': 'Kodak24',
        'H_path': r'E:\Data\Test\Kodak24\HI',
        'L_path': 'None',
    },
    {
        'name': 'McMaster',
        'H_path': r'E:\Data\Test\McMaster\HI',
        'L_path': 'None',
    },
]

model_list = [
    {
        'index': 1,
        'name': 'SwinIR',
        'path': r'F:\GraduationThesis\Project\Results\Experiment1\ckpt_epoch_294.pth',
        'is_cpk': True,
    },
    {
        'index': 2,
        'name': 'SwinIR',
        'path': r'F:\GraduationThesis\Project\Results\Experiment_2\2120_G.pth',
        'is_cpk': False,
    },
]


def main():
    test_results = []
    for iter, model_info in enumerate(model_list):
        current_results = {'index': model_info['index'], 'name': model_info['name']}
        os.makedirs(os.path.join(root_path, f"example_{model_info['index']}"), exist_ok=True)
        logger = create_logger(os.path.join(root_path, f"example_{model_info['index']}"), name='msg.log')
        # 模型定义
        model = define_model(model_info['name'], model_info['path'], logger, is_cpk=model_info['is_cpk'])
        model.eval()
        model.to(device)

        # 开始测试
        for idx, data_path in enumerate(data_paths):
            # 读入图片
            data_name = data_path['name']
            H_path = data_path['H_path']
            L_path = data_path['L_path']
            data_set = Dataset_denoising(H_path, L_path, sigma=sigma, n_channels=input_channel)
            current_results[data_name] = {}
            psnr, ssim, psnr_y, ssim_y = 0.0, 0.0, 0.0, 0.0
            for i in range(0, len(data_set)):
                img_H, img_L = data_set[i]['img_H'], data_set[i]['img_L']
                img_H = img_H.unsqueeze(0)
                img_L = img_L.unsqueeze(0).to(device)
                img_output = model(img_L)





def define_model(model_name, pth, logger, is_cpk=True):
    model = None
    if model_name == 'SwinIR':
        if os.path.exists(pth):
            logger.info(f'Loading model from {pth}')
        if is_cpk:
            param_key, settings = load_model_from_checkpoint(pth, logger)
            model = SwinIR(
                upscale=settings['upscale'],
                in_chans=settings['in_chans'],
                img_size=settings['img_size'],
                window_size=settings['window_size'],
                img_range=settings['img_range'],
                depths=settings['depths'],
                embed_dim=settings['embed_dim'],
                num_heads=settings['num_heads'],
                mlp_ratio=settings['mlp_ratio'],
                upsampler=settings['upsampler'],
                resi_connection=settings['resi_connection'],
            )
            msg = model.load_state_dict(param_key, strict=False)
            logger.info(msg)
        else:
            model = SwinIR(upscale=1, in_chans=3, img_size=48, window_size=8,
                           img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                           mlp_ratio=2, upsampler='', resi_connection='1conv')
            param_key_g = 'params'
            pretrained_model = torch.load(pth)
            model.load_state_dict(
                pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
                strict=True)
    elif model_name == 'Uformer':
        print('hello Uformer!')
    return model


def load_model_from_checkpoint(pth, logger):
    logger.info(f"==============> Resuming form {pth}....................")
    if pth.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            pth, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(pth, map_location='cpu')
    model_params, settings = checkpoint['model'], checkpoint['config']['net']
    del checkpoint
    return model_params, settings


class Dataset_denoising(Dataset):
    def __init__(self, H_path, L_path, sigma, n_channels=3):
        super(Dataset_denoising, self).__init__()
        self.H_path = H_path
        self.L_path = L_path
        self.sigma = sigma
        self.n_channels = n_channels

        self.images = []

        for h_dir, l_dir in zip(self.H_path, self.L_path):
            h_ps = sorted(glob.glob(os.path.join(h_dir, '*')))
            if l_dir != 'None':
                l_ps = sorted(glob.glob(os.path.join(l_dir, '*')))
                for h_p, l_p in zip(h_ps, l_ps):
                    self.images.append({'H': h_p, 'L': l_p})
            else:
                for h_p in h_ps:
                    self.images.append({'H': h_p, 'L': 'None'})

    def __getitem__(self, index):
        H_path, L_path = self.images[index]['H'], self.images[index]['L']

        img_H = read_images(H_path)  # HWC-RGB
        img_L = None

        img_H = np.float32(img_H / 255.0)
        if L_path != 'None':
            img_L = read_images(L_path)
            img_L = np.float32(img_L / 255.0)
        else:
            img_L = np.copy(img_H)
            # 添加噪声
            np.random.seed(seed=10)
            img_L += np.random.normal(0, self.sigma / 255.0, img_L.shape)
        # HWC to CHW
        img_L = torch.from_numpy(np.ascontiguousarray(img_L)).permute(2, 0, 1).float()
        img_H = torch.from_numpy(np.ascontiguousarray(img_H)).permute(2, 0, 1).float()

        return {'H': img_H, 'L': img_L, 'H_path': H_path, 'L_path': L_path}

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    main()
