import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from .load_images import read_images, random_crop_img, random_crop_2img, augment_img


class Dataset_denoising(Dataset):
    def __init__(self, config):
        super(Dataset_denoising, self).__init__()
        self.state = config.state
        self.n_channels = config.n_channels
        self.patch_size = None if self.state == 'Val' else config.image_size
        self.sigma = config.sigma
        self.H_path = config.H_path
        self.L_path = config.L_path

        self.images = []

        for h_dir, l_dir in zip(config.H_path, config.L_path):
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

        img_H = read_images(H_path)  # HWN-RGB
        img_L = None

        if self.state == 'Train':
            H, W, _ = img_H.shape
            if L_path != 'None':
                img_L = read_images(L_path)
                img_H, img_L = random_crop_2img(img_H, img_L, self.patch_size)
                img_H = torch.from_numpy(np.ascontiguousarray(img_H)).permute(2, 0, 1).float().div(255.0)
                img_L = torch.from_numpy(np.ascontiguousarray(img_L)).permute(2, 0, 1).float().div(255.0)
            else:
                img_H = random_crop_img(img_H, self.patch_size)
                img_H = torch.from_numpy(np.ascontiguousarray(img_H)).permute(2, 0, 1).float().div(255.0)
                img_L = img_H.clone()

                # 添加噪声
                noise = torch.randn(img_L.size()).mul_(self.sigma / 255.0)
                img_L.add_(noise)
        elif self.state == 'Val':
            img_H = np.float32(img_H / 255.0)
            if L_path != 'None':
                img_L = read_images(L_path)
                img_L = np.float32(img_L / 255.0)
            else:
                img_L = np.copy(img_H)
                # 添加噪声
                np.random.seed(seed=0)
                img_L += np.random.normal(0, self.sigma / 255.0, img_L.shape)
            # HWC to CHW
            img_L = torch.from_numpy(np.ascontiguousarray(img_L)).permute(2, 0, 1).float()
            img_H = torch.from_numpy(np.ascontiguousarray(img_H)).permute(2, 0, 1).float()

        return {'H': img_H, 'L': img_L, 'H_path': H_path, 'L_path': L_path}

    def __len__(self):
        return len(self.images)
