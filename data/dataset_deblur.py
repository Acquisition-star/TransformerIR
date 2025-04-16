import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from .load_images import read_images, random_crop_2img, crop_2img


class Dataset_deblur_train(Dataset):
    def __init__(self, patch_size, input_channels, H_path, L_path):
        """
        数据加载
        :param patch_size: 图像块大小
        :param input_channels: 输入维度
        """
        super(Dataset_deblur_train, self).__init__()
        self.n_channels = input_channels
        self.patch_size = patch_size
        self.H_path = H_path
        self.L_path = L_path

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

        img_H = read_images(H_path)  # HWN-RGB
        img_L = read_images(L_path)  # HWN-RGB
        img_H, img_L = random_crop_2img(img_H, img_L, self.patch_size)
        img_H = torch.from_numpy(np.ascontiguousarray(img_H)).permute(2, 0, 1).float().div(255.0)
        img_L = torch.from_numpy(np.ascontiguousarray(img_L)).permute(2, 0, 1).float().div(255.0)
        return {'H': img_H, 'L': img_L, 'H_path': H_path, 'L_path': L_path}

    def __len__(self):
        return len(self.images)


class Dataset_deblur_val(Dataset):
    def __init__(self, input_channels, H_path, L_path, patch_size=None):
        super(Dataset_deblur_val, self).__init__()
        self.n_channels = input_channels
        self.H_path = H_path
        self.L_path = L_path
        self.patch_size = patch_size

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

        img_H = read_images(H_path)  # WHC-RGB
        img_H = np.float32(img_H / 255.0)
        img_L = read_images(L_path)  # WHC-RGB
        img_L = np.float32(img_L / 255.0)

        if self.patch_size is not None:
            img_H, img_L = crop_2img(img_H, img_L, self.patch_size)

        # HWC to CHW
        img_L = torch.from_numpy(np.ascontiguousarray(img_L)).permute(2, 0, 1).float()
        img_H = torch.from_numpy(np.ascontiguousarray(img_H)).permute(2, 0, 1).float()
        return {'H': img_H, 'L': img_L, 'H_path': H_path, 'L_path': L_path}

    def __len__(self):
        return len(self.images)
