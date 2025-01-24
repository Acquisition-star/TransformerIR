import random
import numpy as np
import torch.utils.data as data
from .load_images import get_image_paths, read_images, image_crop, img_resize_np, augment_img, single2tensor3


class DatasetSR(data.Dataset):
    def __init__(self, config, is_train=True):
        super(DatasetSR, self).__init__()
        self.config = config
        self.is_train = is_train
        self.n_channels = config.n_channels
        self.scale = config.up_scale
        self.patch_size = config.H_size if hasattr(config, 'H_size') else 96
        self.L_size = self.patch_size // self.scale

        # 获取 H/L 图片文件路径
        self.paths_H = get_image_paths(config.dataroot_H)
        self.paths_L = get_image_paths(config.dataroot_L)

        assert self.paths_H, 'Error: H path is empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L),
                                                                                           len(self.paths_H))

    def __getitem__(self, index):
        L_path = None

        # 获得 H image
        H_path = self.paths_H[index]
        img_H = np.float32(read_images(H_path, self.n_channels) / 255.0)

        # 修整图像
        img_H = image_crop(img_H, self.scale)

        # 获得 L image
        if self.paths_L:
            L_path = self.paths_L[index]
            img_L = np.float32(read_images(L_path, self.n_channels) / 255.0)
        else:
            H, W = img_H.shape[:2]
            img_L = img_resize_np(img_H, 1 / self.scale, True)

        if self.is_train:
            H, W, C = img_L.shape

            # 随机裁剪
            rand_H = random.randint(0, max(0, H - self.L_size))
            rand_W = random.randint(0, max(0, W - self.L_size))
            img_L = img_L[rand_H:rand_H + self.L_size, rand_W:rand_W + self.L_size, :]

            rnd_h_H, rnd_w_H = int(rand_H * self.scale), int(rand_W * self.scale)
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            # augmentation - flip and/or rotate
            mode = random.randint(0, 7)
            img_L, img_H = augment_img(img_L, mode=mode), augment_img(img_H, mode=mode)

        img_H, img_L = single2tensor3(img_H), single2tensor3(img_L)

        if L_path is None:
            L_path = H_path

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
