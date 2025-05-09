import os
import numpy as np
import torch
import math
import cv2
import random

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_image_paths(dataroot):
    paths = None
    if isinstance(dataroot, str):
        paths = sorted(_get_paths_from_images(dataroot))
    elif isinstance(dataroot, list):
        paths = []
        for i in dataroot:
            paths += sorted(_get_paths_from_images(i))
    return paths


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a directory'.format(path)
    images = []
    for path, _, files in sorted(os.walk(path)):
        for filename in sorted(files):
            if is_image_file(filename):
                img_path = os.path.join(path, filename)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def read_images(path, n_channels=3):
    """
    根据输入图片地址读入图片
    :param path: 输入图片地址
    :param n_channels: 输入图片的通道数
    :return: 读取完成的图片WHN-RGB
    """
    img = None
    if n_channels == 1:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img


def random_crop_img(img, patch):
    """
    将图片随机裁剪为patch大小的图像块
    :param img: 图像
    :param patch: 图像块大小
    :return: 裁剪完成的图像ppC-RGB
    """
    H, W, _ = img.shape
    # 随机裁剪图像块
    random_start_h = random.randint(0, max(0, H - patch))
    random_start_w = random.randint(0, max(0, W - patch))
    patch_H = img[random_start_h:random_start_h + patch, random_start_w:random_start_w + patch, :]

    # 随机操作图像
    patch_H = augment_img(patch_H, mode=(random.randint(0, 7)))
    return patch_H


def random_crop_2img(img1, img2, patch):
    """
    将两张图片随机裁剪为patch大小的图像块
    :param img1: 图像1
    :param img2: 图像2
    :param patch: 图像块大小
    :return: 裁剪完成的图像ppC-RGB
    """
    assert img1.shape == img2.shape, 'img1 and img2 have different shape'
    H, W, _ = img1.shape
    # 随机裁剪图像块
    random_start_h = random.randint(0, max(0, H - patch))
    random_start_w = random.randint(0, max(0, W - patch))
    patch_1 = img1[random_start_h:random_start_h + patch, random_start_w:random_start_w + patch, :]
    patch_2 = img2[random_start_h:random_start_h + patch, random_start_w:random_start_w + patch, :]

    # 随机操作图像
    mode = random.randint(0, 7)
    patch_1 = augment_img(patch_1, mode=mode)
    patch_2 = augment_img(patch_2, mode=mode)
    return patch_1, patch_2


def crop_2img(img1, img2, patch):
    """
    将两张图片裁剪为patch大小的图像块
    :param img1: 图像1
    :param img2: 图像2
    :param patch: 图像块大小
    :return: 裁剪完成的图像HWC-RGB
    """
    assert img1.shape == img2.shape, 'img1 and img2 have different shape'
    H, W, _ = img1.shape
    # 随机裁剪图像
    random_start_h = random.randint(0, max(0, H - patch[1]))
    random_start_w = random.randint(0, max(0, W - patch[0]))
    patch_1 = img1[random_start_h:random_start_h + patch[1], random_start_w:random_start_w + patch[0], :]
    patch_2 = img2[random_start_h:random_start_h + patch[1], random_start_w:random_start_w + patch[0], :]
    return patch_1, patch_2


def image_crop(img_in, scale):
    """
    对输入图像进行裁剪，使其宽度和高度能够被给定的 scale 整除
    :param img_in: 输入图像
    :param scale: int
    :return: 裁剪完成的图像
    """
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img


def img_resize_np(img, scale, antialiasing=True):
    """
    图像的缩放操作
    :param img: 输入图像
    :param scale: 缩放比例
    :param antialiasing: 是否启用抗锯齿
    :return:
    """
    # Now the scale should be the same for H and W
    # input: img: Numpy, HWC or HW [0,1]
    # output: HWC or HW [0,1] w/o round
    img = torch.from_numpy(img)
    need_squeeze = True if img.dim() == 2 else False
    if need_squeeze:
        img.unsqueeze_(2)

    in_H, in_W, in_C = img.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        for j in range(out_C):
            out_1[i, :, j] = img_aug[idx:idx + kernel_width, :, j].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        for j in range(out_C):
            out_2[:, i, j] = out_1_aug[:, idx:idx + kernel_width, j].mv(weights_W[i])
    if need_squeeze:
        out_2.squeeze_()

    return out_2.numpy()


def augment_img(img, mode):
    if mode == 0:
        return img  # 原始图像
    elif mode == 1:
        return np.flipud(img)  # 上下翻转
    elif mode == 2:
        return np.rot90(img)  # 顺时针旋转90度
    elif mode == 3:
        return np.flipud(np.rot90(img))  # 顺时针旋转90度，上下翻转
    elif mode == 4:
        return np.rot90(img, k=3)  # 逆时针旋转90度
    elif mode == 5:
        return np.flipud(np.rot90(img, k=3))  # 逆时针旋转90度，上下翻转
    elif mode == 6:
        return np.rot90(img, k=2)  # 顺时针旋转180度
    elif mode == 7:
        return np.flipud(np.rot90(img, k=2))  # 顺时针旋转180度，上下翻转


def single2tensor3(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
