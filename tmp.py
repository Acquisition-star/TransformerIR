import numpy as np
import torch
import torch.nn as nn
from data.load_images import read_images, random_crop_2img, crop_2img

patch_size = [1280, 704]
H_path = r'D:\Data\Deblur\GoPro\test\input\GOPR0384_11_00-000001.png'
L_path = r'D:\Data\Deblur\GoPro\test\input\GOPR0384_11_00-000001.png'

img_H = read_images(H_path)  # WHC-RGB
img_H = np.float32(img_H / 255.0)
img_L = read_images(L_path)  # WHC-RGB
img_L = np.float32(img_L / 255.0)

print(img_H.shape)

img_H, img_L = crop_2img(img_H, img_L, patch_size)

print(img_H.shape)
