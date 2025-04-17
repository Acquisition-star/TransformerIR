import argparse
import numpy as np
import os
from skimage.util import img_as_ubyte
import h5py
import scipy.io as sio
from tqdm import tqdm

parser = argparse.ArgumentParser('MAT transfer to image!', add_help=False)
parser.add_argument('--src', type=str, default=r'D:\Data\Denoising\DND')
parser.add_argument('--out', type=str, default=r'D:\Data\Denoising\dnd_image')

args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

# Load info
infos = h5py.File(os.path.join(args.src, 'info.mat'), 'r')
info = infos['info']
bb = info['boundingboxes']

for i in tqdm(range(50)):


print('Finish Transfer')

