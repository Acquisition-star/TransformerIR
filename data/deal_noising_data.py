import os
import numpy as np
import glob
import cv2


def main():
    root = r'E:\Data\Test\urban100'
    H_path = f'{root}\HI'
    noises = [15, 25, 50]

    for noise in noises:
        L_path = f'{root}\\Noise{noise}'

        os.makedirs(L_path, exist_ok=True)

        for idx, path in enumerate(sorted(glob.glob(os.path.join(H_path, '*')))):
            # 读入图片
            filename, img_lq, img_gt = get_image_pair(path, noise)  # HWC-BGR, float32
            img_lq = (img_lq * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(L_path, f'{filename}_noise{noise}.png'), img_lq)

    print('数据集构建成功！')


def get_image_pair(path, noise):
    (filename, suffix) = os.path.splitext(os.path.basename(path))
    img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    np.random.seed(seed=0)
    img_lq = img_gt + np.random.normal(0, noise / 255., img_gt.shape)
    return filename, img_lq, img_gt


if __name__ == '__main__':
    main()
