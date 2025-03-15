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

# 工具函数
from utils.util import calculate_psnr, calculate_ssim, bgr2ycbcr
from utils.logger import create_logger
from data.load_images import read_images, random_crop_img, random_crop_2img, augment_img
from define_models import define_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser('TransformerIR evaluation script', add_help=False)
parser.add_argument('--task_type', type=str, default='denoising', help='task type')
parser.add_argument('--output', type=str, default='results/', help='path to output folder')
parser.add_argument('--env', type=str, default='experiment_1', help='experiment name')
parser.add_argument('--model_name', type=str, default='Uformer-B', help='model name')
parser.add_argument("--pth", type=str,
                    default=r'F:\GraduationThesis\Project\TransformerIR\analysis\model_zoo\Uformer\Uformer_B.pth',
                    help="path to pretrained model")
parser.add_argument("--is_cpk", action='store_true', default=False, help="whether to use checkpoint")
parser.add_argument("--crop", action='store_true', default=False, help="whether to crop images")
parser.add_argument("--img_size", type=int, default=256, help="image size")

args = parser.parse_known_args()[0]

root_path = f'{args.output}{args.task_type}/{args.env}'

os.makedirs(root_path, exist_ok=True)

model_info = {
    'name': args.model_name,
    'pth': args.pth,
    'is_cpk': args.is_cpk,
    'img_size': (args.img_size, args.img_size) if args.crop else None,
}

logger = create_logger(root_path, name=f"{model_info['name']}_{args.env}")

data_list = [
    {
        'name': 'SIDD',
        'H_path': r'E:\Data\SIDD\val\groundtruth',
        'L_path': r'E:\Data\SIDD\val\input',
    },
    {
        'name': 'CBSD68',
        'H_path': r'E:\Data\Test\CBSD68\HI',
    },
    {
        'name': 'Kodak24',
        'H_path': r'E:\Data\Test\Kodak24\HI',
    },
    {
        'name': 'McMaster',
        'H_path': r'E:\Data\Test\McMaster\HI',
    },
]


class Dataset_denoising_val(Dataset):
    def __init__(self, input_channels, H_path, L_path=None, sigma=None, img_size=None):
        super(Dataset_denoising_val, self).__init__()
        self.n_channels = input_channels
        self.sigma = sigma
        self.H_path = H_path
        self.L_path = L_path
        self.img_size = img_size

        self.images = []

        h_ps = sorted(glob.glob(os.path.join(self.H_path, '*')))
        if self.L_path is not None:
            l_ps = sorted(glob.glob(os.path.join(self.L_path, '*')))
            for h_p, l_p in zip(h_ps, l_ps):
                self.images.append({'H': h_p, 'L': l_p})
        else:
            for h_p in h_ps:
                self.images.append({'H': h_p, 'L': 'None'})

    def __getitem__(self, index):
        H_path, L_path = self.images[index]['H'], self.images[index]['L']
        name = os.path.basename(H_path)

        img_H = read_images(H_path)  # HWC-RGB
        img_H = np.float32(img_H / 255.0)

        img_L = None

        if L_path != 'None':
            img_L = read_images(L_path)  # HWC-RGB
            img_L = np.float32(img_L / 255.0)
        else:
            img_L = np.copy(img_H)
            # 添加噪声
            np.random.seed(seed=0)
            img_L += np.random.normal(0, self.sigma / 255.0, img_L.shape)
        # HWC to CHW
        img_L = torch.from_numpy(np.ascontiguousarray(img_L)).permute(2, 0, 1).float()
        img_H = torch.from_numpy(np.ascontiguousarray(img_H)).permute(2, 0, 1).float()
        if self.img_size is not None:
            img_L = img_L[:, :self.img_size[0], :self.img_size[1]]
            img_H = img_H[:, :self.img_size[0], :self.img_size[1]]
        return {'H': img_H, 'L': img_L, 'img_name': name}

    def __len__(self):
        return len(self.images)


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
        if data_info['name'] != 'SIDD':
            for n in [15, 25, 50]:
                l = {'name': f'{data_info["name"]}_n{n}', 'H_path': data_info["H_path"], 'L_path': None, 'sigma': n}
                data_lists.append(l)
        else:
            data_info['sigma'] = None
            data_lists.append(data_info)
    return data_lists


def main():
    test_results = {'模型': model_info['name'], '模型文件': model_info['pth']}

    # 数据信息处理
    data_lists = deal_list()

    model = define_model(model_info, logger)
    model.to(device)
    model.eval()

    # 模型速度、内存、计算复杂度
    # macs, params = get_model_complexity_info(model, (3, 128, 128), print_per_layer_stat=True)
    #
    # logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # test_results['计算复杂度'] = macs
    # test_results['参数量'] = params

    for data_info in data_lists:
        data_set = Dataset_denoising_val(input_channels=3, H_path=data_info['H_path'],
                                         L_path=data_info['L_path'], sigma=data_info['sigma'],
                                         img_size=model_info['img_size'])

        img_save_path = os.path.join(root_path, data_info['name'])
        os.makedirs(img_save_path, exist_ok=True)

        logger.info(f"{model_info['name']} start to test on {data_info['name']}!")
        avg_psnr, avg_ssim, avg_psnr_y, avg_ssim_y = 0.0, 0.0, 0.0, 0.0
        start_time = time.time()
        for index in range(0, len(data_set)):
            L_img, H_img = data_set[index]['L'], data_set[index]['H']  # CHW-RGB
            image_name = data_set[index]['img_name']

            L_img = L_img.unsqueeze(0).to(device)  # CHW-RGB --> NCHW-RGB

            with torch.no_grad():
                output = model(L_img)

            visual_H = trans_img(H_img)
            visual_O = trans_img(output.squeeze(0).cpu())

            cv2.imwrite(f'{img_save_path}/{image_name}', visual_O)

            psnr = calculate_psnr(visual_O, visual_H)
            ssim = calculate_ssim(visual_O, visual_H)

            visual_H = bgr2ycbcr(visual_H.astype(np.float32) / 255.) * 255.
            visual_O = bgr2ycbcr(visual_O.astype(np.float32) / 255.) * 255.
            psnr_y = calculate_psnr(visual_O, visual_H)
            ssim_y = calculate_ssim(visual_O, visual_H)

            avg_psnr += psnr
            avg_ssim += ssim
            avg_psnr_y += psnr_y
            avg_ssim_y += ssim_y

            logger.info('Testing {:d} \t {:20s} \t - PSNR: {:.2f} dB; SSIM: {:.4f}; '
                        'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}; '.
                        format(index, image_name, psnr, ssim, psnr_y, ssim_y))
        end_time = time.time()
        total_time = end_time - start_time  # 计算总时间
        avg_time = total_time / len(data_set)
        avg_psnr /= len(data_set)
        avg_ssim /= len(data_set)
        avg_psnr_y /= len(data_set)
        avg_ssim_y /= len(data_set)

        test_results[data_info['name']] = round(avg_psnr, 2)
        test_results[f"{data_info['name']}_avg_time"] = round(avg_time, 2)

        logger.info(
            '{} -- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}  -- Average PSNR_Y/SSIM_Y: {:.2f} dB; {:.4f}  -- Average Time: {:.4f}'.format(
                data_info['name'], avg_psnr, avg_ssim, avg_psnr_y, avg_ssim_y, avg_time))

        logger.info(f"{data_info['name']} finish to test!!!\n\n")

    # 保存结果到cvs文件中
    df = pd.DataFrame([test_results])
    # 保存到 CSV 文件，不保存行索引，并且使用 utf-8-sig 编码以保证在 Excel 中显示中文
    df.to_csv(f'{root_path}/test_results.csv', index=False, encoding='utf-8-sig')


main()
