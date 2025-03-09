# 对于论文代码中提供的模型进行测试数据采集

import os
import torch
import glob
import cv2
import numpy as np
from collections import OrderedDict

from model.nets.SwinIR import SwinIR
from model.nets.Uformer import Uformer

from utils.util import calculate_psnr, calculate_ssim, bgr2ycbcr
from utils.logger import create_logger

# 数据设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
noise = 15
border = 0
window_size = 8
task_type = 'color_dn'
env = 'example_2'
root_path = f'results/{task_type}/{env}'
data_paths = [r'E:\Data\Test\CBSD68\HI', r'E:\Data\Test\Kodak24\HI', r'E:\Data\Test\McMaster\HI']
os.makedirs(root_path, exist_ok=True)
logger = create_logger(root_path, name=f'color_dn_{noise}')

model_list = {
    'SwinIR': r'F:\GraduationThesis\Project\Results\Experiment_2\2120_G.pth',
    # 'Uformer-B': r'',
}


def main():
    # 模型定义
    test_results = {}
    for model_name, pth in model_list.items():
        model = define_model(model_name, pth)
        model.eval()
        model.to(device)
        os.makedirs(os.path.join(root_path, model_name), exist_ok=True)
        test_results[model_name] = {}

        # 数据集
        for data_path in data_paths:
            data_name = data_path.split('\\')[-2]
            logger.info(f'{data_name} start to test')
            img_path = os.path.join(root_path, model_name, data_name)
            os.makedirs(img_path, exist_ok=True)
            test_results[model_name][data_name] = OrderedDict()
            test_results[model_name][data_name]['psnr'] = []
            test_results[model_name][data_name]['ssim'] = []
            test_results[model_name][data_name]['psnr_y'] = []
            test_results[model_name][data_name]['ssim_y'] = []
            psnr, ssim, psnr_y, ssim_y = 0, 0, 0, 0
            # 测试过程
            for idx, path in enumerate(sorted(glob.glob(os.path.join(data_path, '*')))):
                # 读入图片
                filename, img_lq, img_gt = get_image_pair(path, noise)  # HWC-BGR, float32
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

                # 保存图片
                output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                if output.ndim == 3:
                    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HWC-BGR
                output = (output * 255.0).round().astype(np.uint8)
                cv2.imwrite(f'{img_path}/{filename}_SwinIR.png', output)

                # 计算psnr
                img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
                img_gt = img_gt[:h_old, :w_old, ...]
                img_gt = np.squeeze(img_gt)

                psnr = calculate_psnr(output, img_gt, border=border)
                ssim = calculate_ssim(output, img_gt, border=border)
                test_results[model_name][data_name]['psnr'].append(psnr)
                test_results[model_name][data_name]['ssim'].append(ssim)
                if img_gt.ndim == 3:
                    output_y = bgr2ycbcr(output.astype(np.float32) / 255.) * 255.
                    img_gt_y = bgr2ycbcr(img_gt.astype(np.float32) / 255.) * 255.
                    psnr_y = calculate_psnr(output_y, img_gt_y, border=border)
                    ssim_y = calculate_ssim(output_y, img_gt_y, border=border)
                    test_results[model_name][data_name]['psnr_y'].append(psnr_y)
                    test_results[model_name][data_name]['ssim_y'].append(ssim_y)
                logger.info('Testing {:d} {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f}; '
                            'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}; '.
                            format(idx, filename, psnr, ssim, psnr_y, ssim_y))
            logger.info(f'{data_name} finish to test')
            average_psnr = np.mean(test_results[model_name][data_name]['psnr'])
            average_ssim = np.mean(test_results[model_name][data_name]['ssim'])
            average_psnr_y = np.mean(test_results[model_name][data_name]['psnr_y'])
            average_ssim_y = np.mean(test_results[model_name][data_name]['ssim_y'])
            logger.info(
                '{} -- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}  -- Average PSNR_Y/SSIM_Y: {:.2f} dB; {:.4f}'.format(
                    img_path, average_psnr, average_ssim, average_psnr_y, average_ssim_y))


def define_model(model_name, pth):
    model = None
    if model_name == 'SwinIR':
        if os.path.exists(pth):
            logger.info(f'Loading model from {pth}')
        model = SwinIR(upscale=1, in_chans=3, img_size=48, window_size=8,
                       img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                       mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'
        pretrained_model = torch.load(pth)
        model.load_state_dict(
            pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
            strict=True)
    return model


def get_image_pair(path, noise):
    (filename, suffix) = os.path.splitext(os.path.basename(path))
    img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    np.random.seed(seed=10)
    img_lq = img_gt + np.random.normal(0, noise / 255., img_gt.shape)
    return filename, img_lq, img_gt


if __name__ == '__main__':
    main()
