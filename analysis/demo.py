import os
import torch
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from model.nets.SwinIR import SwinIR
from utils.util import calculate_psnr, calculate_ssim, bgr2ycbcr


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 模型定义
    model = define_SwinIR('model_zoo/SwinIR/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth')
    model.eval()
    model.to(device)

    # 参数设置
    # data_path = [r'E:\Data\Test\CBSD68', r'E:\Data\Test\Kodak24', r'E:\Data\Test\McMaster', r'E:\Data\Test\urban100']
    data_path = r'E:\Data\Test\CBSD68'
    save_path = 'results/color_dn/SwinIR__noise15'
    noise = 15
    border = 0
    window_size = 8

    os.makedirs(save_path, exist_ok=True)
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['psnr_b'] = []
    psnr, ssim, psnr_y, ssim_y, psnr_b = 0, 0, 0, 0, 0

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
        cv2.imwrite(f'{save_path}/{filename}_SwinIR.png', output)

        # 计算psnr
        img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
        img_gt = img_gt[:h_old, :w_old, ...]
        img_gt = np.squeeze(img_gt)

        psnr = calculate_psnr(output, img_gt, border=border)
        ssim = calculate_ssim(output, img_gt, border=border)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        if img_gt.ndim == 3:
            output_y = bgr2ycbcr(output.astype(np.float32) / 255.) * 255.
            img_gt_y = bgr2ycbcr(img_gt.astype(np.float32) / 255.) * 255.
            psnr_y = calculate_psnr(output_y, img_gt_y, border=border)
            ssim_y = calculate_ssim(output_y, img_gt_y, border=border)
            test_results['psnr_y'].append(psnr_y)
            test_results['ssim_y'].append(ssim_y)
        print('Testing {:d} {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f}; '
              'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}; '.
              format(idx, filename, psnr, ssim, psnr_y, ssim_y))

    average_psnr = np.mean(test_results['psnr'])
    average_ssim = np.mean(test_results['ssim'])
    print('\n{} \n-- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}'.format(save_path, average_psnr, average_ssim))
    average_psnr_y = np.mean(test_results['psnr_y'])
    average_ssim_y = np.mean(test_results['ssim_y'])
    print('-- Average PSNR_Y/SSIM_Y: {:.2f} dB; {:.4f}'.format(average_psnr_y, average_ssim_y))


def define_SwinIR(path):
    if os.path.exists(path):
        print(f'Loading model from {path}')
    model = SwinIR(upscale=1, in_chans=3, img_size=128, window_size=8,
                   img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                   mlp_ratio=2, upsampler='', resi_connection='1conv')
    param_key_g = 'params'
    pretrained_model = torch.load(path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
                          strict=True)
    return model


def get_image_pair(path, noise):
    (filename, suffix) = os.path.splitext(os.path.basename(path))
    img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    np.random.seed(seed=0)
    img_lq = img_gt + np.random.normal(0, noise / 255., img_gt.shape)
    return filename, img_lq, img_gt


def show_image(img):
    """
    将通过cv2读取的图片显示
    :param img: HWC-BGR
    :return: None
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    main()
