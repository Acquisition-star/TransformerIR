import os
import torch
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

# 工具函数
from utils.util import calculate_psnr, calculate_ssim, bgr2ycbcr
from utils.logger import create_logger
from utils.checkpoint import load_checkpoint_model
from data.dataset_denoising import Dataset_denoising_val

# 模型引入
from model.nets.SwinIR import SwinIR
from model.nets.Uformer import Uformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

noise = 15
experiment_index = 1
task_type = 'denoising'
env = f'example_{experiment_index}'
root_path = f'results/{task_type}/{env}'

os.makedirs(root_path, exist_ok=True)

model_info = {
    'name': 'SwinIR',
    'pth': r'F:\GraduationThesis\Project\TransformerIR\analysis\model_zoo\SwinIR\005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth',
    'is_cpk': False
}

logger = create_logger(root_path, name=f"{model_info['name']}_{env}")

data_list = [
    {
        'name': 'SIDD',
        'H_path': r'E:\Data\SIDD\val\groundtruth',
        'L_path': r'E:\Data\SIDD\val\input',
    },
    {
        'name': 'CBSD68_n15',
        'H_path': r'E:\Data\Test\CBSD68\HI',
        'L_path': r'E:\Data\Test\CBSD68\Noise15',
    },
    {
        'name': 'CBSD68_n25',
        'H_path': r'E:\Data\Test\CBSD68\HI',
        'L_path': r'E:\Data\Test\CBSD68\Noise25',
    },
    {
        'name': 'CBSD68_n50',
        'H_path': r'E:\Data\Test\CBSD68\HI',
        'L_path': r'E:\Data\Test\CBSD68\Noise50',
    },
    {
        'name': 'Kodak24_n15',
        'H_path': r'E:\Data\Test\Kodak24\HI',
        'L_path': r'E:\Data\Test\Kodak24\Noise15',
    },
    {
        'name': 'Kodak24_n25',
        'H_path': r'E:\Data\Test\Kodak24\HI',
        'L_path': r'E:\Data\Test\Kodak24\Noise25',
    },
    {
        'name': 'Kodak24_n50',
        'H_path': r'E:\Data\Test\Kodak24\HI',
        'L_path': r'E:\Data\Test\Kodak24\Noise50',
    },
    {
        'name': 'McMaster_n15',
        'H_path': r'E:\Data\Test\McMaster\HI',
        'L_path': r'E:\Data\Test\McMaster\Noise15',
    },
    {
        'name': 'McMaster_n25',
        'H_path': r'E:\Data\Test\McMaster\HI',
        'L_path': r'E:\Data\Test\McMaster\Noise25',
    },
    {
        'name': 'McMaster_n50',
        'H_path': r'E:\Data\Test\McMaster\HI',
        'L_path': r'E:\Data\Test\McMaster\Noise50',
    },
    # {
    #     'name': 'urban100_n15',
    #     'H_path': r'E:\Data\Test\urban100',
    #     'L_path': r'E:\Data\Test\urban100\Noise15',
    # },
    # {
    #     'name': 'urban100_n25',
    #     'H_path': r'E:\Data\Test\urban100',
    #     'L_path': r'E:\Data\Test\urban100\Noise25',
    # },
    # {
    #     'name': 'urban100_n50',
    #     'H_path': r'E:\Data\Test\urban100',
    #     'L_path': r'E:\Data\Test\urban100\Noise50',
    # },
]


def define_model(model_info):
    model = None
    if 'SwinIR' in model_info['name']:
        if os.path.exists(model_info['pth']):
            logger.info(f"Loading model from {model_info['pth']}")
        model = SwinIR(
            upscale=1,
            in_chans=3,
            img_size=128,
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='',
            resi_connection='1conv'
        )
        if model_info['is_cpk']:
            load_checkpoint_model(model, model_info['pth'], logger)
        else:
            param_key_g = 'params'
            pretrained_model = torch.load(model_info['pth'])
            model.load_state_dict(
                pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
                strict=True)
    elif 'Uformer' in model_info['name']:
        if os.path.exists(model_info['pth']):
            logger.info(f"Loading model from {model_info['pth']}")
        model = Uformer()
        model = Uformer(
            img_size=128,
            embed_dim=32,
            win_size=8,
            token_projection='linear',
            token_mlp='leff',
            depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],
            modulator=True,
            dd_in=3
        )
        if model_info['is_cpk']:
            load_checkpoint_model(model, model_info['pth'], logger)
        else:
            cpk = torch.load(model_info['pth'])
            try:
                model.load_state_dict(cpk["state_dict"])
            except:
                state_dict = cpk["state_dict"]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if 'module.' in k else k
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
    else:
        raise Exception("Model error!")
    return model


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


def main():
    # test_results = {}
    for data_info in data_list:
        data_set = Dataset_denoising_val(sigma=noise, input_channels=3, H_path=[data_info['H_path']],
                                         L_path=[data_info['L_path']])
        model = define_model(model_info)
        model.to(device)
        model.eval()
        logger.info(f"Number of params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        logger.info(f"{model_info['name']} start to test on {data_info['name']}!")
        avg_psnr, avg_ssim, avg_psnr_y, avg_ssim_y = 0.0, 0.0, 0.0, 0.0
        for index in range(0, len(data_set)):
            L_img, H_img = data_set[index]['L'], data_set[index]['H']  # CHW-RGB
            image_name = os.path.basename(data_set[index]['H_path'])

            L_img = L_img.unsqueeze(0).to(device)  # CHW-RGB --> NCHW-RGB

            with torch.no_grad():
                output = model(L_img)

            visual_H = trans_img(H_img)
            visual_O = trans_img(output.squeeze(0).cpu())

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

        avg_psnr /= len(data_set)
        avg_ssim /= len(data_set)
        avg_psnr_y /= len(data_set)
        avg_ssim_y /= len(data_set)

        logger.info(
            '{} -- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}  -- Average PSNR_Y/SSIM_Y: {:.2f} dB; {:.4f}'.format(
                data_info['name'], avg_psnr, avg_ssim, avg_psnr_y, avg_ssim_y))

        logger.info(f"{data_info['name']} finish to test!!!\n\n")


if __name__ == '__main__':
    main()
