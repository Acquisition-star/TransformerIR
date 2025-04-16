## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

## Download training and testing data for single-image motion deblurring task
import os
import gdown
import shutil

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='train, test or train-test')
parser.add_argument('--dataset', type=str, default='GoPro', help='all, GoPro, HIDE, RealBlur_R, RealBlur_J')
parser.add_argument('--output', type=str, default=r'D:\Data\DownloadData', help='output folder')
args = parser.parse_args()

### Google drive IDs ######
GoPro_train = '1zgALzrLCC_tcXKu_iHQTHukKUVT1aodI'  ## https://drive.google.com/file/d/1zgALzrLCC_tcXKu_iHQTHukKUVT1aodI/view?usp=sharing
GoPro_test = '1k6DTSHu4saUgrGTYkkZXTptILyG9RRll'  ## https://drive.google.com/file/d/1k6DTSHu4saUgrGTYkkZXTptILyG9RRll/view?usp=sharing
HIDE_test = '1XRomKYJF1H92g1EuD06pCQe4o6HlwB7A'  ## https://drive.google.com/file/d/1XRomKYJF1H92g1EuD06pCQe4o6HlwB7A/view?usp=sharing
RealBlurR_test = '1glgeWXCy7Y0qWDc0MXBTUlZYJf8984hS'  ## https://drive.google.com/file/d/1glgeWXCy7Y0qWDc0MXBTUlZYJf8984hS/view?usp=sharing
RealBlurJ_test = '1Rb1DhhXmX7IXfilQ-zL9aGjQfAAvQTrW'  ## https://drive.google.com/file/d/1Rb1DhhXmX7IXfilQ-zL9aGjQfAAvQTrW/view?usp=sharing

dataset = args.dataset

for data in args.data.split('-'):
    if data == 'train':
        print('GoPro Training Data!')
        os.makedirs(os.path.join(args.output, 'GoPro'), exist_ok=True)
        gdown.download(id=GoPro_train, output=f'{args.output}/GoPro/train.zip', quiet=False)
        # os.system(f'gdrive download {GoPro_train} --path {args.output}/GoPro')
        print('Extracting GoPro data...')
        shutil.unpack_archive(f'{args.output}/GoPro/train.zip', f'{args.output}/GoPro')
        # os.rename(os.path.join(args.output, 'GoPro', 'train'), os.path.join('Datasets', 'Downloads', 'GoPro'))
        os.remove(f'{args.output}/GoPro/train.zip')

    if data == 'test':
        if dataset == 'all' or dataset == 'GoPro':
            print('GoPro Testing Data!')
            os.makedirs(os.path.join(args.output, 'GoPro'), exist_ok=True)
            gdown.download(id=GoPro_test, output=f'{args.output}/GoPro/test.zip', quiet=False)
            # os.system(f'gdrive download {GoPro_test} --path {args.output}/GoPro')
            print('Extracting GoPro Data...')
            shutil.unpack_archive(f'{args.output}/GoPro/test.zip', f'{args.output}/GoPro')
            os.remove(f'{args.output}/GoPro/test.zip')

        if dataset == 'all' or dataset == 'HIDE':
            print('HIDE Testing Data!')
            gdown.download(id=HIDE_test, output=f'{args.output}/test.zip', quiet=False)
            # os.system(f'gdrive download {HIDE_test} --path {args.output}/')
            print('Extracting HIDE Data...')
            shutil.unpack_archive(f'{args.output}/test.zip', f'{args.output}')
            os.rename(os.path.join(args.output, 'test'), os.path.join(args.output, 'HIDE'))
            os.remove(f'{args.output}/test.zip')

        if dataset == 'all' or dataset == 'RealBlur_R':
            print('RealBlur_R Testing Data!')
            gdown.download(id=RealBlurR_test, output=f'{args.output}/test.zip', quiet=False)
            # os.system(f'gdrive download {RealBlurR_test} --path {args.output}/')
            print('Extracting RealBlur_R Data...')
            shutil.unpack_archive(f'{args.output}/test.zip', f'{args.output}')
            os.rename(os.path.join(args.output, 'test'), os.path.join(args.output, 'RealBlur_R'))
            os.remove(f'{args.output}/test.zip')

        if dataset == 'all' or dataset == 'RealBlur_J':
            print('RealBlur_J testing Data!')
            gdown.download(id=RealBlurJ_test, output=f'{args.output}/test.zip', quiet=False)
            # os.system(f'gdrive download {RealBlurJ_test} --path {args.output}/')
            print('Extracting RealBlur_J Data...')
            shutil.unpack_archive(f'{args.output}/test.zip', f'{args.output}')
            os.rename(os.path.join(args.output, 'test'), os.path.join(args.output, 'RealBlur_J'))
            os.remove(f'{args.output}/test.zip')

print('Download completed successfully!')
