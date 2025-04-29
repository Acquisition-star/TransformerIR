# 训练命令  
``python main.py --cfg configs/base/demo_denoising.yaml --dataloader_workers 1 --batch_size 1 --env denoising``  
  
``python main.py --cfg configs/base/demo_deblur.yaml --dataloader_workers 1 --batch_size 1 --env deblur``  
  
``python main.py --cfg configs/base/demo_conv_denoising.yaml --dataloader_workers 1 --batch_size 1 --env denoising``  
  
``python main.py --cfg configs/base/demo_conv_deblur.yaml --dataloader_workers 1 --batch_size 1 --env deblur``  

## 论文模型训练
### Restormer
`python main.py --cfg configs/Denoising/Restormer.yaml --dataloader_workers 1 --batch_size 1 --env demo`  
### MB-TaylorFormer V2
`python baseline_train.py --cfg configs/Denoising/MB-TaylorFormerV2-B.yaml --dataloader_workers 1 --batch_size 4 --env demo`  

