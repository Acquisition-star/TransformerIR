# 训练命令  
## Baseline训练
### Identity
``python baseline_train.py --cfg configs/Denoising/Baseline/demo.yaml --dataloader_workers 1 --batch_size 24 --env baseline_0``
### WindowAttention
``python baseline_train.py --cfg configs/Denoising/Baseline/b1_demo.yaml --dataloader_workers 1 --batch_size 24 --env baseline_1``  
### Shifted-WindowAttention
``python baseline_train.py --cfg configs/Denoising/Baseline/b2_demo.yaml --dataloader_workers 1 --batch_size 24 --env baseline_2``  
### ChannelAttention
``python baseline_train.py --cfg configs/Denoising/Baseline/b3_demo.yaml --dataloader_workers 1 --batch_size 24 --env baseline_3``  
### Multi-Dconv Head Transposed Self-Attention
``python baseline_train.py --cfg configs/Denoising/Baseline/b4_demo.yaml --dataloader_workers 1 --batch_size 24 --env baseline_4``  
### Channel + Window Attention
``python baseline_train.py --cfg configs/Denoising/Baseline/try.yaml --dataloader_workers 1 --batch_size 12 --env TRY``  
  


## 论文模型训练
### MB-TaylorFormer V2
`python main.py --cfg configs/Denoising/MB-TaylorFormerV2-B.yaml --dataloader_workers 1 --batch_size 4 --env demo`  

