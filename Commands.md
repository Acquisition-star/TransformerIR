# 训练命令  
## Baseline训练
### Identity
``python baseline_train.py --cfg configs/Denoising/Baseline/demo.yaml --dataloader_workers 1 --batch_size 24 --env baseline_0``
### Window Attention
``python baseline_train.py --cfg configs/Denoising/Baseline/demo1.yaml --dataloader_workers 1 --batch_size 24 --env baseline_1``  
### Shifted Window Attention
``python baseline_train.py --cfg configs/Denoising/Baseline/demo2.yaml --dataloader_workers 1 --batch_size 24 --env baseline_2``  
### Channel Attention
``python baseline_train.py --cfg configs/Denoising/Baseline/demo3.yaml --dataloader_workers 1 --batch_size 24 --env baseline_3``  
### Frequency Domain-based Self-Attention Solver
``python baseline_train.py --cfg configs/Denoising/Baseline/demo4.yaml --dataloader_workers 1 --batch_size 24 --env baseline_4``
### Multi-Dconv Head Transposed Self-Attention
``python baseline_train.py --cfg configs/Denoising/Baseline/demo5.yaml --dataloader_workers 1 --batch_size 24 --env baseline_5``  
### Sparse Attention
``python baseline_train.py --cfg configs/Denoising/Baseline/demo6.yaml --dataloader_workers 1 --batch_size 24 --env baseline_6``  
### Taylor Expanded Multi-head Self-Attention
``python baseline_train.py --cfg configs/Denoising/Baseline/demo7.yaml --dataloader_workers 1 --batch_size 24 --env baseline_7``  
### Strip Attention
``python baseline_train.py --cfg configs/Denoising/Baseline/demo8.yaml --dataloader_workers 1 --batch_size 4 --env baseline_8`` 


## 论文模型训练
### MB-TaylorFormer V2
`python baseline_train.py --cfg configs/Denoising/MB-TaylorFormerV2-B.yaml --dataloader_workers 1 --batch_size 4 --env demo`  

