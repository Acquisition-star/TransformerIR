# 训练命令  
## Baseline训练
### Identity
``python main.py --cfg configs/Denoising/Baseline/restormer_demo0.yaml --dataloader_workers 1 --batch_size 1 --env baseline_0``
### Window Attention
``python main.py --cfg configs/Denoising/Baseline/restormer_demo1.yaml --dataloader_workers 1 --batch_size 1 --env baseline_1``  
### Shifted Window Attention
``python main.py --cfg configs/Denoising/Baseline/restormer_demo2.yaml --dataloader_workers 1 --batch_size 1 --env baseline_2``  
### Channel Attention
``python main.py --cfg configs/Denoising/Baseline/restormer_demo3.yaml --dataloader_workers 1 --batch_size 1 --env baseline_3``  
### Multi-DConv Head Transposed Self-Attention
``python main.py --cfg configs/Denoising/Baseline/restormer_demo4.yaml --dataloader_workers 1 --batch_size 2 --env baseline_4``
### Sparse Attention
``python main.py --cfg configs/Denoising/Baseline/restormer_demo5.yaml --dataloader_workers 1 --batch_size 2 --env baseline_5``  
### Taylor Expanded Multi-head Self-Attention
``python main.py --cfg configs/Denoising/Baseline/restormer_demo6.yaml --dataloader_workers 1 --batch_size 1 --env baseline_6``  
### Frequency Domain-based Self-Attention Solver
``python main.py --cfg configs/Denoising/Baseline/restormer_demo7.yaml --dataloader_workers 1 --batch_size 1 --env baseline_7``  
### Strip Attention
``python main.py --cfg configs/Denoising/Baseline/restormer_demo8.yaml --dataloader_workers 1 --batch_size 1 --env baseline_8`` 


## 论文模型训练
### Restormer
`python main.py --cfg configs/Denoising/Restormer.yaml --dataloader_workers 1 --batch_size 1 --env demo`  
### MB-TaylorFormer V2
`python baseline_train.py --cfg configs/Denoising/MB-TaylorFormerV2-B.yaml --dataloader_workers 1 --batch_size 4 --env demo`  

