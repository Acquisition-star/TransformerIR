# 训练命令  
## Baseline训练
### Identity
``python baseline_train.py --cfg configs/Denoising/Baseline/demo.yaml --dataloader_workers 1 --batch_size 48 --env baseline_0``
### WindowAttention
### ChannelAttention
  

#### 实验3
**Local**  
``python main_train.py --cfg configs/Denoising/Local/e3_swinir_denoising_patch96_local.yaml --dataloader_workers 1 --batch_size 2 --env experiment_3``  
**AUtoDL**  
``python main_train.py --cfg configs/Denoising/AutoDL/e3_swinir_denoising_patch96_autodl.yaml --dataloader_workers 4 --batch_size 48 --env experiment_3``  
  
#### 实验4
**Local**  
``python main_train.py --cfg configs/Denoising/Local/e4_swinir_denoising_patch48_local.yaml --dataloader_workers 1 --batch_size 4 --env experiment_4``  
**AUtoDL**  
``python main_train.py --cfg configs/Denoising/AutoDL/e4_swinir_denoising_patch48_autodl.yaml --dataloader_workers 4 --batch_size 32 --env experiment_4``  
  
#### 实验5
**Local**  
``python main_train.py --cfg configs/Denoising/Local/e5_uformer_denoising_patch128_local.yaml --dataloader_workers 1 --batch_size 12 --env experiment_5``  
**AutoDL**  
``python main_train.py --cfg configs/Denoising/AutoDL/e5_uformer_denoising_patch128_autodl.yaml --dataloader_workers 4 --batch_size 48 --env experiment_5``  

#### 实验6
**Local**  
``python main_train.py --cfg configs/Denoising/Local/e6_uformer_denoising_patch128_local.yaml --dataloader_workers 1 --batch_size 12 --env experiment_6``  
**AutoDL**  
``python main_train.py --cfg configs/Denoising/AutoDL/e6_uformer_denoising_patch128_autodl.yaml --dataloader_workers 4 --batch_size 48 --env experiment_6``  

#### 实验7
**Local**  
``python main_train.py --cfg configs/Denoising/Local/e7_uformer_denoising_patch128_local.yaml --dataloader_workers 1 --batch_size 12 --env experiment_7``  
**AutoDL**  
``python main_train.py --cfg configs/Denoising/AutoDL/e7_uformer_denoising_patch128_autodl.yaml --dataloader_workers 4 --batch_size 48 --env experiment_7``

#### 实验8
**Local**  
``python main_train.py --cfg configs/Denoising/Local/e8_nafnet_denoising_patch32_local.yaml --dataloader_workers 1 --batch_size 48 --env experiment_8``  
**AutoDL**  
``python main_train.py --cfg configs/Denoising/AutoDL/e8_nafnet_denoising_patch32_autodl.yaml --dataloader_workers 4 --batch_size 96 --env experiment_8``  
  
#### 实验9
**Local**  
``python main_train.py --cfg configs/Denoising/Local/e9_nafnet_denoising_patch64_local.yaml --dataloader_workers 1 --batch_size 48 --env experiment_9``  
**AutoDL**  
``python main_train.py --cfg configs/Denoising/AutoDL/e9_nafnet_denoising_patch64_autodl.yaml --dataloader_workers 4 --batch_size 48 --env experiment_9``  

#### 实验10
**Local**  
``python main_train.py --cfg configs/Denoising/Local/e10_stripformer_denoising_patch64_local.yaml --dataloader_workers 1 --batch_size 48 --env experiment_10``  
**AutoDL**  
``python main_train.py --cfg configs/Denoising/AutoDL/e10_stripformer_denoising_patch64_autodl.yaml --dataloader_workers 4 --batch_size 48 --env experiment_10``