# 测试命令
``python test_sidd.py --task_type denoising --env demo --cfg config.yaml --pth ckpt.pth --crop --img_size 256``  
  
``python test_sidd.py --task_type denoising --env test1 --cfg F:\GraduationThesis\Project\Results\Restormer\denoising\baseline_0\Experiment_1\config.yaml --pth F:\GraduationThesis\Project\Results\Restormer\denoising\baseline_0\Experiment_1\ckpt_epoch_294000.pth``  
  
``python model_analysis.py --cfg F:\GraduationThesis\Project\Results\Restormer\denoising\baseline_0\config.yaml --pth F:\GraduationThesis\Project\Results\Restormer\denoising\baseline_0\ckpt_epoch_294000.pth``



## TRY
1. 尝试将空间注意力与通道注意力结合  
`python test_sidd.py --task_type denoising --env demo --cfg F:\GraduationThesis\Project\Results\Baseline\TRY\baseline_denoising_color_sidd\config.yaml --pth F:\GraduationThesis\Project\Results\Baseline\TRY\baseline_denoising_color_sidd\checkpoints\ckpt_epoch_99.pth --crop --img_size 128`  
2. MB-TaylorFormer V2  
`python test_sidd.py --env MBTaylor --cfg F:\GraduationThesis\Project\Results\Baseline\MBTaylor\MB-TaylorFormer_V2\config.yaml --pth F:\GraduationThesis\Project\Results\Baseline\MBTaylor\MB-TaylorFormer_V2\checkpoints\ckpt_epoch_21.pth --crop --img_size 256`  

  
# 模型测试
## Denoising
### Uformer
* Uformer-B
    ``python test_sidd.py --task_type denoising --env uformer_b --cfg ../configs/Denoising/uformer_B.yaml --pth ../analysis/model_zoo/Uformer/Uformer_B.pth --cpk``