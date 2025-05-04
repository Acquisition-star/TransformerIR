# 测试命令
``python analysis/test_sidd.py --output path --env test --pth ckpt.pth``  
  
``python analysis/test_deblur.py --output path --env test --pth ckpt.pth``
  
``python analysis/model_analysis.py --pth ckpt.pth``  
  
``python analysis/test_deblur.py --env test --pth "F:\GraduationThesis\Project\Results\Restormer\denoising\Window Attention\Experiment_3\ckpt_epoch_100000.pth"``
  
# 模型测试
## Denoising
### Uformer
* Uformer-B
    ``python test_sidd.py --task_type denoising --env uformer_b --cfg ../configs/Denoising/uformer_B.yaml --pth ../analysis/model_zoo/Uformer/Uformer_B.pth --cpk``