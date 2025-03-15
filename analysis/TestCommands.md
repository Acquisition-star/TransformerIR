# 测试命令
#### 测试1
`python test_sidd.py --task_type denoising --env experiment_1 --model_name SwinIR --pth F:\GraduationThesis\Project\TransformerIR\analysis\model_zoo\SwinIR\005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth`
#### 测试2
`python test_sidd.py --task_type denoising --env experiment_2 --model_name SwinIR --pth F:\GraduationThesis\Project\TransformerIR\analysis\model_zoo\SwinIR\005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth`
#### 测试3
`python test_sidd.py --task_type denoising --env experiment_3 --model_name SwinIR --pth F:\GraduationThesis\Project\TransformerIR\analysis\model_zoo\SwinIR\005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth`
#### 测试4
`python test_sidd.py --task_type denoising --env experiment_4 --model_name Uformer-B --pth F:\GraduationThesis\Project\TransformerIR\analysis\model_zoo\Uformer\Uformer_B.pth --crop --img_size 256`  
#### 测试5
`python test_sidd.py --task_type denoising --env experiment_5 --model_name NAFNet_32 --pth F:\GraduationThesis\Project\TransformerIR\analysis\model_zoo\NAFNet\NAFNet-SIDD-width32.pth`  
#### 测试6
`python test_sidd.py --task_type denoising --env experiment_6 --model_name NAFNet_64 --pth F:\GraduationThesis\Project\TransformerIR\analysis\model_zoo\NAFNet\NAFNet-SIDD-width64.pth`  
#### 测试7 bug
`python test_sidd.py --task_type denoising --env experiment_7 --model_name Stripformer --pth F:\GraduationThesis\Project\TransformerIR\analysis\model_zoo\Stripformer\Stripformer_gopro.pth --crop --img_size 256`  

  
## 模型测试命令
#### E1
`python model_analysis.py --model_name SwinIR`  
#### E2
`python model_analysis.py --model_name Uformer-T`  
#### E3
`python model_analysis.py --model_name Uformer-S`  
#### E4
`python model_analysis.py --model_name Uformer-B`
#### E5
`python model_analysis.py --model_name NAFNet-32`  
#### E6
`python model_analysis.py --model_name NAFNet-64`  
#### E7
`python model_analysis.py --model_name Stripformer`  

## 实验测试命令
```angular2html
# 实验1
python test_sidd.py --task_type denoising --env train_experiment_1 --model_name SwinIR --pth F:\GraduationThesis\Project\Results\Experiment_1\ckpt_epoch_294.pth --is_cpk


# 实验2
python test_sidd.py --task_type denoising --env train_experiment_2 --model_name SwinIR --pth F:\GraduationThesis\Project\Results\Experiment_2\2120_G.pth


# 实验3
python test_sidd.py --task_type denoising --env train_experiment_3 --model_name SwinIR --pth F:\GraduationThesis\Project\Results\Experiment_3\ckpt_epoch_15.pth --is_cpk


# 实验4
python test_sidd.py --task_type denoising --env train_experiment_4 --model_name SwinIR --pth F:\GraduationThesis\Project\Results\Experiment_2\2120_G.pth  


# 实验5
python test_sidd.py --task_type denoising --env train_experiment_5 --model_name Uformer-T --pth F:\GraduationThesis\Project\Results\Experiment_5\ckpt_epoch_86.pth --is_cpk --crop --img_size 256  
```
