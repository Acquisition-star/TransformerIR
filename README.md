# TransformerIR
基于注意力机制的图像复原方法

# 环境配置

## 创建环境

```sh
conda create -n IR python=3.11
conda activate TransformerIR
```

## 包导入

```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install keras tensorflow-intel
pip install --upgrade optree>=0.13.0
pip install timm
pip install opencv-python
pip install termcolor==1.1.0 yacs==0.1.8 pyyaml scipy
pip install pandas matplotlib seaborn colorama einops ptflops lpips
```
