import torch
import torch.nn as nn

f1 = nn.AdaptiveAvgPool2d(1)
f2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

input = torch.randn(4, 3 * 3, 3, 3, 1)

