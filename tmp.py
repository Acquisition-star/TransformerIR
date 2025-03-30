import torch
import torch.nn as nn

f1 = nn.AdaptiveAvgPool2d(1)
f2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

input = torch.randn(1, 32, 128, 128)
output_f1 = f1(input)
output_f2 = f2(output_f1)
print(output_f1.shape)
print(output_f2.shape)


