import torch
import torch.nn as nn

from model.components.MDTA import MDTA
from model.components.SwinAttention import ShiftedWindowAttention


class WindowChannelAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, bias=True):
        super(WindowChannelAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.f1 = MDTA(
            dim=self.dim,
            num_heads=self.num_heads,
            bias=bias,
        )

        self.f2 = ShiftedWindowAttention(
            channels=self.dim,
            window_size=self.window_size,
            num_heads=self.num_heads,
            shifted=False,
            qkv_bias=bias,
            qk_scale=None,
        )

    def forward(self, x):
        x = x + self.f1(x)
        x = x + self.f2(x)
        return x
