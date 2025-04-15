import torch
import torch.nn as nn
from einops import rearrange

from model.components.MDTA import MDTA
from model.components.SwinAttention import ShiftedWindowAttention

try:
    from timm.layers import trunc_normal_
except ImportError:
    from timm.models.layers import trunc_normal_


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


def window_partition(x, window_size):
    """
    将特征图进行窗口划分
    :param x: (B, C, H, W)
    :param window_size: 窗口大小
    :return: (B * num_windows. C, window_size, window_size)
    """
    B, C, H, W = x.size()
    windows = x.reshape(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = windows.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, window_size, window_size)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    将特征图复原
    :param windows: (B * num_windows, C, window_size, window_size)
    :param window_size: 窗口大小
    :param H: Height of image
    :param W: Width of image
    :return: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, -1, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, H, W)
    return x


class MyAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, attn_drop=0., proj_drop=0., bias=True):
        super(MyAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 相对位置矩阵
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        N = self.window_size * self.window_size

        # 窗口划分
        windows = window_partition(x, window_size=self.window_size)  # B_, C, Ww, Wh

        q, k, v = self.qkv_dwconv(self.qkv(windows)).chunk(3, dim=1)  # B_, C, Ww, Wh
        q = q * self.scale

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn_channel = (q @ k.transpose(-2, -1)) * self.temperature  # B_, head, C // head, C // head
        attn = (q.transpose(-2, -1) @ k)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww, Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, head, N, N
        attn_channel = self.attn_drop(self.softmax(attn_channel))
        attn = self.attn_drop(self.softmax(attn))

        out = (attn_channel @ v @ attn)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=self.window_size,
                        w=self.window_size)
        out = window_reverse(out, self.window_size, H, W)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out
