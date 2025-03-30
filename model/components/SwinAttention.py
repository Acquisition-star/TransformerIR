# SwinIR: Image Restoration Using Swin Transformer
# Swin Transformer Block

import torch
import torch.nn as nn
from timm.layers import trunc_normal_


def window_partition(x, window_size):
    """
    将特征图进行窗口划分
    :param x: (B, C, H, W)
    :param window_size: 窗口大小
    :return: (B * num_windows, window_size, window_size, C)
    """
    B, C, H, W = x.size()
    windows = x.reshape(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = windows.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    将特征图复原
    :param windows: (B * num_windows, window_size, window_size, C)
    :param window_size: 窗口大小
    :param H: Height of image
    :param W: Width of image
    :return: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return x


class ShiftedWindowAttention(nn.Module):
    def __init__(self,
                 channels,
                 window_size,
                 num_heads,
                 shifted=False,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super(ShiftedWindowAttention, self).__init__()
        self.channels = channels
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = channels // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.shift_size = window_size // 2 if shifted else 0

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

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: 输入特征 (B, C, H, W)
        B, C, H, W = x.shape
        N = self.window_size * self.window_size

        # 位移
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))

        mask = self.calculate_mask((H, W)).to(x.device)

        # 窗口划分
        windows = window_partition(x, self.window_size).reshape(-1, N, C)
        B_ = windows.shape[0]
        qkv = self.qkv(windows).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        attn = attn + relative_position_bias.unsqueeze(0)

        maskW = mask.shape[0]
        attn = attn.view(B_ // maskW, maskW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        windows = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        windows = self.proj(windows)
        windows = self.proj_drop(windows)
        windows.reshape(B_, self.window_size, self.window_size, C)
        x = window_reverse(windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        return x

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask.permute(0, 3, 1, 2),
                                        self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask


if __name__ == '__main__':
    model = ShiftedAttention(channels=32, window_size=8, num_heads=8, shifted=True)
    input = torch.randn((1, 32, 128, 128))
    output = model(input)
    print(output.shape)
