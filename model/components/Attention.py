import torch
import torch.nn as nn

from model.components.ChannelAttention import ChannelAttention
from model.components.FSAS import FSAS
from model.components.MDTA import MDTA
from model.components.SparseGSA import SparseAttention
from model.components.SwinAttention import ShiftedWindowAttention
from model.components.T_MSA import Attention
from model.components.WindowChannelAttention import WindowChannelAttention, MyAttention


def build_attention(index, dim=32, window_size=8, num_heads=8, bias=True, attn_type='ShiftedWindowAttention'):
    attn = None
    if attn_type == 'Identity':
        attn = nn.Identity()
    elif attn_type == 'WindowAttention':
        attn = ShiftedWindowAttention(
            channels=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=bias,
            qk_scale=None,
        )
    elif attn_type == 'ShiftedWindowAttention':
        attn = ShiftedWindowAttention(
            channels=dim,
            window_size=window_size,
            num_heads=num_heads,
            shifted=False if index % 2 == 0 else True,
            qkv_bias=bias,
            qk_scale=None,
        )
    elif attn_type == 'ChannelAttention':
        attn = ChannelAttention(channels=dim)
    elif attn_type == 'Frequency domain-based self-attention solver':
        attn = FSAS(dim=dim, bias=bias)
    elif attn_type == 'Multi-DConv Head Transposed Self-Attention':
        attn = MDTA(dim=dim, num_heads=num_heads, bias=bias)
    elif attn_type == 'SparseAttention':
        attn = SparseAttention(dim=dim, num_heads=num_heads, bias=bias)
    elif attn_type == 'Taylor Expanded Multi-head Self-Attention':
        attn = Attention(dim=dim, num_heads=num_heads, bias=bias, path=1)
    elif attn_type == 'WindowChannelAttention':
        attn = WindowChannelAttention(dim=dim, window_size=window_size, num_heads=num_heads, bias=bias)
    elif attn_type == 'MyAttention':
        attn = MyAttention(dim=dim, window_size=window_size, num_heads=num_heads, bias=bias)
    else:
        raise ValueError('Unknown attention type {}'.format(attn_type))
    return attn
