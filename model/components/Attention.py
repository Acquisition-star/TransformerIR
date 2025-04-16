import torch
import torch.nn as nn

from model.components.ChannelAttention import ChannelAttention
from model.components.FSAS import FSAS
from model.components.MDTA import MDTA
from model.components.SparseGSA import SparseAttention
from model.components.SwinAttention import ShiftedWindowAttention
from model.components.T_MSA import Attention
from model.components.StripAttention import StripAttention
from model.components.EA import EfficientAttention
from model.components.HA import HAB


def build_attention(index, iter, dim, bias, attn_type, attn_config):
    """
    构造注意力模块
    :param index: 模块编号
    :param iter: 模块内编号
    :param dim: 输入通道数
    :param bias: True or False
    :param attn_config: 注意力参数
    :param attn_type: 注意力类型
        --Identity
        --WindowAttention
        --ShiftedWindowAttention
        --ChannelAttention
        --Frequency domain-based self-attention solver
        --Multi-DConv Head Transposed Self-Attention
        --SparseAttention
        --Taylor Expanded Multi-head Self-Attention
    """
    attn = None
    if attn_type == 'Identity':
        attn = nn.Identity()
    elif attn_type == 'Window Attention':
        attn = ShiftedWindowAttention(
            channels=dim,
            window_size=attn_config.window_size[index],
            num_heads=attn_config.num_heads[index],
            qkv_bias=attn_config.qkv_bias,
            qk_scale=attn_config.qk_scale,
            attn_drop=attn_config.attn_drop,
            proj_drop=attn_config.proj_drop,
        )
    elif attn_type == 'Shifted Window Attention':
        attn = ShiftedWindowAttention(
            channels=dim,
            window_size=attn_config.window_size[index],
            num_heads=attn_config.num_heads[index],
            shifted=False if iter % 2 == 0 else True,
            qkv_bias=attn_config.qkv_bias,
            qk_scale=attn_config.qk_scale,
            attn_drop=attn_config.attn_drop,
            proj_drop=attn_config.proj_drop,
        )
    elif attn_type == 'Channel Attention':
        attn = ChannelAttention(channels=dim)
    elif attn_type == 'Frequency domain-based self-attention solver':
        attn = FSAS(dim=dim, bias=bias)
    elif attn_type == 'Multi-DConv Head Transposed Self-Attention':
        attn = MDTA(dim=dim, num_heads=attn_config.num_heads[index], bias=attn_config.bias)
    elif attn_type == 'Sparse Attention':
        attn = SparseAttention(dim=dim, num_heads=attn_config.num_heads[index], bias=attn_config.bias)
    elif attn_type == 'Taylor Expanded Multi-head Self-Attention':
        attn = Attention(
            dim=dim,
            num_heads=attn_config.num_heads[index],
            bias=attn_config.bias,
            qk_norm=attn_config.qk_norm,
            path=attn_config.path,
            focusing_factor=attn_config.focusing_factor,
        )
    elif attn_type == 'Strip Attention':
        attn = StripAttention(
            dim=dim,
            num_heads=attn_config.num_heads[index],
        )
    elif attn_type == 'Efficient Attention':
        attn = EfficientAttention(
            in_channels=dim,
            key_channels=dim,
            head_count=attn_config.num_heads[index],
            value_channels=dim,
        )
    elif attn_type == 'Hybrid Attention':
        attn = HAB(
            dim=dim,
            num_heads=attn_config.num_heads[index],
            window_size=attn_config.window_size[index],
            shift_size=attn_config.shift_size[index][iter],
            compress_ratio=attn_config.compress_ratio,
            squeeze_factor=attn_config.squeeze_factor,
            conv_scale=attn_config.conv_scale,
            qkv_bias=attn_config.qkv_bias,
            qk_scale=attn_config.qk_scale,
            drop=attn_config.drop,
            attn_drop=attn_config.attn_drop,
            drop_path=attn_config.drop_path,
        )
    else:
        raise ValueError('Unknown attention type {}'.format(attn_type))
    return attn
