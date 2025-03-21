import torch
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_


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


class WindowAttention(nn.Module):
    def __init__(self, channels, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(WindowAttention, self).__init__()
        self.channels = channels
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = channels // num_heads
        self.scale = qk_scale or head_dim ** -0.5

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
        """
        :param x: 输入特征 (B, C, H, W)
        """
        N = self.window_size * self.window_size
        _, C, H, W = x.size()

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

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        windows = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        windows = self.proj(windows)
        windows = self.proj_drop(windows)
        windows.reshape(B_, self.window_size, self.window_size, C)
        x = window_reverse(windows, self.window_size, H, W)
        return x


class ShiftedWindowAttention(nn.Module):
    def __init__(self, channels, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 index=0):
        super(ShiftedWindowAttention, self).__init__()
        self.channels = channels
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = channels // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.shift_size = 0 if (index % 2 == 0) else window_size // 2

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

    def forward(self, x, mask=None):
        """
        :param x: 输入特征 (B, C, H, W)
        :param mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B, C, H, W = x.shape
        N = self.window_size * self.window_size
        x = x.reshape(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x = x.reshape(B, C, H, W)

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
        attn = attn.reshape(B_ // maskW, maskW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        windows = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        windows = self.proj(windows)
        windows = self.proj_drop(windows)
        windows.reshape(B_, self.window_size, self.window_size, C)
        x = window_reverse(windows, self.window_size, H, W)
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

        mask_windows = window_partition(img_mask.permute(0, 3, 1, 2), self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super(ChannelAttention, self).__init__()
        self.channels = channels
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

    def forward(self, x):
        return self.sca(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # B, H * W, C
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class BaseBlock(nn.Module):
    def __init__(self, index, channels=32, window_size=8, attn_type=None):
        super(BaseBlock, self).__init__()
        self.channels = channels
        self.window_size = window_size
        self.index = index

        self.norm1 = LayerNorm2d(self.channels)
        self.norm2 = LayerNorm2d(self.channels)

        if attn_type is None:
            self.attention = nn.Identity()
        elif attn_type == 'WindowAttention':
            self.attention = WindowAttention(channels=self.channels, window_size=self.window_size, num_heads=4)
        elif attn_type == 'ShiftedWindowAttention':
            self.attention = ShiftedWindowAttention(channels=self.channels, window_size=self.window_size, num_heads=4,
                                                    index=self.index)
        elif attn_type == 'ChannelAttention':
            self.attention = ChannelAttention(channels=self.channels)
        else:
            raise NotImplementedError('Not implemented attention type {}'.format(attn_type))

        self.mlp = Mlp(in_features=self.channels, hidden_features=2 * self.channels, out_features=self.channels)

    def forward(self, x):
        x = self.norm1(x)
        x = x + self.attention(x)
        x = self.norm2(x)
        x = x + self.mlp(x)
        return x


class TransformerIR(nn.Module):
    def __init__(self, img_size=256, channels=3, window_size=8, embedding_dim=32,
                 middle_blks=2, encoder_blk_nums=None, decoder_blk_nums=None, attn_type=None):
        super(TransformerIR, self).__init__()
        self.img_size = img_size
        self.channels = channels
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.middle_blks = middle_blks
        self.attn_type = attn_type
        self.encoder_blk_nums = encoder_blk_nums if encoder_blk_nums is not None else [2, 2]
        self.decoder_blk_nums = decoder_blk_nums if decoder_blk_nums is not None else [2, 2]
        assert len(self.encoder_blk_nums) == len(self.decoder_blk_nums), \
            f'{len(self.encoder_blk_nums)} != {len(self.decoder_blk_nums)}'
        self.blk_length = len(self.encoder_blk_nums)

        self.intro = nn.Conv2d(in_channels=self.channels,
                               out_channels=self.embedding_dim,
                               kernel_size=3,
                               padding=1,
                               stride=1)
        self.outro = nn.Conv2d(in_channels=self.embedding_dim,
                               out_channels=self.channels,
                               kernel_size=3,
                               padding=1,
                               stride=1)
        self.encoders = nn.ModuleList()  # 编码过程
        self.decoders = nn.ModuleList()  # 解码过程
        self.up_samples = nn.ModuleList()  # 上采样层
        self.down_samples = nn.ModuleList()  # 下采样层

        # 编码件
        for index, num in enumerate(self.encoder_blk_nums):
            cur_channels = self.embedding_dim * (2 ** index)
            self.encoders.append(
                nn.Sequential(
                    *[BaseBlock(channels=cur_channels, window_size=self.window_size, attn_type=self.attn_type,
                                index=index)
                      for _ in range(num)]
                )
            )
            self.down_samples.append(
                nn.Conv2d(in_channels=cur_channels, out_channels=2 * cur_channels, kernel_size=2, stride=2, padding=0)
            )

        # 中间件
        self.middle_blks = nn.Sequential(
            *[BaseBlock(channels=self.embedding_dim * (2 ** len(self.encoder_blk_nums)), window_size=self.window_size,
                        attn_type=self.attn_type, index=index)
              for index in range(self.middle_blks)]
        )

        # 解码件
        for index, num in enumerate(self.decoder_blk_nums):
            cur_channels = self.embedding_dim * (2 ** (self.blk_length - index))
            self.up_samples.append(
                nn.ConvTranspose2d(in_channels=cur_channels, out_channels=cur_channels // 2, kernel_size=2, stride=2)
            )
            self.decoders.append(
                nn.Sequential(
                    *[BaseBlock(channels=cur_channels // 2, window_size=self.window_size, attn_type=self.attn_type,
                                index=index)
                      for _ in range(num)]
                )
            )

    def forward(self, x):
        # B, C, H, W = x.size()
        shortcut = x.contiguous()
        x = self.intro(x)

        encoder_outputs = []

        for encoder, down_sample in zip(self.encoders, self.down_samples):
            x = encoder(x)
            encoder_outputs.append(x)
            x = down_sample(x)

        x = self.middle_blks(x)

        for decoder, up_sample, encoder_output in zip(self.decoders, self.up_samples, encoder_outputs[::-1]):
            x = up_sample(x)
            x = x + encoder_output
            x = decoder(x)

        x = self.outro(x)
        x = x + shortcut

        return x


if __name__ == '__main__':
    model = TransformerIR(attn_type='ShiftedWindowAttention')
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    print(out.shape)
