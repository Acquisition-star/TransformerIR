import torch
import torch.nn as nn

from model.components.WindowAttention import WindowAttention, ShiftedWindowAttention
from model.components.ChannelAttention import ChannelAttention
from model.components.MultiDConvAttention import MultiDConvAttention


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
    def __init__(self, index, channels=32, window_size=8, num_heads=8, attn_type=None):
        super(BaseBlock, self).__init__()
        self.channels = channels
        self.window_size = window_size
        self.index = index
        self.num_heads = num_heads

        self.norm1 = LayerNorm2d(self.channels)
        self.norm2 = LayerNorm2d(self.channels)

        ############################################## Attention Methods ##############################################
        if attn_type is None:
            self.attention = nn.Identity()
        elif attn_type == 'WindowAttention':
            self.attention = WindowAttention(channels=self.channels, window_size=self.window_size,
                                             num_heads=self.num_heads)
        elif attn_type == 'ShiftedWindowAttention':
            self.attention = ShiftedWindowAttention(channels=self.channels, window_size=self.window_size,
                                                    num_heads=self.num_heads,
                                                    index=self.index)
        elif attn_type == 'ChannelAttention':
            self.attention = ChannelAttention(channels=self.channels)
        elif attn_type == 'Multi-Dconv Head Transposed Attention':
            self.attention = MultiDConvAttention(dim=self.channels, num_heads=self.num_heads, bias=False)
        else:
            raise NotImplementedError('Not implemented attention type {}'.format(attn_type))

        self.mlp = Mlp(in_features=self.channels, hidden_features=2 * self.channels, out_features=self.channels)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerIR(nn.Module):
    def __init__(self, img_size=256, channels=3, window_size=8, embedding_dim=32, num_heads=8,
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
                                index=iter)
                      for iter in range(num)]
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
                                index=iter)
                      for iter in range(num)]
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
    model = TransformerIR(attn_type='ChannelAttention')
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    print(out.shape)
