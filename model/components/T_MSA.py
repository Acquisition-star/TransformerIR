# MB-TaylorFormer V2: Improved Multi-branch Linear Transformer Expanded by Taylor Formula for Image Restoration
# Taylor Expanded Multi-head Self-Attention

import torch
import torch.nn as nn
from einops import rearrange


class refine_att(nn.Module):
    """Convolutional relative position encoding."""

    def __init__(self, Ch, h, window, path):

        super().__init__()

        if isinstance(window, int):
            # Set the same window size for all attention heads.
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:

            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) *
                            (dilation - 1)) // 2
            cur_conv = nn.Conv2d(
                cur_head_split * Ch * path,
                cur_head_split * path,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split * path,
            )

            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.num_path = path
        self.channel_splits = [x * Ch for x in self.head_splits]

    def forward(self, v, size):
        """foward function"""
        B, h, N, Ch = v.shape
        H, W = size

        # We don't use CLS_TOKEN
        v_img = v

        # q = rearrange(q, '(B p) head N c -> B head N (p c)', B=b // self.num_path, p=self.num_path)
        # k = rearrange(k, '(B p) head c N -> B head (p c) N', B=b // self.num_path, p=self.num_path)
        v_img = rearrange(v_img, "B h (H W) Ch -> B h Ch H W", H=H, W=W)

        # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        # q_img = rearrange(q_img, "(p B) h (H W) Ch -> B p h Ch H W", H=H, W=W, p=self.num_path)
        # k_img = rearrange(k_img, "(p B) h Ch (H W) -> B p h Ch H W", H=H, W=W, p=self.num_path)
        v_img = rearrange(v_img, "b h Ch H W -> b (h Ch) H W", H=H, W=W)
        # qk_concat = rearrange(qk_concat, "(p B) c H W -> B (p c) H W", H=H, W=W, p=self.num_path)
        # Split according to channels.
        # qk_concat= rearrange(qk_concat, "B h Ch H W -> B (h Ch) H W", H=H, W=W)
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        v_img_list_reshape = []
        for i in range(len(v_img_list)):
            v_img_list_reshape.append(rearrange(v_img_list[i], "(p B) c H W -> B (p c) H W", H=H, W=W, p=self.num_path))
        v_att_list = [
            conv(x) for conv, x in zip(self.conv_list, v_img_list_reshape)
        ]
        v_img_list_reshape = []
        for i in range(len(v_att_list)):
            v_img_list_reshape.append(rearrange(v_att_list[i], "B (p c) H W -> (p B) c H W", H=H, W=W, p=self.num_path))

        v_att = torch.cat(v_img_list_reshape, dim=1)
        # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].
        v_att = rearrange(v_att, "B (h Ch) H W -> B h (H W) Ch", h=h)

        return v_att


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, shared_refine_att=None, qk_norm=1, path=2, focusing_factor=4):
        super(Attention, self).__init__()
        self.norm = qk_norm
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(path, num_heads, 1, 1))
        # self.Leakyrelu=nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.qkv = nn.Conv2d(dim * path, dim * 3 * path, kernel_size=1, groups=path, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3 * path, dim * 3 * path, kernel_size=3, stride=1, padding=1,
                                    groups=dim * 3 * path, bias=bias)
        self.project_out = nn.Conv2d(dim * path, dim * path, kernel_size=1, groups=path, bias=bias)
        self.num_path = path
        if num_heads == 8:
            crpe_window = {
                3: 2,
                5: 3,
                7: 3
            }
        elif num_heads == 1:
            crpe_window = {
                3: 1,
            }
        elif num_heads == 2:
            crpe_window = {
                3: 2,
            }
        elif num_heads == 4:
            crpe_window = {
                3: 2,
                5: 2,
            }
        self.refine_att = refine_att(Ch=dim // num_heads,
                                     h=num_heads,
                                     window=crpe_window,
                                     path=path)
        self.focusing_factor = focusing_factor
        self.scale = nn.Parameter(torch.ones(path, num_heads, 1, 1))
        # self.N=N
        # self.one_M=nn.Parameter(torch.full((N, dim // self.num_heads), N), requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.shape

        relu = nn.ReLU(inplace=False)
        x = rearrange(x, '(p B) c h w -> B (p c) h w', B=b // self.num_path, p=self.num_path)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = rearrange(qkv, 'B (p c) h w -> (p B) c h w', B=b // self.num_path, p=self.num_path)
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)

        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        # scale = nn.Softplus()(self.scale)
        # scale=rearrange(scale, 'b n (head c) -> b head n c', head=self.num_heads)
        # q=q/scale
        # scale=rearrange(scale, 'b head n c -> b head c n', head=self.num_heads)
        # k=k/scale
        # q = torch.nn.functional.normalize(q, dim=-1)
        # q_norm=torch.norm(q,p=2,dim=-1,keepdim=True)
        # q=torch.div(q,q_norm+1e-8)
        # k_norm=torch.norm(k,p=2,dim=-2,keepdim=True)
        # q_norm=q.norm(dim=-1, keepdim=True)
        # k_norm=k.norm(dim=-2, keepdim=True)
        q_norm = torch.norm(q, p=2, dim=-1, keepdim=True) + 1e-8
        q_1 = torch.div(q, q_norm)
        k_norm = torch.norm(k, p=2, dim=-2, keepdim=True) + 1e-8
        k_1 = torch.div(k, k_norm)

        q_2 = relu(q) ** self.focusing_factor
        k_2 = relu(k) ** self.focusing_factor

        # q=q**self.focusing_factor
        # k=k**self.focusing_factor#
        q_2 = (q_2 / (q_2.norm(dim=-1, keepdim=True) + 1e-8))  # *q_norm
        k_2 = (k_2 / (k_2.norm(dim=-2, keepdim=True) + 1e-8))  # * k_norm
        # attention_map=1+q@k
        # a_n=torch.sum(attention_map,-1).unsqueeze(-1)

        #  N=h*w
        # a_n=N+q@(k@torch.ones_like(attention_map))+1e-8
        # print(a_n*torch.ones_like(v, *, dtype=None, layout=None, requires_grad=False, memory_format=torch.preserve_format))

        #    attention_map=torch.div(attention_map,a_n)

        #   attention_map=attention_map[0][0].squeeze(0)
        #   attention_map = rearrange(attention_map[50], '(h w) -> h w', h=h,w=w)
        #    print(attention_map)
        #   attention_map=attention_map.cpu().numpy()

        #   import matplotlib.pyplot as plt
        #   from matplotlib import cm

        #   plt.imshow(attention_map, interpolation='none', cmap=cm.coolwarm, origin='lower')
        #   plt.colorbar(shrink=.92)

        #   plt.xticks(())
        #   plt.yticks(())
        # plt.show()
        # k=torch.div(k,k_norm+1e-8)
        # k = torch.nn.functional.normalize(k, dim=-2)
        refine_weight = self.refine_att(v, size=(h, w))

        # refine_weight=self.Leakyrelu(refine_weight)
        refine_weight = self.sigmoid(refine_weight)
        attn_2 = k_2 @ v
        attn_1 = k_1 @ v

        # attn = attn.softmax(dim=-1)
        # if h*w != self.N:
        #    self.one_M = nn.Parameter(torch.full((h*w, c// self.num_heads), h*w), requires_grad=False)
        #    self.N=h*w
        # print(torch.sum(k, dim=-1).unsqueeze(3).shape)
        scale = self.sigmoid(self.scale)
        out_numerator = torch.sum(v, dim=-2).unsqueeze(2) + (q_1 @ attn_1) + scale.repeat_interleave(b // self.num_path,
                                                                                                     0) * (
                                q_2 @ attn_2)  # self.one_M \

        N = h * w
        # print(q@torch.sum(k, dim=-1).unsqueeze(3).repeat(1,1,1,c//self.num_heads).shape)
        out_denominator = torch.full((N, c // self.num_heads), N).to(q.device) + q_1 @ torch.sum(k_1, dim=-1).unsqueeze(
            3).repeat(1, 1, 1, c // self.num_heads) + q_2 @ torch.sum(
            scale.repeat_interleave(b // self.num_path, 0) * k_2, dim=-1).unsqueeze(
            3).repeat(1, 1, 1, c // self.num_heads) + 1e-8

        # print(torch.ones_like(v))
        # print((k@torch.ones_like(v))- torch.sum(k, dim=-1).unsqueeze(3).repeat(1,1,1,c//self.num_heads))
        # print('1',out_numerator,'2',out_denominator)

        # out_denominator = q@torch.sum(k, dim=-1).unsqueeze(3).repeat(1,1,1,c//self.num_heads)+1e-8
        # out=torch.div(out_numerator,out_denominator)*self.temperature*refine_weight

        out = torch.div(out_numerator, out_denominator)

        out = out * (self.temperature.repeat_interleave(b // self.num_path, 0)) + refine_weight

        out = rearrange(out, 'b head (h w) c-> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = rearrange(out, '(p b) c h w-> b (p c) h w', h=h, w=w, p=self.num_path)
        out = self.project_out(out)
        out = rearrange(out, 'b (p c) h w-> (p b) c h w', h=h, w=w, p=self.num_path)
        return out


if __name__ == '__main__':
    input = torch.randn(1, 32, 128, 128)
    model = Attention(dim=32, num_heads=8, path=1, bias=False)
    out = model(input)
    print(out.shape)
