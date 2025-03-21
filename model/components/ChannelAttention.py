import torch
import torch.nn as nn


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
        return x * self.sca(x)


if __name__ == '__main__':
    input = torch.randn(1, 32, 128, 128)
    net = ChannelAttention(channels=32)
    output = net(input)
    print(output.size())
