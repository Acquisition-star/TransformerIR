import torch
import torch.nn as nn

from utils.lr_scheduler import CosineAnnealingRestartCyclicLR


class SimpleLinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


if __name__ == "__main__":
    model = SimpleLinearModel(3, 1)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-5)

    # 学习率调度器构建
    lr_scheduler = CosineAnnealingRestartCyclicLR(optimizer, [4, 6], [1, 1], [3e-4, 1e-6])

    iter = 1

    while iter <= 10:
        print('iter:', iter, 'lr:', lr_scheduler.get_last_lr())
        lr_scheduler.step()
        iter += 1

