import torch
import torch.nn as nn
from utils.lr_scheduler import build_scheduler

model = nn.Linear(10, 1)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.002,
    betas=(0.9, 0.999),
    weight_decay=0.0)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    [1, 2, 3],
    0.1
)

print(f'信息：{lr_scheduler.last_epoch}、{lr_scheduler.get_last_lr()}、{lr_scheduler.get_lr()}\n')

for epoch in range(0, 5):
    optimizer.zero_grad()
    optimizer.step()
    print(f'第{epoch}轮开始：')
    print(f'信息：{lr_scheduler.last_epoch}、{lr_scheduler.get_last_lr()}、 {lr_scheduler.get_lr()}')
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    lr_scheduler.step()
    print(f'信息：{lr_scheduler.last_epoch}、{lr_scheduler.get_last_lr()}、 {lr_scheduler.get_lr()}')
    print(optimizer.state_dict()['param_groups'][0]['lr'], '\n')

