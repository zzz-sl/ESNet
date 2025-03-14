import torch
from torch import nn
from einops.layers.torch import Rearrange


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv1d(1, 1, 1, padding=0, padding_mode='reflect')

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction = 8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv1d(1, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim // reduction, dim, 1),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn

    
class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        # self.pa2 = nn.Conv1d(2 * dim, dim, 7, padding=3, padding_mode='reflect')
        self.pa2 = nn.Conv1d(3, 1, 1, padding=0, padding_mode='reflect')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        # 原始 pattn1 torch.Size([256, 2])
        # 传入 x torch.Size([64, 256])

        pattn1 = pattn1.permute(1, 0)  # torch.Size([2, 256])

        x2 = torch.cat([x, pattn1], dim=0)
        x2 = x2.squeeze(1)

        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2