import torch
from torch import nn

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv1d(1, 1, 1, padding=0, padding_mode='reflect')

        # self.sa1 = nn.Conv1d(1, 1, 1, padding=0, padding_mode='reflect')

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2.unsqueeze(1))

        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, num_channels=64, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv1d(1, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim // reduction, dim, 1),
        )

    def forward(self, x):
        x_gap = self.gap(x.unsqueeze(1))
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        # self.pa2 = nn.Conv1d(2 * dim, dim, 7, padding=3, padding_mode='reflect')
        # self.pa = nn.Conv1d(3, 1, 1, padding=0, padding_mode='reflect')
        self.pa2 = nn.Conv1d(3, 1, 1, padding=0, padding_mode='reflect')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn):
        # 原始 pattn1 torch.Size([256, 2])
        # 传入 x torch.Size([64, 256])
        pattn = pattn.permute(0, 2, 1)  # torch.Size([2, 256])  # batch_size x 2 x 256
        x3 = torch.cat([x.unsqueeze(1), pattn], dim=1)
        x3 = x3.squeeze(1)     # batch_size x 3 x 256

        pattn3 = self.pa2(x3)    # batch_size x 1 x 256
        pattn3 = self.sigmoid(pattn3)

        return pattn3