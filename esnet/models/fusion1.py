import torch
import torch.nn as nn
import torch.nn.functional as F

from esnet.models.cga1 import SpatialAttention, ChannelAttention, PixelAttention

class CGAFusion1(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion1, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv1d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        # print(cattn.shape)  # torch.Size([256, 1])
        # print(sattn.shape)  # torch.Size([1, 2])
        pattn1 = sattn + cattn
        # print(pattn1.shape)   torch.Size([256, 2])
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        # result = self.conv(result).squeeze(1)
        return result