import torch
import torch.nn as nn
import torch.nn.functional as F

from esnet.models.cga import SpatialAttention, ChannelAttention, PixelAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, k_dim, v_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        # 定义线性投影层，用于将输入变换到多头注意力空间
        self.proj_q = nn.Linear(in_dim, k_dim * num_heads, bias=False)
        self.proj_k = nn.Linear(in_dim, k_dim * num_heads, bias=False)
        self.proj_v = nn.Linear(in_dim, v_dim * num_heads, bias=False)
        # 定义多头注意力的线性输出层
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim)

    def forward(self, x, mask=None):
        # batch_size, seq_len, in_dim = x.size()
        batch_size, in_dim = x.size()  # 64x133

        # 对输入进行线性投影, 将每个头的查询、键、值进行切分和拼接
        q = self.proj_q(x).view(batch_size, self.num_heads, self.k_dim).permute(0, 1, 2)
        k = self.proj_k(x).view(batch_size, self.num_heads, self.k_dim).permute(0, 2, 1)
        v = self.proj_v(x).view(batch_size, self.num_heads, self.v_dim).permute(0, 1, 2)
        # 计算注意力权重和输出结果
        attn = torch.matmul(q, k) / self.k_dim ** 0.5  # 注意力得分

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)  # 注意力权重参数
        output = torch.matmul(attn, v).contiguous().view(batch_size, -1)  # 输出结果
        # 对多头注意力输出进行线性变换和输出
        output = self.proj_o(output)

        return output


class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        # self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim2)

    def forward(self, x1, x2, mask=None):
        # batch_size, seq_len1, in_dim1 = x1.size()
        # seq_len2 = x2.size(1)

        batch_size, in_dim1 = x1.size()
        in_dim2 = x2.size(1)

        q1 = self.proj_q1(x1).view(batch_size, self.num_heads, self.k_dim).permute(0, 1, 2)
        k2 = self.proj_k2(x2).view(batch_size, self.num_heads, self.k_dim).permute(0, 2, 1)
        v2 = self.proj_v2(x2).view(batch_size, self.num_heads, self.v_dim).permute(0, 1, 2)

        attn = torch.matmul(q1, k2) / self.k_dim ** 0.5

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2).contiguous().view(batch_size, -1)
        output = self.proj_o(output)

        return output


class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv1d(1, 1, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)   # batch_size x 256 x 1
        sattn = self.sa(initial)   # batch_size x 1 x 2
        # print(cattn.shape)  # torch.Size([256, 1])
        # print(sattn.shape)  # torch.Size([1, 2])
        pattn1 = sattn + cattn  # batch_size x 256 x 2
        # print(pattn1.shape)   torch.Size([256, 2])
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        # print("CGAFusion: \n", pattn2)
        pattn2 = pattn2.squeeze(1)
        # result = initial + pattn2 * x + (1 - pattn2) * y
        # print("pattn2 * x + (1 - pattn2) * y: \n", (pattn2 * x + (1 - pattn2) * y).shape)
        # result = (pattn2 * y + (1 - pattn2) * x).unsqueeze(1)  # esnet1
        result = (pattn2 * x + (1 - pattn2) * y).unsqueeze(1)  # esnet
        # print("result: \n", result.shape)
        result = self.conv(result)
        # print("result: \n", result)
        # result = self.conv(result).squeeze(1)
        return result

    # def forward(self, x, y):
    #     initial = x + y
    #     cattn = self.ca(initial)   # batch_size x 256 x 1
    #     sattn = self.sa(initial)   # batch_size x 1 x 2
    #     # print(cattn.shape)  # torch.Size([256, 1])
    #     # print(sattn.shape)  # torch.Size([1, 2])
    #     pattn1 = sattn + cattn  # batch_size x 256 x 2
    #     # print(pattn1.shape)   torch.Size([256, 2])
    #     pattn2 = self.sigmoid(self.pa(initial, pattn1))
    #     # print("CGAFusion: \n", pattn2)
    #     pattn2 = pattn2.squeeze(1)
    #     result = initial + pattn2 * x + (1 - pattn2) * y
    #     # result = self.conv(result).squeeze(1)
    #     return result