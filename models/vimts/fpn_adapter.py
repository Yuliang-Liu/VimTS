import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
import torch
import numpy as np
from .ops.modules import MSDeformAttn
from torch import nn, Tensor
from typing import Optional

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class Conv_BN_ReLU(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=1,
                 stride=1,
                 padding=0):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes,
                              out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class FPEM_v2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPEM_v2, self).__init__()
        planes = out_channels
        self.linear = nn.ModuleList([zero_module(nn.Conv2d(planes, planes, kernel_size=1)) for i in range(4)])
        self.adapter = nn.ModuleList([Conv_Adapter(planes, skip_connect=False) for i in range(4)])


    def forward(self, f1, f2, f3, f4):
        f1 = self.adapter[0](f1)
        f2 = self.adapter[1](f2)
        f3 = self.adapter[2](f3)
        f4 = self.adapter[3](f4)
        return [self.linear[0](f1), self.linear[1](f2), self.linear[2](f3), self.linear[3](f4)]

class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)

        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class T_Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = zero_module(nn.Linear(D_features, D_hidden_features))
        self.D_fc3 = zero_module(nn.Linear(D_features, D_hidden_features))
        self.D_fc2 = zero_module(nn.Linear(D_hidden_features, D_features)) 
        self.s1 = zero_module(nn.MultiheadAttention(D_hidden_features, 8, dropout=0.1))
        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = zero_module(nn.LayerNorm(D_hidden_features))

        self.s2 = zero_module(nn.MultiheadAttention(D_hidden_features, 8, dropout=0.1))
        self.dropout2 = nn.Dropout(0.1)
        self.norm2 = zero_module(nn.LayerNorm(D_hidden_features))

        self.ad = zero_module(Adapter(D_hidden_features, mlp_ratio=2))
        

    def forward(self, x, t):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        ts = self.D_fc3(t)
        tgt = self.s1(ts.flatten(1,2).transpose(0,1), 
                    xs.flatten(1,2).transpose(0,1), 
                    xs.flatten(1,2).transpose(0,1))[0].transpose(0, 1).reshape(ts.shape)
        tgt = ts + self.dropout1(tgt)
        tgt = self.norm1(tgt)

        tgt_inter = torch.swapdims(tgt, 0, 2)
        tgt_inter2 = self.s2(tgt_inter.flatten(0, 1).transpose(0, 1), 
                    tgt_inter.flatten(0, 1).transpose(0, 1), 
                    tgt_inter.flatten(0, 1).transpose(0, 1))[0].transpose(0, 1).reshape(tgt_inter.shape)
        tgt_inter = tgt_inter + self.dropout2(tgt_inter2)
        tgt = torch.swapdims(self.norm2(tgt_inter), 0, 2)
        tgt = self.ad(tgt)
        tgt = self.act(tgt)
        tgt = self.D_fc2(tgt)

        if self.skip_connect:
            x = t + tgt
        else:
            x = tgt
        return x

class TA_Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.ad2 = T_Adapter(D_features, mlp_ratio=0.25)
        self.ad3 = T_Adapter(D_features, mlp_ratio=0.25)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                ):              
        x1 = self.ad2(tgt[:,:,0:1], tgt)
        x2 = self.ad3(tgt[:,:,1:], x1)
        return x2

class Conv_Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Conv2d(D_features, D_hidden_features, 1, 1)
        self.D_fc2 = nn.Conv2d(D_hidden_features, D_features, 1, 1)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        return x
