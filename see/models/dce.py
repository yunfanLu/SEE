import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from see.datasets.basic_batch import EVENT_LOW_LIGHT_BATCH as ELB

"""
@inproceedings{Zero-DCE,
 author = {Guo, Chunle Guo and Li, Chongyi and Guo, Jichang and Loy, Chen Change and Hou, Junhui and Kwong, Sam and Cong, Runmin},
 title = {Zero-reference deep curve estimation for low-light image enhancement},
 booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)},
 pages    = {1780-1789},
 month = {June},
 year = {2020}
}
"""


class DepthPointConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DepthPointConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0, groups=1
        )

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out


class DeepCurveEstimationNet(nn.Module):
    def __init__(self, scale_factor=1):
        super(DeepCurveEstimationNet, self).__init__()

        self.scale_factor = scale_factor
        number_feature = 32
        self.e_conv1 = DepthPointConv(3, number_feature)
        self.e_conv2 = DepthPointConv(number_feature, number_feature)
        self.e_conv3 = DepthPointConv(number_feature, number_feature)
        self.e_conv4 = DepthPointConv(number_feature, number_feature)
        self.e_conv5 = DepthPointConv(number_feature * 2, number_feature)
        self.e_conv6 = DepthPointConv(number_feature * 2, number_feature)
        self.e_conv7 = DepthPointConv(number_feature * 2, 3)

    def enhance(self, x, xr):
        x = x + xr * (torch.pow(x, 2) - x)
        x = x + xr * (torch.pow(x, 2) - x)
        x = x + xr * (torch.pow(x, 2) - x)
        x = x + xr * (torch.pow(x, 2) - x)
        x = x + xr * (torch.pow(x, 2) - x)
        x = x + xr * (torch.pow(x, 2) - x)
        x = x + xr * (torch.pow(x, 2) - x)
        x = x + xr * (torch.pow(x, 2) - x)
        return x

    def forward(self, batch):
        x = batch[ELB.LL]
        H, W = x.shape[-2:]
        if self.scale_factor != 1:
            x = F.interpolate(x, scale_factor=1.0 / self.scale_factor, mode="bilinear")
        x1 = F.relu(self.e_conv1(x))
        x2 = F.relu(self.e_conv2(x1))
        x3 = F.relu(self.e_conv3(x2))
        x4 = F.relu(self.e_conv4(x3))
        x5 = F.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = F.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        xr = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        if self.scale_factor != 1:
            xr = F.interpolate(xr, size=(H, W), mode="bilinear")
        enhance_image = self.enhance(x, xr)
        # Store the enhancement curve and the enhanced image
        batch[ELB.LEC] = xr
        batch[ELB.PRD] = enhance_image
        return batch
