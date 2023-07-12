
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SkipFusion(nn.Module):
    def __init__(self,width):
        super(SkipFusion, self).__init__()

        self.global_img1_1=DoubleConv(in_channels=width,out_channels=width)
        self.global_img1_2=Conv2dReLU(in_channels=width*2,out_channels=width,kernel_size=1)
        self.global_img1_3=Conv2dReLU(in_channels=width,out_channels=width,kernel_size=3,padding=1)

        self.global_gaze1_1 = DoubleConv(in_channels=width, out_channels=width)
        self.global_gaze1_2 = Conv2dReLU(in_channels=width * 2, out_channels=width, kernel_size=1)
        self.global_gaze1_3 = Conv2dReLU(in_channels=width, out_channels=width, kernel_size=3, padding=1)

        self.con1 = Conv2dReLU(in_channels=width * 2, out_channels=width, kernel_size=1)
    def forward(self,x,y):

        global_img = self.global_img1_1(x)
        global_gaze = self.global_gaze1_1(y)
        global_img2 = torch.cat([global_gaze, x], dim=1)
        global_gaze2 = torch.cat([global_img, y], dim=1)
        global_img2 = self.global_img1_2(global_img2)
        global_gaze2 = self.global_gaze1_2(global_gaze2)
        global_img2 = self.global_img1_3(global_img2)
        global_gaze2 = self.global_gaze1_3(global_gaze2)
        global_img_gaze = torch.cat([global_img2, global_gaze2], dim=1)
        res = self.con1(global_img_gaze)
        x=x+res
        return  x

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x,y=None):
        return self.maxpool_conv(x)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

class FusionBlock(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.down1=Down(width,width*2)
        self.down2=Down(width*2,width*4)
        self.dw1=PreActBottleneck(cin=3,cout=width*2)
        self.con=Conv2dReLU(in_channels=width*4,out_channels=width*2,kernel_size=1)
    def forward(self,x,dw_feature):
        x=self.down1(x)
        feature=x
        dw_feature=self.dw1(dw_feature)
        con=torch.cat([x,dw_feature],dim=1)
        x=self.con(con)
        x=self.down2(x)
        
        return x

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1/', nn.Sequential(OrderedDict(
                [('unit1/', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}/', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2/', nn.Sequential(OrderedDict(
                [('unit1/', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}/', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3/', nn.Sequential(OrderedDict(
                [('unit1/', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}/', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))
        self.inc = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=1, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))
        self.down1=FusionBlock(width)
        self.down2 = (Down(width*4, width*8))
        self.down3 = (Down(width*8, width*16))
        self.downs=[self.down1,self.down2,self.down3]

        self.skip_fusion1=SkipFusion(width*4)
        self.skip_fusion2=SkipFusion(width*8)
        self.skip_fusions=nn.Sequential(self.skip_fusion1,self.skip_fusion2)





    def forward(self, x,gaze,dw_feature):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        gaze=self.inc(gaze)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            gaze = self.downs[i](gaze,dw_feature)
            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            feat=self.skip_fusions[i](feat,gaze)
            features.append(feat)
        x = self.body[-1](x)
        gaze=self.downs[-1](gaze)
        return x, features[::-1], gaze
