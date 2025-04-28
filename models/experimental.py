# YOLO 🚀 by Ultralytics, GPL-3.0 license
"""
Experimental modules
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import Conv, C3
import math
from utils.downloads import attempt_download


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    # Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  # ch_in, ch_out, kernel, stride, ch_strategy
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, n - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k)**2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList(
            [nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, device=None, inplace=True, fuse=True):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    from models.yolo import Detect, Model

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location='cpu')  # load
        ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

        # Model compatibility updates
        if not hasattr(ckpt, 'stride'):
            ckpt.stride = torch.tensor([32.])
        if hasattr(ckpt, 'names') and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval())  # model in eval mode

    # Module compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace  # torch 1.7.0 compatibility
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
    return model


class channel_att(nn.Module):

    def __init__(self, channel, b=1, gamma=2):
        super(channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class local_att(nn.Module):

    def __init__(self, channel, reduction=16):
        super(local_att, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel,
                                  out_channels=channel // reduction,
                                  kernel_size=1,
                                  stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction,
                             out_channels=channel,
                             kernel_size=1,
                             stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction,
                             out_channels=channel,
                             kernel_size=1,
                             stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


class Channel_Att_Bridge(nn.Module):

    def __init__(self, in_channels, split_att='fc'):
        super().__init__()
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att_layers = nn.Linear(in_channels, in_channels) if split_att == 'fc' else nn.Conv1d(
            in_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t):
        att = self.avgpool(t).squeeze(-1).transpose(-1, -2)
        att = self.get_all_att(att)

        if self.split_att != 'fc':
            att = att.transpose(-1, -2)

        att = self.sigmoid(self.att_layers(att))

        if self.split_att == 'fc':
            att = att.transpose(-1, -2).unsqueeze(-1).expand_as(t)
        else:
            att = att.unsqueeze(-1).expand_as(t)

        return att * t


class Spatial_Att_Bridge(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3), nn.Sigmoid())

    def forward(self, t):
        '''
        处理单个输入张量 t
        '''
        avg_out = torch.mean(t, dim=1, keepdim=True)
        max_out, _ = torch.max(t, dim=1, keepdim=True)
        att = torch.cat([avg_out, max_out], dim=1)
        att = self.conv2d(att)
        return att * t


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class HXConv(nn.Module):

    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        # (1, k) 卷积，卷积核大小为 [1, k]
        self.h_conv = nn.Conv2d(in_channels, out_channels, (1, k), padding=(0, k // 2))
        # (k, 1) 卷积，卷积核大小为 [k, 1]
        self.v_conv = nn.Conv2d(in_channels, out_channels, (k, 1), padding=(k // 2, 0))

    def forward(self, x):
        return self.h_conv(x) + self.v_conv(x)


class HKXF(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.catt = Channel_Att_Bridge(c1)
        # using channel attention to select the most important features
        self.cv2 = HXConv(c_, c2, k=k[1])
        self.Iconv = Conv(c1, c2, 3, 1)
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.weight * x + (1 - self.weight) * self.cv2(self.cv1(self.catt(x)))


class HKC3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k1=3, k2=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(HKXF(c_, c_, shortcut, g, k=(k1, k2), e=1.0) for _ in range(n)))


class GradientOperator(nn.Module):

    def __init__(self, cin):
        super().__init__()
        self.cin = cin
        # 水平梯度卷积核
        self.conv_x = nn.Conv2d(cin, 1, 3, padding=1, bias=False)
        # 垂直梯度卷积核
        self.conv_y = nn.Conv2d(cin, 1, 3, padding=1, bias=False)

        # 初始化近似Sobel算子
        self._init_weights()

    def _init_weights(self):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.conv_x.weight.data = sobel_x.view(1, 1, 3, 3).repeat(1, self.cin, 1, 1)
        self.conv_y.weight.data = sobel_y.view(1, 1, 3, 3).repeat(1, self.cin, 1, 1)

    def forward(self, x):
        grad_x = self.conv_x(x)  # [B,1,H,W]
        grad_y = self.conv_y(x)  # [B,1,H,W]
        return grad_x, grad_y


class OrientationBinning(nn.Module):

    def __init__(self, num_bins=9):
        super().__init__()
        # Quantity of compartments by direction
        self.num_bins = num_bins
        # Directional filter bank: Each filter corresponds to one direction
        self.filters = nn.Conv2d(2, num_bins, 1, bias=False)  #  input is [grad_x, grad_y]

        # Initialize the Angle parameters
        angles = torch.linspace(0, np.pi, num_bins + 1)[:-1]  # 0~180°
        weight_x = torch.cos(angles).view(num_bins, 1)
        weight_y = torch.sin(angles).view(num_bins, 1)
        self.filters.weight.data = torch.cat([weight_x, weight_y], dim=1).unsqueeze(-1).unsqueeze(-1)

    def forward(self, grad_x, grad_y):
        # Merge the gradients and calculate the directional response
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

        # Stack the gradient vectors as channel dimensions
        grad_vector = torch.cat([grad_x, grad_y], dim=1)  # [B,2,H,W]

        # Calculate the response intensities in all directions
        orientation_response = self.filters(grad_vector)  # [B, num_bins, H, W]

        # Soft box sorting is implemented using softmax
        orientation_weights = F.softmax(orientation_response, dim=1)

        # The direction feature map is obtained by weighted merging
        binned = grad_mag * orientation_weights  # [B, num_bins, H, W]
        return binned


class GOE(nn.Module):

    def __init__(self, in_channels=3, num_bins=9):
        super().__init__()
        self.gradient = GradientOperator(in_channels)
        self.binning = OrientationBinning(num_bins)
        self.norm = nn.InstanceNorm2d(num_bins, affine=True)

    def forward(self, x):
        grad_x, grad_y = self.gradient(x)
        binned = self.binning(grad_x, grad_y)
        normalized = self.norm(binned)
        return normalized


class GOEBlock(nn.Module):

    def __init__(self, cin, cout):
        super().__init__()
        self.hog_branch = GOE(cin)
        self.conv = nn.Conv2d(cin + 9, cout, 1)  # 融合原始特征与HOG

    def forward(self, x):
        hog_feat = self.hog_branch(x)
        fused = torch.cat([x, hog_feat], dim=1)
        return self.conv(fused)
