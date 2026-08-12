import torch 
import torch.nn as nn

from typing import Callable, Optional

from detectron2.layers import get_norm, ShapeSpec

# ---------------------------------------------------------------------------
# Common primitives
# ---------------------------------------------------------------------------

class ConvNormAct(nn.Module):
    """Conv + Norm + Activation, reusable across all backbone families.

    `order` lets a single class serve both post-activation (ResNet, RegNet,
    DarkNet) and pre-activation (Pre-Act ResNet) block styles.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        dilation=1,
        norm="BN",
        activation: Optional[Callable[[], nn.Module]] = nn.ReLU,
        order: str = "conv_norm_act",  # or "norm_act_conv"
        bias=False,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.order = order
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, groups=groups, dilation=dilation, bias=bias,
        )
        norm_channels = in_channels if order == "norm_act_conv" else out_channels
        self.norm = get_norm(norm, norm_channels) if norm else nn.Identity()
        self.act = activation() if activation is not None else nn.Identity()

    def forward(self, x):
        if self.order == "norm_act_conv":  # pre-activation
            return self.conv(self.act(self.norm(x)))
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class SqueezeExcite(nn.Module):
    """Optional SE gate, used by RegNet-Y and can be toggled for other archs."""

    def __init__(self, channels, se_ratio=0.25):
        super().__init__()
        hidden = max(1, int(channels * se_ratio))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        s = self.avg_pool(x)
        s = self.act(self.fc1(s))
        s = self.gate(self.fc2(s))
        return x * s


def make_shortcut(in_channels, out_channels, stride, norm):
    """Shared downsample/projection shortcut used by every residual block."""
    if in_channels == out_channels and stride == 1:
        return None
    return ConvNormAct(
        in_channels, out_channels, kernel_size=1, stride=stride,
        norm=norm, activation=None,
    )


# ---------------------------------------------------------------------------
# Stems
# ---------------------------------------------------------------------------

class BasicStem(nn.Module):
    """7x7 s2 conv + 3x3 s2 maxpool -- ResNet / Pre-Act ResNet."""

    def __init__(self, in_channels=3, out_channels=64, norm="BN", activation=nn.ReLU):
        super().__init__()
        self.conv = ConvNormAct(in_channels, out_channels, 7, stride=2,
                                 norm=norm, activation=activation)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.out_channels = out_channels
        self.stride = 4

    def forward(self, x):
        return self.pool(self.conv(x))


class SimpleStem(nn.Module):
    """3x3 s2 conv -- RegNet."""

    def __init__(self, in_channels=3, out_channels=32, norm="BN", activation=nn.ReLU):
        super().__init__()
        self.conv = ConvNormAct(in_channels, out_channels, 3, stride=2,
                                 norm=norm, activation=activation)
        self.out_channels = out_channels
        self.stride = 2

    def forward(self, x):
        return self.conv(x)


class DarkStem(nn.Module):
    """3x3 s1 conv, LeakyReLU -- DarkNet-53."""

    def __init__(self, in_channels=3, out_channels=32, norm="BN"):
        super().__init__()
        self.conv = ConvNormAct(
            in_channels, out_channels, 3, stride=1, norm=norm,
            activation=lambda: nn.LeakyReLU(0.1, inplace=True),
        )
        self.out_channels = out_channels
        self.stride = 1

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# Residual blocks (share ConvNormAct + make_shortcut)
# ---------------------------------------------------------------------------

class BottleneckBlock(nn.Module):
    """ResNet: 1x1 -> 3x3 -> 1x1, post-activation, add, relu."""

    def __init__(self, in_channels, out_channels, bottleneck_channels, stride=1,
                 norm="BN", groups=1, dilation=1):
        super().__init__()
        self.shortcut = make_shortcut(in_channels, out_channels, stride, norm)
        self.conv1 = ConvNormAct(in_channels, bottleneck_channels, 1, norm=norm)
        self.conv2 = ConvNormAct(bottleneck_channels, bottleneck_channels, 3,
                                  stride=stride, groups=groups, dilation=dilation, norm=norm)
        self.conv3 = ConvNormAct(bottleneck_channels, out_channels, 1, norm=norm, activation=None)
        self.act_out = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x) if self.shortcut is not None else x
        out = self.conv3(self.conv2(self.conv1(x)))
        return self.act_out(out + identity)


class PreActBottleneckBlock(nn.Module):
    """Pre-Act ResNet: BN-ReLU-Conv ordering, add before any further act."""

    def __init__(self, in_channels, out_channels, bottleneck_channels, stride=1,
                 norm="BN", groups=1, dilation=1):
        super().__init__()
        self.shortcut = make_shortcut(in_channels, out_channels, stride, norm)
        pre = "norm_act_conv"
        self.conv1 = ConvNormAct(in_channels, bottleneck_channels, 1, norm=norm, order=pre)
        self.conv2 = ConvNormAct(bottleneck_channels, bottleneck_channels, 3, stride=stride,
                                  groups=groups, dilation=dilation, norm=norm, order=pre)
        self.conv3 = ConvNormAct(bottleneck_channels, out_channels, 1, norm=norm, order=pre)

    def forward(self, x):
        identity = self.shortcut(x) if self.shortcut is not None else x
        out = self.conv3(self.conv2(self.conv1(x)))
        return out + identity


class ResBottleneckBlock(nn.Module):
    """RegNet block: group-width conv + optional SE (mirrors detectron2 regnet.py)."""

    def __init__(self, in_channels, out_channels, bottleneck_channels, stride=1,
                 norm="BN", group_width=1, se_ratio=None):
        super().__init__()
        groups = bottleneck_channels // group_width if group_width else 1
        self.shortcut = make_shortcut(in_channels, out_channels, stride, norm)
        self.conv1 = ConvNormAct(in_channels, bottleneck_channels, 1, norm=norm)
        self.conv2 = ConvNormAct(bottleneck_channels, bottleneck_channels, 3,
                                  stride=stride, groups=groups, norm=norm)
        self.se = SqueezeExcite(bottleneck_channels, se_ratio) if se_ratio else None
        self.conv3 = ConvNormAct(bottleneck_channels, out_channels, 1, norm=norm, activation=None)
        self.act_out = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x) if self.shortcut is not None else x
        out = self.conv2(self.conv1(x))
        if self.se is not None:
            out = self.se(out)
        out = self.conv3(out)
        return self.act_out(out + identity)


class DarkResidualBlock(nn.Module):
    """DarkNet-53: 1x1 -> 3x3, LeakyReLU, add (no channel-squeeze bottleneck)."""

    def __init__(self, in_channels, out_channels, bottleneck_channels, stride=1, norm="BN"):
        super().__init__()
        leaky = lambda: nn.LeakyReLU(0.1, inplace=True)
        self.shortcut = make_shortcut(in_channels, out_channels, stride, norm)
        self.conv1 = ConvNormAct(in_channels, bottleneck_channels, 1, norm=norm, activation=leaky)
        self.conv2 = ConvNormAct(bottleneck_channels, out_channels, 3, stride=stride,
                                  norm=norm, activation=leaky)

    def forward(self, x):
        identity = self.shortcut(x) if self.shortcut is not None else x
        out = self.conv2(self.conv1(x))
        return out + identity  # DarkNet adds without a post-activation


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class FrozenBatchNorm2d(nn.Module):
    """copy and modified from https://github.com/facebookresearch/detr/blob/master/models/backbone.py
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """
    def __init__(self, num_features, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        n = num_features
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps
        self.num_features = n 

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}".format(**self.__dict__)
        )


def get_activation(act: str, inpace: bool=True):
    '''get activation
    '''
    act = act.lower()
    
    if act == 'silu':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()
    
    elif act == 'gelu':
        m = nn.GELU()
        
    elif act is None:
        m = nn.Identity()
    
    elif isinstance(act, nn.Module):
        m = act

    else:
        raise RuntimeError('')  

    if hasattr(m, 'inplace'):
        m.inplace = inpace
    
    return m 
