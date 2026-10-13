"""
Unified Detectron2 Backbone
============================
Single Backbone module capable of instantiating ResNet, Pre-Act ResNet,
RegNet, DarkNet from one GeneralizedStagedNet runtime, plus a separate
DLA(Backbone) class for the tree-aggregation topology.

Register with:
    cfg.MODEL.BACKBONE.NAME = "build_unified_backbone"
    cfg.MODEL.BACKBONE.ARCH = "resnet" | "preresnet" | "regnet" | "darknet" | "dla"

This is a design skeleton: stage width/depth generation for RegNet (the
`generate_regnet_full` style rule from the DDS paper), and the exact DLA
level configuration, are left as stubs (`# TODO`) since they are
arch-specific numeric formulas, not structural code.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.layers import get_norm, ShapeSpec
from detectron2.modeling import Backbone, BACKBONE_REGISTRY


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


# ---------------------------------------------------------------------------
# Generalized staged runtime (ResNet / Pre-Act ResNet / RegNet / DarkNet)
# ---------------------------------------------------------------------------

@dataclass
class StageSpec:
    block_class: Callable
    num_blocks: int
    out_channels: int
    bottleneck_channels: int
    stride: int
    block_kwargs: dict = field(default_factory=dict)


class GeneralizedStagedNet(Backbone):
    """Shared runtime for any 'stem + sequential stages of residual blocks' net."""

    def __init__(self, stem, stage_specs: List[StageSpec], out_features, freeze_at=0):
        super().__init__()
        self.stem = stem
        self._out_features = out_features
        self.stages_and_names = []
        current_channels = stem.out_channels
        current_stride = stem.stride

        for i, spec in enumerate(stage_specs):
            blocks = []
            in_ch = current_channels
            for b in range(spec.num_blocks):
                blocks.append(spec.block_class(
                    in_channels=in_ch if b == 0 else spec.out_channels,
                    out_channels=spec.out_channels,
                    bottleneck_channels=spec.bottleneck_channels,
                    stride=spec.stride if b == 0 else 1,
                    **spec.block_kwargs,
                ))
            stage = nn.Sequential(*blocks)
            name = f"res{i + 2}"  # res2, res3, res4, res5 ...
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            current_stride *= spec.stride
            current_channels = spec.out_channels
            self._out_feature_strides = getattr(self, "_out_feature_strides", {})
            self._out_feature_channels = getattr(self, "_out_feature_channels", {})
            self._out_feature_strides[name] = current_stride
            self._out_feature_channels[name] = current_channels

        self._freeze(freeze_at)

    def _freeze(self, freeze_at):
        if freeze_at >= 1:
            for p in self.stem.parameters():
                p.requires_grad = False
        for idx, (stage, _) in enumerate(self.stages_and_names, start=2):
            if freeze_at >= idx:
                for p in stage.parameters():
                    p.requires_grad = False

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self._out_feature_channels[name],
                             stride=self._out_feature_strides[name])
            for name in self._out_features
        }


# ---------------------------------------------------------------------------
# Per-architecture config translators
# (numeric width/depth rules are arch-specific -- filled in as stubs)
# ---------------------------------------------------------------------------

def build_resnet_stage_config(cfg, input_shape):
    stem = BasicStem(input_shape.channels, 64, norm=cfg.MODEL.RESNETS.NORM)
    depth_to_blocks = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3],
                        101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}
    num_blocks_per_stage = depth_to_blocks[cfg.MODEL.RESNETS.DEPTH]
    out_channels = [256, 512, 1024, 2048]
    bottleneck_channels = [64, 128, 256, 512]
    stages = [
        StageSpec(BottleneckBlock, n, oc, bc, stride=2 if i > 0 else 1,
                  block_kwargs={"norm": cfg.MODEL.RESNETS.NORM})
        for i, (n, oc, bc) in enumerate(zip(num_blocks_per_stage, out_channels, bottleneck_channels))
    ]
    return stem, stages


def build_preresnet_stage_config(cfg, input_shape):
    stem = BasicStem(input_shape.channels, 64, norm=cfg.MODEL.RESNETS.NORM)
    depth_to_blocks = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3],
                        101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}
    num_blocks_per_stage = depth_to_blocks[cfg.MODEL.RESNETS.DEPTH]
    out_channels = [256, 512, 1024, 2048]
    bottleneck_channels = [64, 128, 256, 512]
    stages = [
        StageSpec(PreActBottleneckBlock, n, oc, bc, stride=2 if i > 0 else 1,
                  block_kwargs={"norm": cfg.MODEL.RESNETS.NORM})
        for i, (n, oc, bc) in enumerate(zip(num_blocks_per_stage, out_channels, bottleneck_channels))
    ]
    return stem, stages


def build_darknet_stage_config(cfg, input_shape):
    stem = DarkStem(input_shape.channels, 32, norm=cfg.MODEL.DARKNET.NORM)
    # DarkNet-53 stage layout
    num_blocks_per_stage = [1, 2, 8, 8, 4]
    out_channels = [64, 128, 256, 512, 1024]
    bottleneck_channels = [32, 64, 128, 256, 512]
    stages = [
        StageSpec(DarkResidualBlock, n, oc, bc, stride=2,
                  block_kwargs={"norm": cfg.MODEL.DARKNET.NORM})
        for n, oc, bc in zip(num_blocks_per_stage, out_channels, bottleneck_channels)
    ]
    return stem, stages


def build_regnet_stage_config(cfg, input_shape):
    """TODO: port the DDS-paper width/depth generator (generate_regnet_full)
    from detectron2/modeling/backbone/regnet.py -- it computes per-stage
    (num_blocks, width, group_width) from (w_a, w_0, w_m, depth, group_width).
    """
    stem = SimpleStem(input_shape.channels, cfg.MODEL.REGNETS.STEM_WIDTH,
                       norm=cfg.MODEL.REGNETS.NORM)
    stages = []  # TODO: fill via DDS width/depth rule, using ResBottleneckBlock
    return stem, stages


STAGE_BUILDERS = {
    "resnet": build_resnet_stage_config,
    "preresnet": build_preresnet_stage_config,
    "regnet": build_regnet_stage_config,
    "darknet": build_darknet_stage_config,
}


# ---------------------------------------------------------------------------
# DLA: separate topology (tree aggregation) -- reuses ConvNormAct only
# ---------------------------------------------------------------------------

class DLABasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, norm="BN"):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, 3, stride=stride, norm=norm)
        self.conv2 = ConvNormAct(out_channels, out_channels, 3, stride=1, norm=norm, activation=None)
        self.shortcut = make_shortcut(in_channels, out_channels, stride, norm)
        self.act_out = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x) if self.shortcut is not None else x
        out = self.conv2(self.conv1(x))
        return self.act_out(out + identity)


class Root(nn.Module):
    """Aggregates multiple child feature maps via concat + conv (DLA-specific)."""

    def __init__(self, in_channels, out_channels, kernel_size=1, norm="BN"):
        super().__init__()
        self.conv = ConvNormAct(in_channels, out_channels, kernel_size, norm=norm)

    def forward(self, *xs):
        return self.conv(torch.cat(xs, dim=1))


class Tree(nn.Module):
    """Recursive tree of blocks + Root aggregation -- core DLA structure."""

    def __init__(self, levels, block, in_channels, out_channels, stride=1, norm="BN"):
        super().__init__()
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, norm=norm)
            self.tree2 = block(out_channels, out_channels, 1, norm=norm)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels, stride, norm=norm)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels, 1, norm=norm)
        self.root = Root(2 * out_channels, out_channels, norm=norm)
        self.downsample = nn.MaxPool2d(stride, stride=stride) if stride > 1 else None
        self.project = make_shortcut(in_channels, out_channels, 1, norm)

    def forward(self, x):
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        x1 = self.tree1(x)
        x2 = self.tree2(x1)
        return self.root(x2, x1, residual) if False else self.root(x2, x1)


class DLA(Backbone):
    """Deep Layer Aggregation -- distinct topology, cannot reuse GeneralizedStagedNet."""

    def __init__(self, cfg, input_shape, out_features):
        super().__init__()
        levels = cfg.MODEL.DLA.LEVELS       # e.g. [1, 1, 1, 2, 2, 1]
        channels = cfg.MODEL.DLA.CHANNELS   # e.g. [16, 32, 64, 128, 256, 512]
        norm = cfg.MODEL.DLA.NORM
        self._out_features = out_features
        self.base = ConvNormAct(input_shape.channels, channels[0], 7, stride=1, norm=norm)
        self.level0 = ConvNormAct(channels[0], channels[1], 3, stride=1, norm=norm)
        self.level_trees = nn.ModuleList()
        in_ch = channels[1]
        self._out_feature_channels, self._out_feature_strides = {}, {}
        stride_acc = 2
        for i, (lv, ch) in enumerate(zip(levels[2:], channels[2:]), start=2):
            tree = Tree(lv, DLABasicBlock, in_ch, ch, stride=2, norm=norm)
            self.level_trees.append(tree)
            in_ch = ch
            stride_acc *= 2
            name = f"level{i}"
            self._out_feature_channels[name] = ch
            self._out_feature_strides[name] = stride_acc

    def forward(self, x):
        outputs = {}
        x = self.level0(self.base(x))
        for i, tree in enumerate(self.level_trees, start=2):
            x = tree(x)
            name = f"level{i}"
            if name in self._out_features:
                outputs[name] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self._out_feature_channels[name],
                             stride=self._out_feature_strides[name])
            for name in self._out_features
        }


def build_dla_backbone(cfg, input_shape):
    return DLA(cfg, input_shape, out_features=cfg.MODEL.BACKBONE.OUT_FEATURES)


# ---------------------------------------------------------------------------
# Single registered entry point, dispatched by cfg.MODEL.BACKBONE.ARCH
# ---------------------------------------------------------------------------

@BACKBONE_REGISTRY.register()
def build_unified_backbone(cfg, input_shape):
    arch = cfg.MODEL.BACKBONE.ARCH
    if arch == "dla":
        return build_dla_backbone(cfg, input_shape)
    if arch not in STAGE_BUILDERS:
        raise ValueError(f"Unknown backbone arch: {arch}")
    stem, stages = STAGE_BUILDERS[arch](cfg, input_shape)
    return GeneralizedStagedNet(
        stem, stages,
        out_features=cfg.MODEL.BACKBONE.OUT_FEATURES,
        freeze_at=cfg.MODEL.BACKBONE.FREEZE_AT,
    )


"""
Example cfg additions (add to your CfgNode defaults):

cfg.MODEL.BACKBONE.NAME = "build_unified_backbone"
cfg.MODEL.BACKBONE.ARCH = "resnet"                # resnet|preresnet|regnet|darknet|dla
cfg.MODEL.BACKBONE.OUT_FEATURES = ["res2","res3","res4","res5"]
cfg.MODEL.BACKBONE.FREEZE_AT = 2

cfg.MODEL.RESNETS.DEPTH = 50
cfg.MODEL.RESNETS.NORM = "FrozenBN"

cfg.MODEL.DARKNET.NORM = "BN"

cfg.MODEL.REGNETS.STEM_WIDTH = 32
cfg.MODEL.REGNETS.NORM = "BN"
# ... plus w_a, w_0, w_m, depth, group_width, se_ratio for the DDS generator

cfg.MODEL.DLA.LEVELS = [1, 1, 1, 2, 2, 1]
cfg.MODEL.DLA.CHANNELS = [16, 32, 64, 128, 256, 512]
cfg.MODEL.DLA.NORM = "BN"
cfg.MODEL.BACKBONE.OUT_FEATURES = ["level3","level4","level5"]  # for dla
"""
