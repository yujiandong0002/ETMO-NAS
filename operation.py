import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .utils import min_divisible_value
import torch.distributed as dist
from torch.autograd.function import Function

def sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end


def copy_bn(target_bn, src_bn):
    feature_dim = (
        target_bn.num_channels
        if isinstance(target_bn, nn.GroupNorm)
        else target_bn.num_features
    )

    target_bn.weight.data.copy_(src_bn.weight.data[:feature_dim])
    target_bn.bias.data.copy_(src_bn.bias.data[:feature_dim])
    if type(src_bn) in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]:
        target_bn.running_mean.data.copy_(src_bn.running_mean.data[:feature_dim])
        target_bn.running_var.data.copy_(src_bn.running_var.data[:feature_dim])


class DynamicLinear(nn.Module):
    def __init__(self, max_in_features, max_out_features, bias=True):
        super(DynamicLinear, self).__init__()

        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.bias = bias

        self.linear = nn.Linear(self.max_in_features, self.max_out_features, self.bias)

        self.active_out_features = self.max_out_features

    def get_active_weight(self, out_features, in_features):
        return self.linear.weight[:out_features, :in_features]

    def get_active_bias(self, out_features):
        return self.linear.bias[:out_features] if self.bias else None

    def forward(self, x, out_features=None):
        if out_features is None:
            out_features = self.active_out_features

        in_features = x.size(1)
        weight = self.get_active_weight(out_features, in_features).contiguous()
        bias = self.get_active_bias(out_features)
        y = F.linear(x, weight, bias)
        return y


class DynamicSeparableConv2d(nn.Module):
    # KERNEL_TRANSFORM_MODE = None  # None or 1
    
    def __init__(self, max_in_channels, kernel_size_list, stride=1, dilation=1, channels_per_group=1):
        super(DynamicSeparableConv2d, self).__init__()
        
        self.max_in_channels = max_in_channels
        self.channels_per_group = channels_per_group
        assert self.max_in_channels % self.channels_per_group == 0
        self.kernel_size_list = kernel_size_list
        self.stride = stride
        self.dilation = dilation
        
        self.conv = nn.Conv3d(
            self.max_in_channels, self.max_in_channels, max(self.kernel_size_list), self.stride,
            groups=self.max_in_channels // self.channels_per_group, bias=False,
        )
        
        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()  # e.g., [3, 5, 7]
        
        self.active_kernel_size = max(self.kernel_size_list)
    
    def get_active_filter(self, in_channel, kernel_size):
        out_channel = in_channel
        max_kernel_size = max(self.kernel_size_list)
        
        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = self.conv.weight[:out_channel, :in_channel, start:end, start:end,start:end]

        return filters
    
    def forward(self, x, kernel_size=None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        in_channel = x.size(1)
        assert in_channel % self.channels_per_group == 0
        
        filters = self.get_active_filter(in_channel, kernel_size).contiguous()
        
        padding = get_same_padding(kernel_size)
        y = F.conv3d(
            x, filters, None, self.stride, padding, self.dilation, in_channel // self.channels_per_group
        )
        return y


class DynamicConv2d(nn.Module):
    def __init__(
        self, max_in_channels, max_out_channels, kernel_size=1, stride=1, dilation=1
    ):
        super(DynamicConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv3d(
            self.max_in_channels,
            self.max_out_channels,
            self.kernel_size,
            stride=self.stride,
            bias=False,
        )

        self.active_out_channel = self.max_out_channels

    def get_active_filter(self, out_channel, in_channel):
        return self.conv.weight[:out_channel, :in_channel, :, :, :]

    def forward(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)
        filters = self.get_active_filter(out_channel, in_channel).contiguous()

        padding = get_same_padding(self.kernel_size)
        filters = (
            self.conv.weight_standardization(filters)
            if isinstance(self.conv, WeightStandardConv2d)
            else filters
        )
        y = F.conv3d(x, filters, None, self.stride, padding, self.dilation, 1)
        return y


class AllReduce(Function):
    @staticmethod
    def forward(ctx, input):
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


class DynamicSE(SEModule):
    def __init__(self, max_channel):
        super(DynamicSE, self).__init__(max_channel)

    def get_active_reduce_weight(self, num_mid, in_channel, groups=None):
        if groups is None or groups == 1:
            return self.fc.reduce.weight[:num_mid, :in_channel, :, :, :]
        else:
            assert in_channel % groups == 0
            sub_in_channels = in_channel // groups
            sub_filters = torch.chunk(
                self.fc.reduce.weight[:num_mid, :, :, :, :], groups, dim=1
            )
            return torch.cat(
                [sub_filter[:, :sub_in_channels, :, :, :] for sub_filter in sub_filters],
                dim=1,
            )

    def get_active_reduce_bias(self, num_mid):
        return (
            self.fc.reduce.bias[:num_mid] if self.fc.reduce.bias is not None else None
        )

    def get_active_expand_weight(self, num_mid, in_channel, groups=None):
        if groups is None or groups == 1:
            return self.fc.expand.weight[:in_channel, :num_mid, :, :, :]
        else:
            assert in_channel % groups == 0
            sub_in_channels = in_channel // groups
            sub_filters = torch.chunk(
                self.fc.expand.weight[:, :num_mid, :, :, :], groups, dim=0
            )
            return torch.cat(
                [sub_filter[:sub_in_channels, :, :, :, :] for sub_filter in sub_filters],
                dim=0,
            )

    def get_active_expand_bias(self, in_channel, groups=None):
        if groups is None or groups == 1:
            return (
                self.fc.expand.bias[:in_channel]
                if self.fc.expand.bias is not None
                else None
            )
        else:
            assert in_channel % groups == 0
            sub_in_channels = in_channel // groups
            sub_bias_list = torch.chunk(self.fc.expand.bias, groups, dim=0)
            return torch.cat(
                [sub_bias[:sub_in_channels] for sub_bias in sub_bias_list], dim=0
            )

    def forward(self, x, groups=None):
        in_channel = x.size(1)
        num_mid = make_divisible(in_channel // self.reduction)

        y = self.avg_pool(x)
        # reduce
        reduce_filter = self.get_active_reduce_weight(
            num_mid, in_channel, groups=groups
        ).contiguous()
        reduce_bias = self.get_active_reduce_bias(num_mid)
        y = F.conv3d(y, reduce_filter, reduce_bias, 1, 0, 1, 1)
        # relu
        y = self.fc.relu(y)
        # expand
        expand_filter = self.get_active_expand_weight(
            num_mid, in_channel, groups=groups
        ).contiguous()
        expand_bias = self.get_active_expand_bias(in_channel, groups=groups)
        y = F.conv3d(y, expand_filter, expand_bias, 1, 0, 1, 1)
        # hard sigmoid
        y = self.fc.h_sigmoid(y)

        return x * y


class DynamicBatchNorm2d(nn.Module):
    
    #SET_RUNNING_STATISTICS = False
    
    def __init__(self, max_feature_dim):
        super(DynamicBatchNorm2d, self).__init__()
        
        self.max_feature_dim = max_feature_dim
        self.bn = nn.BatchNorm3d(self.max_feature_dim)

        # self.exponential_average_factor = 0    # doesn't acculate bn stats
        self.need_sync = False   # sync-batchnormalization, suggested to use in bignasutil

        # reserved to tracking the performance of the largest and smallest network
        self.bn_tracking = nn.ModuleList(
            [
                nn.BatchNorm3d(self.max_feature_dim, affine=False),
                nn.BatchNorm3d(self.max_feature_dim, affine=False)
            ]
        )

    def forward(self, x):
        feature_dim = x.size(1)

        bn = self.bn
        # need_sync

        return F.batch_norm(
                x, bn.running_mean[:feature_dim], bn.running_var[:feature_dim], bn.weight[:feature_dim],
                bn.bias[:feature_dim], bn.training or not bn.track_running_stats,
                bn.momentum, bn.eps,
            )



def make_divisible(v, divisor=8, min_val=None):
    
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


"""Activation."""

def build_activation(act_func, inplace=True):
    if act_func == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_func == "relu6":
        return nn.ReLU6(inplace=inplace)
    elif act_func == "tanh":
        return nn.Tanh()
    elif act_func == "sigmoid":
        return nn.Sigmoid()
    elif act_func == "h_swish":
        return Hswish(inplace=inplace)
    elif act_func == "h_sigmoid":
        return Hsigmoid(inplace=inplace)
    elif act_func == 'swish':
        return MemoryEfficientSwish()
    elif act_func is None or act_func == "none":
        return None
    else:
        raise ValueError("do not support: %s" % act_func)


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)
    

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0

    def __repr__(self):
        return "Hswish()"


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, "invalid kernel size: %s" % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0

    def __repr__(self):
        return "Hsigmoid()"


class WeightStandardConv2d(nn.Conv3d):


    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(WeightStandardConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.WS_EPS = None

    def weight_standardization(self, weight):
        if self.WS_EPS is not None:
            weight_mean = (
                weight.mean(dim=1, keepdim=True)
                .mean(dim=2, keepdim=True)
                .mean(dim=3, keepdim=True)
            )
            weight = weight - weight_mean
            std = (
                weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1)
                + self.WS_EPS
            )
            weight = weight / std.expand_as(weight)
        return weight

    def forward(self, x):
        if self.WS_EPS is None:
            return super(WeightStandardConv2d, self).forward(x)
        else:
            return F.conv2d(
                x,
                self.weight_standardization(self.weight),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

    def __repr__(self):
        return super(WeightStandardConv2d, self).__repr__()[:-1] + ", ws_eps=%s)" % self.WS_EPS
    

# Basic layer to init with Dropout, Conv, BN and activations (order not fixed.)
class MixedDCBRLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_bn=True,
        act_func="relu",
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        super(MixedDCBRLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules["bn"] = nn.BatchNorm3d(in_channels,eps=1e-05,momentum=0.)
            else:
                modules["bn"] = nn.BatchNorm3d(out_channels,eps=1e-05,momentum=0.)
        else:
            modules["bn"] = None
        # activation
        modules["act"] = build_activation(
            self.act_func, self.ops_list[0] != "act" and self.use_bn
        )
        # dropout
        if self.dropout_rate > 0:
            modules["dropout"] = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            modules["dropout"] = None
        # weight
        modules["weight"] = self.weight_op()

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == "weight":
                # dropout before weight operation
                if modules["dropout"] is not None:
                    self.add_module("dropout", modules["dropout"])
                for key in modules["weight"]:
                    self.add_module(key, modules["weight"][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split("_")

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == "bn":
                return True
            elif op == "weight":
                return False
        raise ValueError("Invalid ops_order: %s" % self.ops_order)

    def forward(self, x):
        # similar to nn.Sequential
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def config(self):
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
            "dropout_rate": self.dropout_rate,
            "ops_order": self.ops_order,
        }


class ShuffleLayer(nn.Module):
    def __init__(self, groups):
        super(ShuffleLayer, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        # reshape
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batch_size, -1, height, width)
        return x

    def __repr__(self):
        return "ShuffleLayer(groups=%d)" % self.groups


class ConvLayer(MixedDCBRLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        has_shuffle=False,
        use_se=False,
        use_bn=True,
        act_func="relu",
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        # default normal 3x3_Conv with bn and relu
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle
        self.use_se = use_se

        super(ConvLayer, self).__init__(
            in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order
        )
        if self.use_se:
            self.add_module("se", SEModule(self.out_channels))

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict(
            {
                "conv": nn.Conv3d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=padding,
                    groups=min_divisible_value(self.in_channels, self.groups),
                    bias=self.bias,
                )
            }
        )
        if self.has_shuffle and self.groups > 1:
            weight_dict["shuffle"] = ShuffleLayer(self.groups)

        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.groups == 1:
            if self.dilation > 1:
                conv_str = "%dx%d_DilatedConv" % (kernel_size[0], kernel_size[1])
            else:
                conv_str = "%dx%d_Conv" % (kernel_size[0], kernel_size[1])
        else:
            if self.dilation > 1:
                conv_str = "%dx%d_DilatedGroupConv" % (kernel_size[0], kernel_size[1])
            else:
                conv_str = "%dx%d_GroupConv" % (kernel_size[0], kernel_size[1])
        conv_str += "_O%d" % self.out_channels
        if self.use_se:
            conv_str = "SE_" + conv_str
        conv_str += "_" + self.act_func.upper()
        if self.use_bn:
            if isinstance(self.bn, nn.GroupNorm):
                conv_str += "_GN%d" % self.bn.num_groups
            elif isinstance(self.bn, nn.BatchNorm2d):
                conv_str += "_BN"
        return conv_str

    @property
    def config(self):
        return {
            "name": ConvLayer.__name__,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "groups": self.groups,
            "bias": self.bias,
            "has_shuffle": self.has_shuffle,
            "use_se": self.use_se,
            **super(ConvLayer, self).config,
        }


class SEModule(nn.Module):
    REDUCTION = 4

    def __init__(self, channel, reduction=None):
        super(SEModule, self).__init__()

        self.channel = channel
        self.reduction = SEModule.REDUCTION if reduction is None else reduction
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_mid = make_divisible(self.channel // self.reduction)

        self.fc = nn.Sequential(
            OrderedDict(
                [
                    ("reduce", nn.Conv3d(self.channel, num_mid, 1, 1, 0, bias=True)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("expand", nn.Conv3d(num_mid, self.channel, 1, 1, 0, bias=True)),
                    ("h_sigmoid", Hsigmoid(inplace=True)),
                ]
            )
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

    def __repr__(self):
        return "SE(channel=%d, reduction=%d)" % (self.channel, self.reduction)


def drop_connect(inputs, p, training):

    assert 0 <= p <= 1, 'p must be in range of [0,1]'
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1.0 - p

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1,1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor
    return output


class ZeroLayer(nn.Module):
    def __init__(self):
        super(ZeroLayer, self).__init__()

    def forward(self, x):
        raise ValueError

    @property
    def module_str(self):
        return "Zero"

    @property
    def config(self):
        return {
            "name": ZeroLayer.__name__,
        }


class ResidualBlock(nn.Module):
    def __init__(self, conv, shortcut, drop_connect_rate=0):
        super(ResidualBlock, self).__init__()

        self.conv = conv
        self.mobile_inverted_conv = self.conv   # BigNAS
        self.shortcut = shortcut
        self.drop_connect_rate = drop_connect_rate

    def forward(self, x):
        in_channel = x.size(1)
        if self.conv is None or isinstance(self.conv, ZeroLayer):
            res = x
        elif self.shortcut is None or isinstance(self.shortcut, ZeroLayer):
            res = self.conv(x)
        else:
            im = self.shortcut(x)
            x = self.conv(x)
            if self.drop_connect_rate > 0 and in_channel == im.size(1) and self.shortcut.reduction == 1:
                x = drop_connect(x, p=self.drop_connect_rate, training=self.training)
            res = x + im
        return res

    @property
    def module_str(self):
        return "(%s, %s)" % (
            self.conv.module_str if self.conv is not None else None,
            self.shortcut.module_str if self.shortcut is not None else None,
        )

    @property
    def config(self):
        return {
            "name": ResidualBlock.__name__,
            "conv": self.conv.config if self.conv is not None else None,
            "shortcut": self.shortcut.config if self.shortcut is not None else None,
        }


    @property
    def mobile_inverted_conv(self):
        return self.conv


class MBConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        expand_ratio=6,
        mid_channels=None,
        act_func="relu6",
        use_se=False,
        channels_per_group=1,
    ):
        super(MBConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.act_func = act_func
        self.use_se = use_se
        self.channels_per_group = channels_per_group

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ("conv", nn.Conv3d(self.in_channels, feature_dim, 1, 1, bias=False)),
                ("bn", nn.BatchNorm3d(feature_dim,eps=1e-05, momentum=0.)),
                ("act", build_activation(self.act_func, inplace=True)),
            ]))

        assert feature_dim % self.channels_per_group == 0
        active_groups = feature_dim // self.channels_per_group
        pad = get_same_padding(self.kernel_size)
        
        # assert feature_dim % self.groups == 0
        # active_groups = feature_dim // self.groups
        depth_conv_modules = [
            (
                "conv",
                nn.Conv3d(
                    feature_dim,
                    feature_dim,
                    kernel_size,
                    stride,
                    pad,
                    groups=active_groups,
                    bias=False,
                ),
            ),
            ("bn", nn.BatchNorm3d(feature_dim,eps=1e-05, momentum=0.)),
            ("act", build_activation(self.act_func, inplace=True)),
        ]
        if self.use_se:
            depth_conv_modules.append(("se", SEModule(feature_dim)))
        self.depth_conv = nn.Sequential(OrderedDict(depth_conv_modules))

        self.point_linear = nn.Sequential(
            OrderedDict(
                [
                    ("conv", nn.Conv3d(feature_dim, out_channels, 1, 1, bias=False)),
                    ("bn", nn.BatchNorm3d(out_channels,eps=1e-05, momentum=0.)),
                ]
            )
        )

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @property
    def module_str(self):
        if self.mid_channels is None:
            expand_ratio = self.expand_ratio
        else:
            expand_ratio = self.mid_channels // self.in_channels
        layer_str = "%dx%d_MBConv%d_%s" % (
            self.kernel_size,
            self.kernel_size,
            expand_ratio,
            self.act_func.upper(),
        )
        if self.use_se:
            layer_str = "SE_" + layer_str
        layer_str += "_O%d" % self.out_channels
        if self.channels_per_group is not None:
            layer_str += "_G%d" % self.channels_per_group
        if isinstance(self.point_linear.bn, nn.GroupNorm):
            layer_str += "_GN%d" % self.point_linear.bn.num_groups
        elif isinstance(self.point_linear.bn, nn.BatchNorm3d):
            layer_str += "_BN"

        return layer_str

    @property
    def config(self):
        return {
            "name": MBConvLayer.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "expand_ratio": self.expand_ratio,
            "mid_channels": self.mid_channels,
            "act_func": self.act_func,
            "use_se": self.use_se,
            "channeles_per_group": self.channels_per_group,
        }


class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        use_bn=False,
        act_func=None,
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        super(LinearLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ modules """
        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules["bn"] = nn.BatchNorm1d(in_features)
            else:
                modules["bn"] = nn.BatchNorm1d(out_features)
        else:
            modules["bn"] = None
        # activation
        modules["act"] = build_activation(self.act_func, self.ops_list[0] != "act")
        # dropout
        if self.dropout_rate > 0:
            modules["dropout"] = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            modules["dropout"] = None
        # linear
        modules["weight"] = {
            "linear": nn.Linear(self.in_features, self.out_features, self.bias)
        }

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == "weight":
                if modules["dropout"] is not None:
                    self.add_module("dropout", modules["dropout"])
                for key in modules["weight"]:
                    self.add_module(key, modules["weight"][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split("_")

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == "bn":
                return True
            elif op == "weight":
                return False
        raise ValueError("Invalid ops_order: %s" % self.ops_order)

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def module_str(self):
        return "%dx%d_Linear" % (self.in_features, self.out_features)

    @property
    def config(self):
        return {
            "name": LinearLayer.__name__,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
            "dropout_rate": self.dropout_rate,
            "ops_order": self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        return LinearLayer(**config)


class ShortcutLayer(nn.Module):

    
    def __init__(self, in_channels, out_channels, reduction=1):
        super(ShortcutLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction = reduction

        self.conv = nn.Conv3d(in_channels, out_channels, 1, 1, bias=False)

    def forward(self, x):
        if self.reduction > 1:
            padding = 0 if x.size(-1) % 2 == 0 else 1
            x = F.avg_pool3d(x, (1,self.reduction,self.reduction), padding=(0,padding,padding))
        if self.in_channels != self.out_channels:
            x = self.conv(x)
        return x

    @property
    def module_str(self):
        if self.in_channels == self.out_channels and self.reduction == 1:
            conv_str = 'IdentityShortcut'
        else:
            if self.reduction == 1:
                conv_str = '%d-%d_Shortcut' % (self.in_channels, self.out_channels)
            else:
                conv_str = '%d-%d_R%d_Shortcut' % (self.in_channels, self.out_channels, self.reduction)
        return conv_str

    @property
    def config(self):
        return {
            'name': ShortcutLayer.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'reduction': self.reduction,
        }

    @staticmethod
    def build_from_config(config):
        return ShortcutLayer(**config)


    

