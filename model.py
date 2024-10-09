
import os
import re
import math
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from copy import deepcopy
import torch.nn.functional as F
from utils import val2list
from collections import OrderedDict
from operation import ResidualBlock
from layers import DynamicConvLayer, DynamicMBConvLayer, DynamicShortcutLayer, DynamicLinearLayer
class CTIPETPlusDataset(Dataset):
    def __init__(self, ct_path, pet_path, petct_path, transform=None, step='train'):
        self.ct_path = ct_path
        self.pet_path = pet_path
        self.petct_path = petct_path

        self.transform = transform
        self.transformNone = transforms.Compose([transforms.ToTensor()])

        self.ctimage_slice = []
        self.petimage_slice = []
        self.petctimage_slice = []
        self.labels_ct = {}
        self.labels_pet = {}
        self.labels_petct = {}
        self.step = step
        for negposlab in sorted(os.listdir(self.ct_path)):
            lab = negposlab[-1]
            if lab == '0':
                negpath = os.path.join(self.ct_path, negposlab)
                for negobject in sorted(os.listdir(negpath)):
                    negopath = os.path.join(self.ct_path, negposlab, negobject)
                    self.ctimage_slice.append(negopath)
                    self.labels_ct[negopath] = 0
            elif lab == '1':
                pospath = os.path.join(self.ct_path, negposlab)
                for posobject in sorted(os.listdir(pospath)):
                    posopath = os.path.join(self.ct_path, negposlab, posobject)
                    self.ctimage_slice.append(posopath)
                    self.labels_ct[posopath] = 1

        for negposlab2 in sorted(os.listdir(self.pet_path)):
            lab2 = negposlab2[-1]
            if lab2 == '0':
                negpath2 = os.path.join(self.pet_path, negposlab2)
                for negobject2 in sorted(os.listdir(negpath2)):
                    negopath2 = os.path.join(self.pet_path, negposlab2, negobject2)
                    self.petimage_slice.append(negopath2)
                    self.labels_pet[negopath2] = 0
            elif lab2 == '1':
                pospath2 = os.path.join(self.pet_path, negposlab2)
                for posobject2 in sorted(os.listdir(pospath2)):
                    posopath2 = os.path.join(self.pet_path, negposlab2, posobject2)
                    self.petimage_slice.append(posopath2)
                    self.labels_pet[posopath2] = 1

        for negposlab3 in sorted(os.listdir(self.petct_path)):
            lab3 = negposlab3[-1]
            if lab3 == '0':
                negpath3 = os.path.join(self.petct_path, negposlab3)
                for negobject3 in sorted(os.listdir(negpath3)):
                    negopath3 = os.path.join(self.petct_path, negposlab3, negobject3)
                    self.petctimage_slice.append(negopath3)
                    self.labels_petct[negopath3] = 0
            elif lab3 == '1':
                pospath3 = os.path.join(self.petct_path, negposlab3)
                for posobject3 in sorted(os.listdir(pospath3)):
                    posopath3 = os.path.join(self.petct_path, negposlab3, posobject3)
                    self.petctimage_slice.append(posopath3)
                    self.labels_petct[posopath3] = 1

    def __len__(self):
        return len(self.ctimage_slice)

    # 需要第idx个用户
    def __getitem__(self, idx):
        obj_path = self.ctimage_slice[idx]
        obj = []
        image_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for imame in sorted(os.listdir(obj_path)):
            im_so = re.findall(r'.*li(.*)_', imame)[0]
            image_list[int(im_so) - 1] = imame
        # print(image_list)
        # print(image_list)
        for imameg in image_list:
            # print(imameg)
            image1 = np.load(os.path.join(obj_path, imameg))
            if self.transform:
                imaget = self.transform(image1)
                # print(imaget.shape)
            else:
                imaget = self.transformNone(image1)
                # print(imaget.shape)
            obj.append(imaget)

        data = torch.stack(obj)
        data = data.transpose(0, 1)
        # print(data.shape)
        label = self.labels_ct[obj_path]
        # print(label)

        obj_path_pet = self.petimage_slice[idx]
        obj_pet = []
        image_list_pet = [0, 0, 0, 0, 0, 0, 0]
        for imame_pet in sorted(os.listdir(obj_path_pet)):
            im_so_pet = re.findall(r'.*li(.*)_', imame_pet)[0]
            image_list_pet[int(im_so_pet) - 1] = imame_pet
        # print(image_list)
        # print(image_list)
        for imameg_pet in image_list_pet:
            # print(imameg)
            image1_pet = np.load(os.path.join(obj_path_pet, imameg_pet))
            if self.transform:
                imaget_pet = self.transform(image1_pet)
                # print(imaget.shape)
            else:
                imaget_pet = self.transformNone(image1_pet)
                # print(imaget.shape)
            obj_pet.append(imaget_pet)

        data_pet = torch.stack(obj_pet)
        data_pet = data_pet.transpose(0, 1)
        # print(data.shape)
        label_pet = self.labels_pet[obj_path_pet]
        # print(label)

        obj_path_petct = self.petctimage_slice[idx]
        obj_petct = []
        image_list_petct = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for imame_petct in sorted(os.listdir(obj_path_petct)):
            im_so_petct = re.findall(r'.*li(.*)_', imame_petct)[0]
            image_list_petct[int(im_so_petct) - 1] = imame_petct
        # print(image_list)
        # print(image_list)
        for imameg_petct in image_list_petct:
            # print(imameg)
            image1_petct = np.load(os.path.join(obj_path_petct, imameg_petct))
            if self.transform:
                imaget_petct = self.transform(image1_petct)
                # print(imaget.shape)
            else:
                imaget_petct = self.transformNone(image1_petct)
                # print(imaget.shape)
            obj_petct.append(imaget_petct)

        data_petct = torch.stack(obj_petct)
        data_petct = data_petct.transpose(0, 1)
        # print(data.shape)
        label_petct = self.labels_petct[obj_path_petct]
        # print(label)
        if label == label_pet and label == label_petct and label_petct == label_pet:
            return data, data_pet, data_petct, label
        else:
            print("error")
class CELossSoft(torch.nn.modules.loss._Loss):

    def forward(self, output, soft_logits, temperature=1.):
        output, soft_logits = output / temperature, soft_logits / temperature
        soft_target_prob = soft_logits  #如果不是概率分布，转换为概率分布
        output_log_prob = F.log_softmax(output, dim=1)
        kd_loss = -torch.sum(soft_target_prob * output_log_prob, dim=1)
        loss = kd_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

def _label_smooth(target, n_classes: int, label_smoothing):
    # convert to one-hot
    batch_size = target.size(0)
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros((batch_size, n_classes), device=target.device)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return soft_target

def CrossEntropyLoss_label_smoothed(pred, target):
    label_smoothing = 0.2
    soft_target = _label_smooth(target, pred.size(1), label_smoothing)
    ce_criterion = nn.CrossEntropyLoss()
    ads = torch.argmax(soft_target, dim=1)
    loss = ce_criterion(pred, torch.argmax(soft_target, dim=1))
    return loss
class TripleETMONASModel(nn.Module):

    def __init__(self, first_conv_one, first_conv_two, first_conv_three, blocks_one, blocks_two, blocks_three,
                 last_conv_one, last_conv_two, last_conv_three, classifier, resolution, use_v3_head=True):
        super(TripleETMONASModel, self).__init__()

        self.first_conv_one = first_conv_one
        self.first_conv_two = first_conv_two
        self.first_conv_three = first_conv_three
        self.blocks_one = nn.ModuleList(blocks_one)
        self.blocks_two = nn.ModuleList(blocks_two)
        self.blocks_three = nn.ModuleList(blocks_three)
        self.last_conv_one = last_conv_one
        self.last_conv_two = last_conv_two
        self.last_conv_three = last_conv_three
        self.classifier = classifier

        self.resolution = resolution  # input size
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.use_v3_head = use_v3_head
        self.set_bn_param(0.98, 1e-05)
    def forward(self, x, x2, x3):
        # resize input to target resolution first
        # Rule: transform images into different sizes

        x = self.first_conv_one(x)
        x2 = self.first_conv_two(x2)
        x3 = self.first_conv_three(x3)

        for block in self.blocks_one:
            x = block(x)
        for block in self.blocks_two:
            x2 = block(x2)
        for block in self.blocks_three:
            x3 = block(x3)
        x = self.last_conv_one(x)
        x2 = self.last_conv_two(x2)
        x3 = self.last_conv_three(x3)
        if not self.use_v3_head:
            x = self.avg_pool(x)  # global average pooling
        if not self.use_v3_head:
            x2 = self.avg_pool(x2)  # global average pooling
        if not self.use_v3_head:
            x3 = self.avg_pool(x3)  # global average pooling
        x = x.view(x.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        out = torch.cat((x, x2, x2, x3), 1)
        out = self.classifier(out)
        out = torch.softmax(out, dim=1)
        # return x
        return out
    def get_parameters(self, keys=None, mode="include"):
        if keys is None:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    yield param
        elif mode == "include":
            for name, param in self.named_parameters():
                flag = False
                for key in keys:
                    if key in name:
                        flag = True
                        break
                if flag and param.requires_grad:
                    yield param
        elif mode == "exclude":
            for name, param in self.named_parameters():
                flag = True
                for key in keys:
                    if key in name:
                        flag = False
                        break
                if flag and param.requires_grad:
                    yield param
        else:
            raise ValueError("do not support: %s" % mode)
    def set_bn_param(self, momentum, eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.SyncBatchNorm):
                if momentum is not None:
                    m.momentum = float(momentum)
                else:
                    m.momentum = None
                m.eps = float(eps)
        return

    def get_bn_param(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.SyncBatchNorm):
                return {
                    'momentum': m.momentum,
                    'eps': m.eps,
                }
        return None

    def reset_running_stats_for_calibration(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.SyncBatchNorm):
                m.training = True
                m.momentum = None  # cumulative moving average
                m.reset_running_stats()

class TripleETMONASOperationModel(nn.Module):

    def __init__(self, supernet_cfg, n_classes=2, bn_param=(0.98, 1e-5)):
        super(TripleETMONASOperationModel, self).__init__()

        self.supernet_cfg = supernet_cfg
        self.n_classes = n_classes
        self.use_v3_head = getattr(self.supernet_cfg, 'use_v3_head', False)
        self.stage_names = ['first_conv', 'mb1', 'mb2', 'mb3', 'mb4', 'mb5', 'mb6', 'mb7', 'last_conv']

        self.width_list, self.depth_list, self.ks_list, self.expand_ratio_list = [], [], [], []
        for name in self.stage_names:
            block_cfg = getattr(self.supernet_cfg, name)
            self.width_list.append(block_cfg.c)
            if name.startswith('mb'):
                self.depth_list.append(block_cfg.d)
                self.ks_list.append(block_cfg.k)
                self.expand_ratio_list.append(block_cfg.t)
        self.resolution_list = self.supernet_cfg.resolutions

        self.cfg_candidates = {
            'resolution': self.resolution_list,
            'width': self.width_list,
            'depth': self.depth_list,
            'kernel_size': self.ks_list,
            'expand_ratio': self.expand_ratio_list
        }

        # first conv layer, including conv, bn, act
        out_channel_list, act_func, stride = \
            self.supernet_cfg.first_conv.c, self.supernet_cfg.first_conv.act_func, self.supernet_cfg.first_conv.s
        self.first_conv_one = DynamicConvLayer(
            in_channel_list=val2list(1), out_channel_list=out_channel_list,
            kernel_size=3, stride=stride, act_func=act_func,
        )

        self.first_conv_two = DynamicConvLayer(
            in_channel_list=val2list(1), out_channel_list=out_channel_list,
            kernel_size=3, stride=stride, act_func=act_func,
        )
        self.first_conv_three = DynamicConvLayer(
            in_channel_list=val2list(1), out_channel_list=out_channel_list,
            kernel_size=3, stride=stride, act_func=act_func,
        )
        # inverted residual blocks
        self.block_group_info = []
        blocks = []
        _block_index = 0
        feature_dim = out_channel_list
        for stage_id, key in enumerate(self.stage_names[1:-1]):
            block_cfg = getattr(self.supernet_cfg, key)
            width = block_cfg.c
            n_block = max(block_cfg.d)
            act_func = block_cfg.act_func
            ks = block_cfg.k
            expand_ratio_list = block_cfg.t
            use_se = block_cfg.se

            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                stride = block_cfg.s if i == 0 else '(1,1,1)'
                if min(expand_ratio_list) >= 4:
                    expand_ratio_list = [_s for _s in expand_ratio_list if _s >= 4] if i == 0 else expand_ratio_list
                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=feature_dim,
                    out_channel_list=output_channel,
                    kernel_size_list=ks,
                    expand_ratio_list=expand_ratio_list,
                    stride=stride,
                    act_func=act_func,
                    use_se=use_se,
                    channels_per_group=getattr(self.supernet_cfg, 'channels_per_group', 1)
                )
                # Rule: add skip-connect, and use 2x2 AvgPool or 1x1 Conv for adaptation
                shortcut = DynamicShortcutLayer(feature_dim, output_channel, reduction=max(list(eval(stride))))
                blocks.append(ResidualBlock(mobile_inverted_conv, shortcut))
                feature_dim = output_channel

        self.blocks_one = nn.ModuleList(blocks)
        self.block_group_info = []
        blocksss = []
        _block_index = 0
        feature_dim = out_channel_list
        for stage_id, key in enumerate(self.stage_names[1:-1]):
            block_cfg = getattr(self.supernet_cfg, key)
            width = block_cfg.c
            n_block = max(block_cfg.d)
            act_func = block_cfg.act_func
            ks = block_cfg.k
            expand_ratio_list = block_cfg.t
            use_se = block_cfg.se

            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                stride = block_cfg.s if i == 0 else '(1,1,1)'
                if min(expand_ratio_list) >= 4:
                    expand_ratio_list = [_s for _s in expand_ratio_list if _s >= 4] if i == 0 else expand_ratio_list
                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=feature_dim,
                    out_channel_list=output_channel,
                    kernel_size_list=ks,
                    expand_ratio_list=expand_ratio_list,
                    stride=stride,
                    act_func=act_func,
                    use_se=use_se,
                    channels_per_group=getattr(self.supernet_cfg, 'channels_per_group', 1)
                )
                # Rule: add skip-connect, and use 2x2 AvgPool or 1x1 Conv for adaptation
                shortcut = DynamicShortcutLayer(feature_dim, output_channel, reduction=max(list(eval(stride))))
                blocksss.append(ResidualBlock(mobile_inverted_conv, shortcut))
                feature_dim = output_channel
        self.blocks_two = nn.ModuleList(blocksss)
        self.block_group_info = []
        cblocksss = []
        _block_index = 0
        feature_dim = out_channel_list
        for stage_id, key in enumerate(self.stage_names[1:-1]):
            block_cfg = getattr(self.supernet_cfg, key)
            width = block_cfg.c
            n_block = max(block_cfg.d)
            act_func = block_cfg.act_func
            ks = block_cfg.k
            expand_ratio_list = block_cfg.t
            use_se = block_cfg.se

            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                stride = block_cfg.s if i == 0 else '(1,1,1)'
                if min(expand_ratio_list) >= 4:
                    expand_ratio_list = [_s for _s in expand_ratio_list if _s >= 4] if i == 0 else expand_ratio_list
                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=feature_dim,
                    out_channel_list=output_channel,
                    kernel_size_list=ks,
                    expand_ratio_list=expand_ratio_list,
                    stride=stride,
                    act_func=act_func,
                    use_se=use_se,
                    channels_per_group=getattr(self.supernet_cfg, 'channels_per_group', 1)
                )
                # Rule: add skip-connect, and use 2x2 AvgPool or 1x1 Conv for adaptation
                shortcut = DynamicShortcutLayer(feature_dim, output_channel, reduction=max(list(eval(stride))))
                cblocksss.append(ResidualBlock(mobile_inverted_conv, shortcut))
                feature_dim = output_channel
        self.blocks_three = nn.ModuleList(cblocksss)
        last_channel, act_func = self.supernet_cfg.last_conv.c, self.supernet_cfg.last_conv.act_func
        if not self.use_v3_head:
            self.last_conv_one = DynamicConvLayer(
                in_channel_list=feature_dim, out_channel_list=last_channel,
                kernel_size=1, act_func=act_func, stride='(1,1,1)'
            )
            self.last_conv_two = DynamicConvLayer(
                in_channel_list=feature_dim, out_channel_list=last_channel,
                kernel_size=1, act_func=act_func, stride='(1,1,1)'
            )
            self.last_conv_three = DynamicConvLayer(
                in_channel_list=feature_dim, out_channel_list=last_channel,
                kernel_size=1, act_func=act_func, stride='(1,1,1)'
            )
        else:
            expand_feature_dim = [f_dim * 6 for f_dim in feature_dim]
            self.last_conv_one = nn.Sequential(OrderedDict([
                ('final_expand_layer', DynamicConvLayer(
                    feature_dim, expand_feature_dim, kernel_size=1, use_bn=True, act_func=act_func)
                 ),
                ('pool', nn.AdaptiveAvgPool3d((1, 1, 1))),
                ('feature_mix_layer', DynamicConvLayer(
                    in_channel_list=expand_feature_dim, out_channel_list=last_channel,
                    kernel_size=1, act_func=act_func, use_bn=False, )
                 ),
            ]))
            self.last_conv_two = nn.Sequential(OrderedDict([
                ('final_expand_layer', DynamicConvLayer(
                    feature_dim, expand_feature_dim, kernel_size=1, use_bn=True, act_func=act_func)
                 ),
                ('pool', nn.AdaptiveAvgPool3d((1, 1, 1))),
                ('feature_mix_layer', DynamicConvLayer(
                    in_channel_list=expand_feature_dim, out_channel_list=last_channel,
                    kernel_size=1, act_func=act_func, use_bn=False, )
                 ),
            ]))
            self.last_conv_three = nn.Sequential(OrderedDict([
                ('final_expand_layer', DynamicConvLayer(
                    feature_dim, expand_feature_dim, kernel_size=1, use_bn=True, act_func=act_func)
                 ),
                ('pool', nn.AdaptiveAvgPool3d((1, 1, 1))),
                ('feature_mix_layer', DynamicConvLayer(
                    in_channel_list=expand_feature_dim, out_channel_list=last_channel,
                    kernel_size=1, act_func=act_func, use_bn=False, )
                 ),
            ]))


        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # final conv layer
        self.classifier = DynamicLinearLayer(
            in_features_list=[a * 4 for a in last_channel], out_features=n_classes, bias=True
        )
        self.init_model()

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        # runtime_depth
        self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]

        self.zero_residual_block_bn_weights()

        self.active_dropout_rate = 0
        self.active_drop_connect_rate = 0

    # Rule: Initialize learnable coefficient \gamma=0
    def zero_residual_block_bn_weights(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, ResidualBlock):
                    if isinstance(m.mobile_inverted_conv, DynamicMBConvLayer) and m.shortcut is not None:
                        m.mobile_inverted_conv.point_linear.bn.bn.weight.zero_()

    def init_model(self, model_init="he_fout"):
        """ Conv2d, BatchNorm2d, BatchNorm1d, Linear, """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.in_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else:
                    raise NotImplementedError
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm3d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, x2, x3):

        # first conv
        x = self.first_conv_one(x)
        x2 = self.first_conv_two(x2)
        x3 = self.first_conv_three(x3)
        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                x = self.blocks_one[idx](x)
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                x2 = self.blocks_two[idx](x2)
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                x3 = self.blocks_three[idx](x3)
        x = self.last_conv_one(x)
        x2 = self.last_conv_two(x2)
        x3 = self.last_conv_three(x3)
        x = self.avg_pool(x)  # global average pooling
        x2 = self.avg_pool(x2)  # global average pooling
        x3 = self.avg_pool(x3)  # global average pooling
        x = x.view(x.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        out = torch.cat((x, x2, x2, x3), dim=1)
        if self.active_dropout_rate > 0 and self.training:
            out = F.dropout(out, p=self.active_dropout_rate)

        out = self.classifier(out)
        out = torch.softmax(out, dim=1)

        return out

    def get_parameters(self, keys=None, mode="include"):
        if keys is None:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    yield param
        elif mode == "include":
            for name, param in self.named_parameters():
                flag = False
                for key in keys:
                    if key in name:
                        flag = True
                        break
                if flag and param.requires_grad:
                    yield param
        elif mode == "exclude":
            for name, param in self.named_parameters():
                flag = True
                for key in keys:
                    if key in name:
                        flag = False
                        break
                if flag and param.requires_grad:
                    yield param
        else:
            raise ValueError("do not support: %s" % mode)

    def set_bn_param(self, momentum, eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.SyncBatchNorm):
                if momentum is not None:
                    m.momentum = float(momentum)
                else:
                    m.momentum = None
                m.eps = float(eps)
        return

    def get_bn_param(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m,nn.BatchNorm1d) or isinstance(m, nn.SyncBatchNorm):
                return {
                    'momentum': m.momentum,
                    'eps': m.eps,
                }
        return None

    """ set, sample and get active sub-networks """

    def set_active_subnet(self, resolution=350, width=None, depth=None, kernel_size=None, expand_ratio=None,
                                resolution2=350, width2=None, depth2=None, kernel_size2=None, expand_ratio2=None,
                                resolution3=350, width3=None, depth3=None, kernel_size3=None, expand_ratio3=None,):
        assert len(depth) == len(kernel_size) == len(expand_ratio) == len(width) - 2
        assert len(depth2) == len(kernel_size2) == len(expand_ratio2) == len(width2) - 2
        assert len(depth3) == len(kernel_size3) == len(expand_ratio3) == len(width3) - 2
        # set resolution
        self.active_resolution = resolution

        # first conv
        self.first_conv_one.active_out_channel = width[0]
        self.first_conv_two.active_out_channel = width2[0]
        self.first_conv_three.active_out_channel = width3[0]

        for stage_id, (c, k, e, d) in enumerate(zip(width[1:-1], kernel_size, expand_ratio, depth)):
            start_idx, end_idx = min(self.block_group_info[stage_id]), max(self.block_group_info[stage_id])
            for block_id in range(start_idx, start_idx + d):
                block = self.blocks_one[block_id]
                # block output channels
                block.mobile_inverted_conv.active_out_channel = c
                if block.shortcut is not None:
                    block.shortcut.active_out_channel = c
                # dw kernel size
                block.mobile_inverted_conv.active_kernel_size = k
                # dw expansion ration
                block.mobile_inverted_conv.active_expand_ratio = e

        for stage_id, (c, k, e, d) in enumerate(zip(width2[1:-1], kernel_size2, expand_ratio2, depth2)):
            start_idx, end_idx = min(self.block_group_info[stage_id]), max(self.block_group_info[stage_id])
            for block_id in range(start_idx, start_idx + d):
                block = self.blocks_two[block_id]
                # block output channels
                block.mobile_inverted_conv.active_out_channel = c
                if block.shortcut is not None:
                    block.shortcut.active_out_channel = c
                # dw kernel size
                block.mobile_inverted_conv.active_kernel_size = k
                # dw expansion ration
                block.mobile_inverted_conv.active_expand_ratio = e
        for stage_id, (c, k, e, d) in enumerate(zip(width3[1:-1], kernel_size3, expand_ratio3, depth3)):
            start_idx, end_idx = min(self.block_group_info[stage_id]), max(self.block_group_info[stage_id])
            for block_id in range(start_idx, start_idx + d):
                block = self.blocks_three[block_id]
                # block output channels
                block.mobile_inverted_conv.active_out_channel = c
                if block.shortcut is not None:
                    block.shortcut.active_out_channel = c
                # dw kernel size
                block.mobile_inverted_conv.active_kernel_size = k
                # dw expansion ration
                block.mobile_inverted_conv.active_expand_ratio = e
        # IRBlocks repated times
        for i, d in enumerate(depth):
            self.runtime_depth[i] = min(len(self.block_group_info[i]), d)

        # last conv
        if not self.use_v3_head:
            self.last_conv_one.active_out_channel = width[-1]
            self.last_conv_two.active_out_channel = width2[-1]
            self.last_conv_three.active_out_channel = width3[-1]
        else:
            # default expansion ratio: 6
            self.last_conv_one.final_expand_layer.active_out_channel = width[-2]
            self.last_conv_two.final_expand_layer.active_out_channel = width2[-2]
            self.last_conv_three.final_expand_layer.active_out_channel = width3[-2]
            self.last_conv_one.feature_mix_layer.active_out_channel = width[-1]
            self.last_conv_two.feature_mix_layer.active_out_channel = width2[-1]
            self.last_conv_three.feature_mix_layer.active_out_channel = width3[-1]

    def set_dropout_rate(self, dropout=0, drop_connect=0, drop_connect_only_last_two_stages=True):
        self.active_dropout_rate = dropout
        for idx, block in enumerate(self.blocks_one):
            if drop_connect_only_last_two_stages:
                if idx not in self.block_group_info[-1] + self.block_group_info[-2]:
                    continue
            x = drop_connect
            y = float(idx)
            this_drop_connect_rate = 7.021665038426933e-08 * x ** 2 - 2.281831732809206e-11 * y ** 2 + 2.432029025488732e-09 * x * y + 0.99999992444491 * x + 0.002857142540689959 * y - 0.04285712885724084
            block.drop_connect_rate = this_drop_connect_rate
        for idx, block in enumerate(self.blocks_two):
            if drop_connect_only_last_two_stages:
                if idx not in self.block_group_info[-1] + self.block_group_info[-2]:
                    continue
            x = drop_connect
            y = float(idx)
            this_drop_connect_rate = 7.021665038426933e-08 * x ** 2 - 2.281831732809206e-11 * y ** 2 + 2.432029025488732e-09 * x * y + 0.99999992444491 * x + 0.002857142540689959 * y - 0.04285712885724084
            block.drop_connect_rate = this_drop_connect_rate
        for idx, block in enumerate(self.blocks_three):
            if drop_connect_only_last_two_stages:
                if idx not in self.block_group_info[-1] + self.block_group_info[-2]:
                    continue
            x = drop_connect
            y = float(idx)
            this_drop_connect_rate = 7.021665038426933e-08 * x ** 2 - 2.281831732809206e-11 * y ** 2 + 2.432029025488732e-09 * x * y + 0.99999992444491 * x + 0.002857142540689959 * y - 0.04285712885724084
            block.drop_connect_rate = this_drop_connect_rate
    def sample_min_subnet(self):
        return self._sample_active_subnet(min_net=True, subnet_seed=0)

    def sample_max_subnet(self):
        return self._sample_active_subnet(max_net=True, subnet_seed=0)

    def sample_active_subnet(self, subnet_seed, compute_flops=False):
        cfg = self._sample_active_subnet(False, False, subnet_seed)
        return cfg

    def sample_active_subnet_within_range(self, targeted_min_flops, targeted_max_flops):
        while True:
            cfg = self._sample_active_subnet()
            cfg['flops'] = self.compute_active_subnet_flops()
            if cfg['flops'] >= targeted_min_flops and cfg['flops'] <= targeted_max_flops:
                return cfg

    def _sample_active_subnet(self, min_net=False, max_net=False, subnet_seed=0):
        def sample_cfg(candidates, sample_min, sample_max):
            if sample_min:
                return min(candidates)
            elif sample_max:
                return max(candidates)
            else:
                return random.choice(candidates)
        random.seed(subnet_seed)
        cfg = {}
        # sample a resolution
        cfg['resolution'] = sample_cfg(self.cfg_candidates['resolution'], min_net, max_net)
        for k in ['width', 'depth', 'kernel_size', 'expand_ratio']:
            cfg[k] = []
            for vv in self.cfg_candidates[k]:
                cfg[k].append(sample_cfg(val2list(vv), min_net, max_net))
        random.seed(subnet_seed+2)
        cfg2 = {}
        # sample a resolution
        cfg2['resolution'] = sample_cfg(self.cfg_candidates['resolution'], min_net, max_net)
        for k in ['width', 'depth', 'kernel_size', 'expand_ratio']:
            cfg2[k] = []
            for vv in self.cfg_candidates[k]:
                cfg2[k].append(sample_cfg(val2list(vv), min_net, max_net))
        random.seed(subnet_seed+3)
        cfg3 = {}
        # sample a resolution
        cfg3['resolution'] = sample_cfg(self.cfg_candidates['resolution'], min_net, max_net)
        for k in ['width', 'depth', 'kernel_size', 'expand_ratio']:
            cfg3[k] = []
            for vv in self.cfg_candidates[k]:
                cfg3[k].append(sample_cfg(val2list(vv), min_net, max_net))

        self.set_active_subnet(cfg['resolution'], cfg['width'], cfg['depth'], cfg['kernel_size'], cfg['expand_ratio'],
                               cfg2['resolution'], cfg2['width'], cfg2['depth'], cfg2['kernel_size'], cfg2['expand_ratio'],
                               cfg3['resolution'], cfg3['width'], cfg3['depth'], cfg3['kernel_size'], cfg3['expand_ratio'])
        return cfg

    def sample_n_subnet(self, segop, n):
        if n == 4:
            self._sample_n_subnet(segop=segop, n=1)
            supsubnet1 = self.get_active_subnet()
            self._sample_n_subnet(segop=segop, n=2)
            supsubnet2 = self.get_active_subnet()
            self._sample_n_subnet(segop=segop, n=3)
            supsubnet3 = self.get_active_subnet()
            self._sample_n_subnet(segop=segop, n=4)
            supsubnet4 = self.get_active_subnet()
            return {'supsubnet1':supsubnet1, 'supsubnet2':supsubnet2, 'supsubnet3':supsubnet3, 'supsubnet4':supsubnet4}

    def _sample_n_subnet(self, segop, n):
        def sample_cfg(candidates, sample_min, sample_max):
            if sample_min:
                return min(candidates)
            elif sample_max:
                return max(candidates)
            else:
                return random.choice(candidates)
        cfg = {}
        # sample a resolution
        cfg['resolution'] = sample_cfg(self.cfg_candidates['resolution'], False, True)
        for k in ['width', 'depth', 'kernel_size', 'expand_ratio']:
            cfg[k] = []
            for vv in self.cfg_candidates[k]:
                cfg[k].append(sample_cfg(val2list(vv),  False, True))

        if segop == 'mb1':
            if n == 1:
                cfg['width'][1] = 16
                cfg['depth'][0] = 1
            elif n == 2:
                cfg['width'][1] = 16
                cfg['depth'][0] = 2
            elif n == 3:
                cfg['width'][1] = 24
                cfg['depth'][0] = 1
            elif n == 4:
                cfg['width'][1] = 24
                cfg['depth'][0] = 2
        self.set_active_subnet(
            cfg['resolution'], cfg['width'], cfg['depth'], cfg['kernel_size'], cfg['expand_ratio'],
            cfg['resolution'], cfg['width'], cfg['depth'], cfg['kernel_size'], cfg['expand_ratio'],
            cfg['resolution'], cfg['width'], cfg['depth'], cfg['kernel_size'], cfg['expand_ratio']
        )
        return cfg

    def get_active_subnet(self, preserve_weight=True):
        with torch.no_grad():
            first_conv_one = self.first_conv_one.get_active_subnet(1, preserve_weight)
            first_conv_two = self.first_conv_two.get_active_subnet(1, preserve_weight)
            first_conv_three = self.first_conv_three.get_active_subnet(1, preserve_weight)

            blocks_one = []
            input_channel_one = first_conv_one.out_channels
            # blocks
            for stage_id, block_idx in enumerate(self.block_group_info):
                depth = self.runtime_depth[stage_id]
                active_idx = block_idx[:depth]
                stage_blocks = []
                for idx in active_idx:
                    stage_blocks.append(ResidualBlock(
                        self.blocks_one[idx].mobile_inverted_conv.get_active_subnet(input_channel_one, preserve_weight),
                        self.blocks_one[idx].shortcut.get_active_subnet(input_channel_one, preserve_weight) if self.blocks_one[
                                                                                                           idx].shortcut is not None else None
                    ))
                    input_channel_one = stage_blocks[-1].mobile_inverted_conv.out_channels
                blocks_one += stage_blocks

            if not self.use_v3_head:
                last_conv_one = self.last_conv_one.get_active_subnet(input_channel_one, preserve_weight)
                in_features_one = last_conv_one.out_channels
            else:
                final_expand_layer_one = self.last_conv_one.final_expand_layer.get_active_subnet(input_channel_one, preserve_weight)
                feature_mix_layer_one = self.last_conv_one.feature_mix_layer.get_active_subnet(input_channel_one * 6,
                                                                                       preserve_weight)
                in_features_one = feature_mix_layer_one.out_channels
                last_conv_one = nn.Sequential(
                    final_expand_layer_one,
                    nn.AdaptiveAvgPool3d((1, 1, 1)),
                    feature_mix_layer_one
                )
            blocks_two = []
            input_channel_two = first_conv_two.out_channels
            # blocks
            for stage_id, block_idx in enumerate(self.block_group_info):
                depth = self.runtime_depth[stage_id]
                active_idx = block_idx[:depth]
                stage_blocks = []
                for idx in active_idx:
                    stage_blocks.append(ResidualBlock(
                        self.blocks_two[idx].mobile_inverted_conv.get_active_subnet(input_channel_two, preserve_weight),
                        self.blocks_two[idx].shortcut.get_active_subnet(input_channel_two, preserve_weight) if self.blocks_two[
                                                                                                               idx].shortcut is not None else None
                    ))
                    input_channel_two = stage_blocks[-1].mobile_inverted_conv.out_channels
                blocks_two += stage_blocks

            if not self.use_v3_head:
                last_conv_two = self.last_conv_two.get_active_subnet(input_channel_two, preserve_weight)
                in_features_two = last_conv_two.out_channels
            else:
                final_expand_layer_two = self.last_conv_two.final_expand_layer.get_active_subnet(input_channel_two,
                                                                                                 preserve_weight)
                feature_mix_layer_two = self.last_conv_two.feature_mix_layer.get_active_subnet(input_channel_two * 6,
                                                                                               preserve_weight)
                in_features_two = feature_mix_layer_two.out_channels
                last_conv_two = nn.Sequential(
                    final_expand_layer_two,
                    nn.AdaptiveAvgPool3d((1, 1, 1)),
                    feature_mix_layer_two
                )
            blocks_three = []
            input_channel_three = first_conv_three.out_channels
            # blocks
            for stage_id, block_idx in enumerate(self.block_group_info):
                depth = self.runtime_depth[stage_id]
                active_idx = block_idx[:depth]
                stage_blocks = []
                for idx in active_idx:
                    stage_blocks.append(ResidualBlock(
                        self.blocks_three[idx].mobile_inverted_conv.get_active_subnet(input_channel_three, preserve_weight),
                        self.blocks_three[idx].shortcut.get_active_subnet(input_channel_three, preserve_weight) if self.blocks_three[
                                                                                                               idx].shortcut is not None else None
                    ))
                    input_channel_three = stage_blocks[-1].mobile_inverted_conv.out_channels
                blocks_three += stage_blocks

            if not self.use_v3_head:
                last_conv_three = self.last_conv_three.get_active_subnet(input_channel_three, preserve_weight)
                in_features_three = last_conv_three.out_channels
            else:
                final_expand_layer_three = self.last_conv_three.final_expand_layer.get_active_subnet(input_channel_three,preserve_weight)
                feature_mix_layer_three = self.last_conv_three.feature_mix_layer.get_active_subnet(input_channel_three * 6,
                                                                                               preserve_weight)
                in_features_three = feature_mix_layer_three.out_channels
                last_conv_three = nn.Sequential(
                    final_expand_layer_three,
                    nn.AdaptiveAvgPool3d((1, 1, 1)),
                    feature_mix_layer_three
                )

            classifier = self.classifier.get_active_subnet(in_features_one + in_features_two+ in_features_two+ in_features_three, preserve_weight)

            _subnet = TripleETMONASModel(
                first_conv_one, first_conv_two, first_conv_three, blocks_one, blocks_two, blocks_three,
                last_conv_one, last_conv_two, last_conv_three, classifier, self.active_resolution, use_v3_head=self.use_v3_head
            )
            _subnet.set_bn_param(**self.get_bn_param())
            return _subnet

    def load_weights_from_pretrained_models(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert isinstance(checkpoint, dict)
        self.load_state_dict(checkpoint, strict=False)

from yacs.config import CfgNode

def define_sc():
    supernet_cfg = CfgNode(new_allowed=True)

    supernet_cfg.use_v3_head = False
    supernet_cfg.resolutions = [(10, 350, 350)]
    # first_conv
    supernet_cfg.first_conv  = CfgNode(new_allowed=True)
    supernet_cfg.first_conv.c = [32, 40]
    supernet_cfg.first_conv.act_func = "swish"
    supernet_cfg.first_conv.s = '(1, 2, 2)'
    # mb1
    supernet_cfg.mb1  = CfgNode(new_allowed=True)
    supernet_cfg.mb1.c = [16, 24]
    supernet_cfg.mb1.d = [1, 2]
    supernet_cfg.mb1.k = [3]
    supernet_cfg.mb1.t = [1]
    supernet_cfg.mb1.s = '(1, 1, 1)'
    supernet_cfg.mb1.act_func = 'swish'
    supernet_cfg.mb1.se = True
    # mb2
    supernet_cfg.mb2 = CfgNode(new_allowed=True)
    supernet_cfg.mb2.c = [24, 32]
    supernet_cfg.mb2.d = [2, 3]
    supernet_cfg.mb2.k = [3]
    supernet_cfg.mb2.t = [1]
    supernet_cfg.mb2.s = '(1, 2, 2)'
    supernet_cfg.mb2.act_func = 'swish'
    supernet_cfg.mb2.se = True
    # mb3
    supernet_cfg.mb3 = CfgNode(new_allowed=True)
    supernet_cfg.mb3.c = [40, 48]
    supernet_cfg.mb3.d = [2, 3]
    supernet_cfg.mb3.k = [3, 5]
    supernet_cfg.mb3.t = [1]
    supernet_cfg.mb3.s = '(1, 2, 2)'
    supernet_cfg.mb3.act_func = 'swish'
    supernet_cfg.mb3.se = True
    # mb4
    supernet_cfg.mb4 = CfgNode(new_allowed=True)
    supernet_cfg.mb4.c = [80, 88]
    supernet_cfg.mb4.d = [2, 3]
    supernet_cfg.mb4.k = [3, 5]
    supernet_cfg.mb4.t = [1]
    supernet_cfg.mb4.s = '(1, 2, 2)'
    supernet_cfg.mb4.act_func = 'swish'
    supernet_cfg.mb4.se = True
    # mb5
    supernet_cfg.mb5 = CfgNode(new_allowed=True)
    supernet_cfg.mb5.c = [112, 120, 128]
    supernet_cfg.mb5.d = [2, 3, 4]
    supernet_cfg.mb5.k = [3, 5]
    supernet_cfg.mb5.t = [1]
    supernet_cfg.mb5.s = '(1, 1, 1)'
    supernet_cfg.mb5.act_func = 'swish'
    supernet_cfg.mb5.se = True
    # mb6
    supernet_cfg.mb6 = CfgNode(new_allowed=True)
    supernet_cfg.mb6.c = [192, 200, 208, 216]
    supernet_cfg.mb6.d = [2, 3, 4]
    supernet_cfg.mb6.k = [3, 5]
    supernet_cfg.mb6.t = [1]
    supernet_cfg.mb6.s = '(1, 2, 2)'
    supernet_cfg.mb6.act_func = 'swish'
    supernet_cfg.mb6.se = True
    # mb7
    supernet_cfg.mb7 = CfgNode(new_allowed=True)
    supernet_cfg.mb7.c = [320, 352]
    supernet_cfg.mb7.d = [1, 2]
    supernet_cfg.mb7.k = [3, 5]
    supernet_cfg.mb7.t = [1]
    supernet_cfg.mb7.s = '(1, 1, 1)'
    supernet_cfg.mb7.act_func = 'swish'
    supernet_cfg.mb7.se = True
    # last_conv
    supernet_cfg.last_conv = CfgNode(new_allowed=True)
    supernet_cfg.last_conv.c = [1280, 1408]
    supernet_cfg.last_conv.act_func = 'swish'
    return supernet_cfg

