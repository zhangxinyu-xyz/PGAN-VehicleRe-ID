# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp
import torch
from torch import nn
import numpy as np
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .attention import PAM
from .BasicBottleneck import ResidualBlock, ResNeXtBottleneck, SELayer


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
            
def weights_init_attention(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.constant_(m.weight, 0.0)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice,
                 pgan='yes', attention_dim=512, prop_num=10, reduction=16, multi_nums=2, embed_num=128, temp=10.0):
        super(Baseline, self).__init__()
        self.model_name = model_name
        self.pgan = True if pgan == 'yes' else False
        self.attention_dim = attention_dim
        self.prop_num = prop_num
        self.reduction = reduction
        self.multi_nums = multi_nums
        self.embed_num = embed_num

        '''The backbone network.'''
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])

        if pretrain_choice == 'imagenet':
            self.base.load_param(osp.expanduser(model_path))
            print('Loading pretrained ImageNet model......')

        '''calculate the channels of multi output.'''
        self.num_features = []
        for i in range(self.multi_nums):
            index = len(self.base._modules['layer' + str(4 - i)]._modules) - 1
            bn_count = len(
                [x for x in self.base._modules['layer' + str(4 - i)]._modules[str(index)]._modules if 'bn' in x])
            self.num_features.append(
                eval('self.base.layer' + str(4 - i) + str([index]) + '.bn' + str(bn_count) + '.num_features'))
        self.num_features = self.num_features[::-1]

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        self.bottleneck = []
        self.classifier = []
        self.refine_concat = []
        self.refine_prop = []
        self.refine_base = []
        for i in range(self.multi_nums):
            self.bottleneck.append(nn.BatchNorm1d(self.embed_num * 2))
            self.bottleneck[i].bias.requires_grad_(False)  # no shift
            self.bottleneck[i].apply(weights_init_kaiming)
            self.classifier.append(nn.Linear(self.embed_num * 2, self.num_classes, bias=False))
            self.classifier[i].apply(weights_init_classifier)
            refine_concat = nn.Sequential(
                SELayer(channel=self.num_features[i] * 2, reduction=8),
                ResidualBlock(input_channels=self.num_features[i] * 2, output_channels=self.embed_num * 2),
                )
            refine_concat.apply(weights_init_kaiming)
            self.refine_concat.append(refine_concat)

            refine_prop = nn.Sequential(
                SELayer(channel=self.num_features[i], reduction=8),
                ResidualBlock(input_channels=self.num_features[i], output_channels=self.embed_num),
            )
            refine_prop.apply(weights_init_kaiming)
            self.refine_prop.append(refine_prop)

            refine_base = nn.Sequential(
                SELayer(channel=self.num_features[i], reduction=8),
                ResidualBlock(input_channels=self.num_features[i], output_channels=self.embed_num),
            )
            refine_base.apply(weights_init_kaiming)
            self.refine_base.append(refine_base)

        self.bottleneck = nn.Sequential(*self.bottleneck)
        self.classifier = nn.Sequential(*self.classifier)
        self.refine_concat = nn.Sequential(*self.refine_concat)
        self.refine_prop = nn.Sequential(*self.refine_prop)
        self.refine_base = nn.Sequential(*self.refine_base)

        self.att_avg_pool = nn.AdaptiveAvgPool2d(1)
        assert self.attention_dim == self.in_planes, "Attention dim should be same with the feat dim"
        self.attention = []
        for i in range(self.multi_nums):
            self.attention.append(PAM(self.num_features[i], reduction=4, temp=temp))
            self.attention[i].apply(weights_init_attention)
        self.attention = nn.Sequential(*self.attention)

    def extract_proposal_feature(self, base_feat, proposal, prop_num=10):
        prop_feat = base_feat.new_zeros(size=base_feat.size(), requires_grad=base_feat.requires_grad)
        prop_feat = prop_feat.unsqueeze(1)
        prop_feat = prop_feat.repeat(1, prop_num, 1, 1, 1)
        for num in range(prop_num):
            prop = proposal[:, num, :, :].unsqueeze(1)
            prop_feat[:, num, :, :, :] = base_feat * prop
        return prop_feat

    def forward(self, x):
        x, proposal = x

        features = {}
        num = 0
        for name, module in self.base._modules.items():
            '''added for backbone.'''
            if name == 'avgpool':
                break
            x = module(x) #
            if 'layer' not in name:
                continue
            if 'resnet' in self.model_name and int(name[-1]) <= 4 - self.multi_nums:
                continue
            base_feat = x.clone()

            prop_feat_m = self.extract_proposal_feature(base_feat, proposal, prop_num=self.prop_num)
            prop_feat_m = prop_feat_m.sum((-2, -1)) / proposal.unsqueeze(2).repeat(1, 1, base_feat.size(1), 1, 1).sum((-2, -1)).type(torch.float)
            att = self.attention[num](prop_feat_m)
            prop_feat_m = proposal * att

            prop_feat_m = prop_feat_m.sum(1, keepdim=True) * base_feat

            prop_feat_m = prop_feat_m + base_feat
            global_prop_feat = torch.cat((base_feat, prop_feat_m), 1)

            # Refine module
            # fusion feature
            global_prop_feat = self.refine_concat[num](global_prop_feat)
            global_prop_feat = self.gap(global_prop_feat)                            # (b, 2048, 1, 1)
            global_prop_feat = global_prop_feat.view(global_prop_feat.shape[0], -1)  # flatten to (bs, 2048)

            # global feature and part-guided feature.
            prop_feat_m = self.refine_prop[num](prop_feat_m)
            base_feat = self.refine_base[num](base_feat)

            prop_feat_m = self.gap(prop_feat_m)
            prop_feat_m = prop_feat_m.view(prop_feat_m.shape[0], -1)
            base_feat = self.gap(base_feat)
            base_feat = base_feat.view(base_feat.shape[0], -1)

            # last neck for softmax function.
            if self.neck == 'no':
                feat = global_prop_feat.clone()
            elif self.neck == 'bnneck':
                feat = self.bottleneck[num](global_prop_feat)

            if self.training:
                cls_score = self.classifier[num](feat)

                if name not in features:
                    features[name] = {}
                features[name]['global'] = global_prop_feat
                features[name]['att'] = att
                features[name]['prop'] = prop_feat_m
                features[name]['base'] = base_feat
                features[name]['cls'] = cls_score
            else:
                if name not in features:
                    features[name] = {}
                features[name]['feat'] = feat
                features[name]['global'] = global_prop_feat
                features[name]['att'] = att
                features[name]['base'] = base_feat
                features[name]['prop'] = prop_feat_m

            num += 1
        return features

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
