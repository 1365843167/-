# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# -*- coding:utf-8 -*-
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division
import time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from copy import deepcopy
from .model_utils import get_parameters
from .basic_batch import find_tensor_peak_batch
from .initialization import weights_init_cpm

class VGG16_base(nn.Module):   #设256输入图大小
  def __init__(self, config, pts_num):
    super(VGG16_base, self).__init__()

    self.config = deepcopy(config)
    self.downsample = 8   #下采样为原图的1/8
    self.pts_num = pts_num  #有16,106,68
    #输入输出通道数、卷积核大小，dilation卷积核元素之间的间距即加0的个数。扩大感受野；是空洞卷积或卷积膨胀操作，pixel-wise逐像素卷积，
    #padding=1在输入的特征图周围包上一层0如5*5变7*7，由于卷积核大小和步长限制导致有时卷积核不能走遍全图，不使用padding会丢失部分值
    #groups从输入通道到输出通道的阻塞连接数，这里没有   bias=True，添加偏置
    # class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    self.features = nn.Sequential(
          nn.Conv2d(  3,  64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),  #默认stride=1，(256+2-3+2*1)/1+1=258
          nn.Conv2d( 64,  64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),  #默认stride=1，(258+2-3+2*1)/1+1=260
          nn.MaxPool2d(kernel_size=2, stride=2),  #(260-2)/2 +1=130
          nn.Conv2d( 64, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),  #(130+2-3+2*1)/1+1=132
          nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),  #(132+2-3+2*1)/1+1=134
          nn.MaxPool2d(kernel_size=2, stride=2),  #(134-2)/2 +1=67
          nn.Conv2d(128, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),  #(67+2-3+2*1)/1+1=69
          nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),  #(69+2-3+2*1)/1+1=71
          nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),  #(71+2-3+2*1)/1+1=73
          nn.MaxPool2d(kernel_size=2, stride=2),  #(73-2)/2 +1=37
          nn.Conv2d(256, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),  #(37+2-3+2*1)/1+1=39
          nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),  #(39+2-3+2*1)/1+1=41
          nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True))  #(41+2-3+2*1)/1+1=43

    self.CPM_feature = nn.Sequential(
          nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), #CPM_1  #(43-3+2*1)/1+1=43
          nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True)) #CPM_2  #(43-3+2*1)/1+1=43

    assert self.config.stages >= 1, 'stages of cpm must >= 1 not : {:}'.format(self.config.stages)
    stage1 = nn.Sequential(
          nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128, 512, kernel_size=1, padding=0), nn.ReLU(inplace=True),
          nn.Conv2d(512, pts_num, kernel_size=1, padding=0))
    stages = [stage1]
    for i in range(1, self.config.stages):
      stagex = nn.Sequential(
          nn.Conv2d(128+pts_num, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=1, padding=0), nn.ReLU(inplace=True),
          nn.Conv2d(128,     pts_num, kernel_size=1, padding=0))
      stages.append( stagex )
    self.stages = nn.ModuleList(stages)
  
  def specify_parameter(self, base_lr, base_weight_decay):
    params_dict = [ {'params': get_parameters(self.features,   bias=False), 'lr': base_lr  , 'weight_decay': base_weight_decay},
                    {'params': get_parameters(self.features,   bias=True ), 'lr': base_lr*2, 'weight_decay': 0},
                    {'params': get_parameters(self.CPM_feature, bias=False), 'lr': base_lr  , 'weight_decay': base_weight_decay},
                    {'params': get_parameters(self.CPM_feature, bias=True ), 'lr': base_lr*2, 'weight_decay': 0},
                  ]
    for stage in self.stages:
      params_dict.append( {'params': get_parameters(stage, bias=False), 'lr': base_lr*4, 'weight_decay': base_weight_decay} )
      params_dict.append( {'params': get_parameters(stage, bias=True ), 'lr': base_lr*8, 'weight_decay': 0} )
    return params_dict

  # return : cpm-stages, locations
  def forward(self, inputs):
    assert inputs.dim() == 4, 'This model accepts 4 dimension input tensor: {}'.format(inputs.size())
    batch_size, feature_dim = inputs.size(0), inputs.size(1)
    batch_cpms, batch_locs, batch_scos = [], [], []

    feature  = self.features(inputs)
    xfeature = self.CPM_feature(feature)
    for i in range(self.config.stages):
      if i == 0: cpm = self.stages[i]( xfeature )
      else:      cpm = self.stages[i]( torch.cat([xfeature, batch_cpms[i-1]], 1) )
      batch_cpms.append( cpm )

    # The location of the current batch
    for ibatch in range(batch_size):
      batch_location, batch_score = find_tensor_peak_batch(batch_cpms[-1][ibatch], self.config.argmax, self.downsample)
      batch_locs.append( batch_location )
      batch_scos.append( batch_score )
    batch_locs, batch_scos = torch.stack(batch_locs), torch.stack(batch_scos)

    return batch_cpms, batch_locs, batch_scos

# use vgg16 conv1_1 to conv4_4 as feature extracation        
model_urls = 'https://download.pytorch.org/models/vgg16-397923af.pth'

def cpm_vgg16(config, pts):
  
  print ('Initialize cpm-vgg16 with configure : {}'.format(config))
  model = VGG16_base(config, pts)
  model.apply(weights_init_cpm)

  if config.pretrained:
    print ('vgg16_base use pre-trained model')
    weights = model_zoo.load_url(model_urls)
    model.load_state_dict(weights, strict=False)
  return model
