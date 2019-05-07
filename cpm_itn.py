#!usr/bin/python
# -*- coding: utf-8 -*-
##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################
from __future__ import division
import time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from .model_utils import get_parameters, load_weight_from_dict, count_network_param
from .basic_batch import find_tensor_peak_batch
from .initialization import weights_init_cpm
from .cycle_util import load_network
from .itn import define_G
class HgResBlock(nn.Module):
    ''' Hourglass residual block '''
    def __init__(self, inplanes, outplanes, stride=1):
        super().__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        midplanes = outplanes // 2
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, midplanes, 1, stride)  # bias=False
        self.bn2 = nn.BatchNorm2d(midplanes)
        self.conv2 = nn.Conv2d(midplanes, midplanes, 3, stride, 1)
        self.bn3 = nn.BatchNorm2d(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, 1, stride)  # bias=False
        self.relu = nn.ReLU(inplace=True)
        if inplanes != outplanes:
            self.conv_skip = nn.Conv2d(inplanes, outplanes, 1, 1)
 
    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.inplanes != self.outplanes:
            residual = self.conv_skip(residual)
        out += residual
        return out
class Hourglass(nn.Module):
    def __init__(self, depth, nFeat, nModules, resBlock):
        super().__init__()
        self.depth = depth
        self.nFeat = nFeat
        self.nModules = nModules  # num residual modules per location
        self.resBlock = resBlock
 
        self.hg = self._make_hour_glass()
        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
 
    def _make_hour_glass(self):
        hg = []
        for i in range(self.depth):
            res = [self._make_residual(self.nModules) for _ in range(3)]  # skip(upper branch); down_path, up_path(lower branch)
            if i == (self.depth - 1):
                res.append(self._make_residual(self.nModules))  # extra one for the middle
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)
 
    def _make_residual(self, n):
        return nn.Sequential(*[self.resBlock(self.nFeat, self.nFeat) for _ in range(n)])
 
    def forward(self, x):
        return self._hour_glass_forward(0, x)
 
    def _hour_glass_forward(self, depth_id, x):
        up1 = self.hg[depth_id][0](x)
        low1 = self.downsample(x)
        low1 = self.hg[depth_id][1](low1)
        if depth_id == (self.depth - 1):
            low2 = self.hg[depth_id][3](low1)
        else:
            low2 = self._hour_glass_forward(depth_id + 1, low1)
        low3 = self.hg[depth_id][2](low2)
        up2 = self.upsample(low3)
        return up1 + up2
 
 
class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, nStacks, nModules, nFeat, nClasses, resBlock=HgResBlock, inplanes=128):
        super().__init__()
        self.nStacks = nStacks
        self.nModules = nModules
        self.nFeat = nFeat
        self.nClasses = nClasses
        self.resBlock = resBlock
        self.inplanes = inplanes
 
        self._make_head()
 
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(nStacks):
            hg.append(Hourglass(4, nFeat, nModules, resBlock))
            res.append(self._make_residual(nModules))
            fc.append(self._make_fc(nFeat, nFeat))
            score.append(nn.Conv2d(nFeat, nClasses, 1))
            if i < (nStacks - 1):
                fc_.append(nn.Conv2d(nFeat, nFeat, 1))
                score_.append(nn.Conv2d(nClasses, nFeat, 1))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)
 
    def _make_head(self):
        self.conv1 = nn.Conv2d(self.inplanes, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.res1 = self.resBlock(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.res2 = self.resBlock(128, 128)
        self.res3 = self.resBlock(128, self.nFeat)
 
    def _make_residual(self, n):
        return nn.Sequential(*[self.resBlock(self.nFeat, self.nFeat) for _ in range(n)])
 
    def _make_fc(self, inplanes, outplanes):
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(True))
 
    def forward(self, x):
        # head
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
 
        x = self.res1(x)
        x = self.pool(x)
        x = self.res2(x)
        x = self.res3(x)
 
        out = []
        for i in range(self.nStacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < (self.nStacks - 1):
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_
 
    return out
class ITN_CPM(nn.Module):
  def __init__(self, model_config):
    super(ITN_CPM, self).__init__()

    self.config = model_config.copy()
    self.downsample = 1

    self.netG_A = define_G()
    self.netG_B = define_G()
    self.HourglassNet = HourglassNet
    self.stage1 = self.HourglassNet(4,1,256,68)
  def _make_Hourglass(self, n):
        return nn.Sequential(*[self.HourglassNet(self.nStacks, self.nModules, self.nFeat, self.nClasses,) for _ in range(n)])
    
    self.features = nn.Sequential(
          nn.Conv2d(  3,  64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d( 64,  64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d( 64, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d(128, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d(256, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True))
  
    self.downsample = 8
    pts_num = self.config.pts_num

    self.CPM_feature = nn.Sequential(
          nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), #CPM_1
          nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True)) #CPM_2

    '''self.stage1 = nn.Sequential(
          nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128, 512, kernel_size=1, padding=0), nn.ReLU(inplace=True),
          nn.Conv2d(512, pts_num, kernel_size=1, padding=0),nn.ReLU(inplace=True),
          nn.Conv2d(pts_num, pts_num, kernel_size=1, padding=0))'''

    self.stage2 = nn.Sequential(
          nn.Conv2d(128*2+pts_num*2, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=1, padding=0), nn.ReLU(inplace=True),
          nn.Conv2d(128,     pts_num, kernel_size=1, padding=0),nn.ReLU(inplace=True),
          nn.Conv2d(pts_num, pts_num, kernel_size=1, padding=0))

    self.stage3 = nn.Sequential(
          nn.Conv2d(128*2+pts_num, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
          nn.Conv2d(128,         256, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
          nn.Conv2d(256,         256, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
          nn.Conv2d(256,         256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(256,         256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(256,         256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(256,         256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(256,         128, kernel_size=1, padding=0), nn.ReLU(inplace=True),
          nn.Conv2d(128,     pts_num, kernel_size=1, padding=0))
  
    assert self.config.num_stages >= 1, 'stages of cpm must >= 1'

  def set_mode(self, mode):
    if mode.lower() == 'train':
      self.train()
      for m in self.netG_A.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
          m.eval()
      for m in self.netG_B.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
          m.eval()
    elif mode.lower() == 'eval':
      self.eval()
    else:
      raise NameError('The wrong mode : {}'.format(mode))
  
  def specify_parameter(self, base_lr, base_weight_decay):
    params_dict = [ {'params': self.netG_A.parameters()                   , 'lr': 0        , 'weight_decay': 0},
                    {'params': self.netG_B.parameters()                   , 'lr': 0        , 'weight_decay': 0},
                    {'params': get_parameters(self.features,   bias=False), 'lr': base_lr  , 'weight_decay': base_weight_decay},
                    {'params': get_parameters(self.features,   bias=True ), 'lr': base_lr*2, 'weight_decay': 0},
                    {'params': get_parameters(self.CPM_feature, bias=False), 'lr': base_lr  , 'weight_decay': base_weight_decay},
                    {'params': get_parameters(self.CPM_feature, bias=True ), 'lr': base_lr*2, 'weight_decay': 0},
                    {'params': get_parameters(self.stage1,      bias=False), 'lr': base_lr  , 'weight_decay': base_weight_decay},
                    {'params': get_parameters(self.stage1,      bias=True ), 'lr': base_lr*2, 'weight_decay': 0},
                    {'params': get_parameters(self.stage2,      bias=False), 'lr': base_lr*4, 'weight_decay': base_weight_decay},
                    {'params': get_parameters(self.stage2,      bias=True ), 'lr': base_lr*8, 'weight_decay': 0},
                    {'params': get_parameters(self.stage3,      bias=False), 'lr': base_lr*4, 'weight_decay': base_weight_decay},
                    {'params': get_parameters(self.stage3,      bias=True ), 'lr': base_lr*8, 'weight_decay': 0}
                  ]
    return params_dict

  # return : cpm-stages, locations
  def forward(self, inputs):
    assert inputs.dim() == 4, 'This model accepts 4 dimension input tensor: {}'.format(inputs.size())
    batch_size = inputs.size(0)
    num_stages, num_pts = self.config.num_stages, self.config.pts_num - 1

    batch_cpms, batch_locs, batch_scos = [], [], []     # [Squence, Points]

    features, stage1s = [], []
    inputs = [inputs, (self.netG_A(inputs)+self.netG_B(inputs))/2]
    for input in inputs:
      feature  = self.features(input)
      feature = self.CPM_feature(feature)
      features.append(feature)
      stage1s.append( self.stage1(feature) )

    xfeature = torch.cat(features, 1)
    #cpm_stage2 =self.HourglassNet(4,1,256,68)
    cpm_stage2 = self.stage2(torch.cat([xfeature, stage1s[0], stage1s[1]], 1))
    cpm_stage3 = self.stage3(torch.cat([xfeature, cpm_stage2], 1))
    batch_cpms = [stage1s[0], stage1s[1]] + [cpm_stage2, cpm_stage3]

    # The location of the current batch
    for ibatch in range(batch_size):#batch_size输入图像
      batch_location, batch_score = find_tensor_peak_batch(cpm_stage3[ibatch], self.config.argmax, self.downsample)
      batch_locs.append( batch_location )
      batch_scos.append( batch_score )
    batch_locs, batch_scos = torch.stack(batch_locs), torch.stack(batch_scos)

    return batch_cpms, batch_locs, batch_scos, inputs[1:]


# use vgg16 conv1_1 to conv4_4 as feature extracation        
model_urls = 'https://download.pytorch.org/models/vgg16-397923af.pth'

def itn_cpm(model_config, cycle_model_path):
  
  print ('Initialize ITN-CPM with configure : {}'.format(model_config))
  model = ITN_CPM(model_config)
  model.apply(weights_init_cpm)

  if model_config.pretrained:
    print ('vgg16_base use pre-trained model')
    weights = model_zoo.load_url(model_urls)
    load_weight_from_dict(model, weights, None, False)#加载预训练模型的参数

  if cycle_model_path:
    load_network(cycle_model_path, 'G_A', model.netG_A)
    load_network(cycle_model_path, 'G_B', model.netG_B)

  print ('initialize the generator network by {} with {} parameters'.format(cycle_model_path, count_network_param(model)))
  return model
