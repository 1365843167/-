# -*- coding:utf-8 -*-
#pythorch
import torch.nn as nn
import torch
from torch import autograd
#https://github.com/JavisPeng/u_net_liver
#http://www.zhongruitech.com/244512251.html  unet 医学ct影像检测博客   语义分割也用到 输入输出改变
#把常用的2个卷积操作简单封装下
class DoubleConv(nn.Module):  #相当于bottenneck
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), #添加了BN层
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

# class torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0,  #逆卷积，转置卷积，反卷积
    # output_padding=0, groups=1, bias=True, dilation=1)
	#output=(input-1)*stride+outputpadding-2*padding+kernelsize   #反卷积后输出特征图大小的公式
	
	# in_channels(int) – 输入信号的通道数
	# out_channels(int) – 卷积产生的通道数
	# kerner_size(int or tuple) - 卷积核的大小
	# stride(int or tuple,optional) - 卷积步长，即要将输入扩大的倍数。
	# padding(int or tuple, optional) - 输入的每一条边补充0的层数，高宽都增加2*padding
	# output_padding(int or tuple, optional) - 输出边补充0的层数，高宽都增加padding
	# groups(int, optional) – 从输入通道到输出通道的阻塞连接数
	# bias(bool, optional) - 如果bias=True，添加偏置
	# dilation(int or tuple, optional) – 卷积核元素之间的间距

class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        # 逆卷积，也可以使用上采样
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)  #ConvTranspose2d pytorch 自带
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out