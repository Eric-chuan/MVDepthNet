'''
a pytorch model to learn motion stereo
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch import Tensor
import cv2
import math
import numpy as np
import time
from numpy.linalg import inv
from .module import image_differentiable_warping
from .sublayers import *
# from .deform_conv import DeformConv
# def deformable_conv(in_planes, out_planes, kernel_size=3, strides=1, padding=1, use_bias=False):
#     #MFN originally use bias true
#     #AANet deform
#     return DeformConv(in_planes,out_planes,kernel_size=kernel_size,stride=strides,padding=padding, dilation=1,groups=1,deformable_groups=1,bias=False)

class depthNet(nn.Module):
    """docstring for depthNet"""

    def __init__(self):
        super(depthNet, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        # input of the net is plane_sweep_volume, left_image
        # build the net
        # implement a hourglass structure with residure learning
        # encoder
        self.conv1   = conv_layer(3, 64, 7, stride=2)             #H / 2
        self.conv2   = conv_layer(64, 128, stride=2)              #H / 4
        self.conv3   = conv_layer(128, 256, stride=2)             #H / 8
        self.conv_redir = conv_layer(256, 32, stride=1)
        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)

        self.conv3_1 = conv_layer(128, 256)   #cost volume
        self.conv4   = conv_layer(256, 512, stride=2)            #H / 16
        self.conv4_1 = conv_layer(512, 512)
        self.conv5   = conv_layer(512, 512, stride=2)            #H / 32
        self.conv5_1 = conv_layer(512, 512)

        self.disp5 = depth_layer(512)


        # decoder and get depth
        self.upconv5 = deconv_layer(512, 256, 3)                 #H / 16
        self.updisp5to4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.iconv5 = nn.ConvTranspose2d(769, 256, 3, 1, 1) #input upconv5 + conv4b + updisp5
        self.disp4 = depth_layer(256)

        self.upconv4 = deconv_layer(256, 128, 3)                 #H / 8
        self.updisp4to3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.iconv4 = nn.ConvTranspose2d(385, 128, 3, 1, 1) #input upconv4 + conv3b + updisp4
        self.disp3 = depth_layer(128)


        self.upconv3 = deconv_layer(128, 64, 3)                #H / 4
        self.updisp3to2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.iconv3 = nn.ConvTranspose2d(322, 64, 3, 1, 1) #input upconv3 + conv2 + updisp3       （left right correlation）
        self.disp2 = depth_layer(64)

        self.upconv2 = deconv_layer(64, 32, 3)
        self.updisp2to1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.iconv2 = nn.ConvTranspose2d(162, 32, 3, 1, 1) #input upconv2 + conv1 + updisp2
        self.disp1 = depth_layer(32)

        self.upconv1 = deconv_layer(32, 16, 3)                #H / 2
        self.updisp1to0 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.iconv1 = nn.ConvTranspose2d(24, 16, 3, 1, 1) #input upconv1 + updisp1 + left + right
        self.disp0 = depth_layer(16)

        # initialize the weights in the net
        total_num = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                total_num += get_trainable_number(m.weight)
                if m.bias is not None:
                    init.constant(m.bias, 0)
                    total_num += get_trainable_number(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                total_num += get_trainable_number(m.weight)
                init.constant(m.bias, 0)
                total_num += get_trainable_number(m.bias)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=1e-3)
                total_num += get_trainable_number(m.weight)
                if m.bias is not None:
                    init.constant(m.bias, 0)
                    total_num += get_trainable_number(m.bias)
        print("model has %d trainable variables"%total_num)
        self.GetFeatureVolume = GetFeatureVolume(64)
        self.GetFeatureCorrelation = GetFeatureCorrelation(64)
        self.GetImgCorrelation = GetImgCorrelation(64)


    def forward(self, left_image, right_image, left_proj, right_proj, gt_proj):
        self.device = left_image.device

        conv1_l = self.conv1(left_image)             #B x 64  x H/2 x W/2
        conv2_l = self.conv2(conv1_l)                #B x 128 x H/4 x W/4
        conv3a_l = self.conv3(conv2_l)               #B x 256 x H/8 x W/8

        conv1_r = self.conv1(right_image)
        conv2_r = self.conv2(conv1_r)
        conv3a_r = self.conv3(conv2_r)

        out_corr = self.GetFeatureCorrelation(conv3a_l, conv3a_r, left_proj, right_proj, gt_proj, self.device)  #B x Ndepth x H/8 x W/8
        out_corr = self.corr_activation(out_corr)
        out_conv3a_redirl = self.conv_redir(conv3a_l)                                                           #B x 32 x H/8 x W/8
        out_conv3a_redirr = self.conv_redir(conv3a_r)
        in_conv3b = torch.cat((out_conv3a_redirl, out_conv3a_redirr, out_corr), 1)                              #B x 128 x H/8 x W/8

        conv3b = self.conv3_1(in_conv3b)                                                                        #B x 256 x H/8  x W/8
        conv4a = self.conv4(conv3b)                                                                             #B x 512 x H/16 x W/16
        conv4b = self.conv4_1(conv4a)
        conv5a = self.conv5(conv4b)                                                                             #B x 512 x H/32 x W/32
        conv5b = self.conv5_1(conv5a)

        disp5 = self.disp5(conv5b)                                                                              #B x 1 x H/32 x W/32
        upconv5 = self.upconv5(conv5b)                                                                          #B x 256 x H/16 x W/16
        updisp5 = self.updisp5to4(disp5)                                                                        #B x 1 x H/16 x W/16
        iconv5 = self.iconv5(torch.cat((upconv5, updisp5, conv4b), 1))                                          #B x 769 x H/16 x W/16 --  #B x 256 x H/16 x W/16

        disp4 = self.disp4(iconv5)                                                                               #B x 1 x H/16 x W/16
        upconv4 = self.upconv4(iconv5)                                                                           #B x 128 x H/8 x W/8
        updisp4 = self.updisp4to3(disp4)                                                                         #B x 1 x H/8 x W/8
        iconv4 = self.iconv4(torch.cat((upconv4, updisp4, conv3b), 1))                                           #B x 385 x H/8 x W/8

        disp3 = self.disp3(iconv4)                                                                               #B x 1 x H/8 x W/8
        updisp3 = self.updisp3to2(disp3)                                                                         #B x 1 x H/4 x W/4
        upconv3 = self.upconv3(iconv4)                                                                           #B x 64 x H/4 x W/4
        costvolume3 = self.GetFeatureCorrelation(conv2_l, conv2_r, left_proj, right_proj, gt_proj, self.device, updisp3)  #B x 1 x H/4 x W/4
        iconv3 = self.iconv3(torch.cat((upconv3, updisp3, conv2_l, conv2_r, costvolume3), 1))                             #B x 322 x H/4 x W/4

        disp2 = self.disp2(iconv3)                                                                               #B x 1 x H/4 x W/4
        updisp2 = self.updisp2to1(disp2)                                                                         #B x 1 x H/2 x W/2
        upconv2 = self.upconv2(iconv3)                                                                           #B x 32 x H/2 x W/2
        costvolume2 = self.GetFeatureCorrelation(conv1_l, conv1_r, left_proj, right_proj, gt_proj, self.device, updisp2)  #B x 1 x H/2 x W/2
        iconv2 = self.iconv2(torch.cat((upconv2, updisp2, conv1_l, conv1_r, costvolume2), 1))                             #B x 162 x H/2 x W/2

        disp1 = self.disp1(iconv2)                                                                                #B x 1 x H/2 x W/2
        updisp1 = self.updisp1to0(disp1)                                                                          #B x 1 x H x W
        upconv1 = self.upconv1(iconv2)                                                                            #B x 16 x H x W
        costvolume2 = self.GetImgCorrelation(left_image, right_image, left_proj, right_proj, gt_proj, self.device, updisp1)  #B x 1 x H x W
        iconv1 = self.iconv1(torch.cat((upconv1, updisp1, left_image, right_image, costvolume2), 1))                        #B x 24 x H/2 x W/2
        # print((iconv1 != iconv1).any())
        disp0 = self.disp0(iconv1)
        disp0 = self.relu(disp0)

        return [disp0, disp1, disp2, disp3, disp4, disp5]