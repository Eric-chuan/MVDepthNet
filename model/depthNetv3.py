'''
A implementation of DispNetS+RIFE(cascade residual network)
'''
from os import umask
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
from .refine import *
# from .deform_conv import DeformConv
# def deformable_conv(in_planes, out_planes, kernel_size=3, strides=1, padding=1, use_bias=False):
#     #MFN originally use bias true
#     #AANet deform
#     return DeformConv(in_planes,out_planes,kernel_size=kernel_size,stride=strides,padding=padding, dilation=1,groups=1,deformable_groups=1,bias=False)
def upsample2d_as(inputs, target_as, mode="bilinear"):
    _, _, h, w = target_as.size()
    return F.interpolate(inputs, [h, w], mode=mode, align_corners=True)

class depthResBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(depthResBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv_layer(in_planes, c//2, stride=2),
            conv_layer(c//2, c, stride=2),
            )
        self.convblock = nn.Sequential(
            conv_layer(c, c),
            conv_layer(c, c),
            conv_layer(c, c),
            conv_layer(c, c),
            conv_layer(c, c),
            conv_layer(c, c),
            conv_layer(c, c),
            conv_layer(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 2, 4, 2, 1)
        self.sigmoid_func = nn.Sigmoid()

    def forward(self, x, disp, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
            disp = F.interpolate(disp, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
        x = torch.cat((x, disp), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor = scale * 2, mode="bilinear", align_corners=False)
        tmp = self.sigmoid_func(tmp)
        disp = tmp[:, :1]
        mask = tmp[:, 1:]
        return disp, mask

class depthNet(nn.Module):
    """docstring for depthNet"""

    def __init__(self):
        super(depthNet, self).__init__()
        # input of the net is plane_sweep_volume, left_image
        # build the net
        # implement a hourglass structure with residure learning
        # encoder
        self.conv1   = conv_layer(6, 32, 7, stride=2)             #H / 2
        self.conv2   = conv_layer(32, 64, stride=2)               #H / 4

        self.conv3   = conv_layer(64, 128, stride=2)              #H / 8
        self.conv3_1 = conv_layer(128, 128)
        self.conv4   = conv_layer(128, 256, stride=2)             #H / 16
        self.conv4_1 = conv_layer(256, 256)
        self.conv5   = conv_layer(256, 512, stride=2)             #H / 32
        self.conv5_1 = conv_layer(512, 512)
        self.conv6   = conv_layer(512, 1024, stride=2)            #H / 64
        self.conv6_1 = conv_layer(1024, 1024)

        self.disp6 = depth_layer(1024)

        # decoder and get depth
        self.upconv5 = deconv_layer(1024, 512)                    #H / 32
        self.updisp6to5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.iconv5 = nn.ConvTranspose2d(1025, 512, 3, 1, 1) #input upconv5 + conv5b + updisp6
        self.disp5 = depth_layer(512)

        self.upconv4 = deconv_layer(512, 256)                     #H / 16
        self.updisp5to4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.iconv4 = nn.ConvTranspose2d(513, 256, 3, 1, 1) #input upconv4 + conv4b + updisp5
        self.disp4 = depth_layer(256)

        self.upconv3 = deconv_layer(256, 128)                      #H / 8
        self.updisp4to3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.iconv3 = nn.ConvTranspose2d(257, 128, 3, 1, 1) #input upconv3 + conv3b + updisp4
        self.disp3 = depth_layer(128)

        self.upconv2 = deconv_layer(128, 64)                      #H / 4
        self.updisp3to2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.iconv2 = nn.ConvTranspose2d(129, 64, 3, 1, 1) #input upconv2 + conv2 + updisp3
        self.disp2 = depth_layer(64)

        self.upconv1 = deconv_layer(64, 32)                        #H / 2
        self.updisp2to1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.iconv1 = nn.ConvTranspose2d(65, 32, 3, 1, 1) #input upconv1 + conv1 + updisp2
        self.disp1 = depth_layer(32)

        self.upconv0 = deconv_layer(32, 32)                        #H / 2
        self.updisp1to0 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.iconv0 = nn.ConvTranspose2d(39, 32, 3, 1, 1) #input upconv0 + left + right + updisp1
        self.disp0 = depth_layer(32)
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid_func = nn.Sigmoid()

        self.getwarpedimg = GetWarpedImg()
        self.resblock1 = depthResBlock(13, c=150)
        self.resblock2 = depthResBlock(14, c=90)
        self.contextnet = Contextnet()
        self.unet = Unet()

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


    def forward(self, left_image, right_image, left_proj, right_proj, gt_proj):
        self.device = left_image.device
        input = torch.cat((left_image, right_image), dim=1)
        conv1 = self.conv1(input)                    #B x 32 x H/2 x W/2
        conv2 = self.conv2(conv1)                    #B x 64 x H/4 x W/4
        conv3a = self.conv3(conv2)                   #B x 128 x H/8 x W/8
        conv3b = self.conv3_1(conv3a)                #B x 128 x H/8 x W/8
        conv4a = self.conv4(conv3b)                  #B x 256 x H/16 x W/16
        conv4b = self.conv4_1(conv4a)
        conv5a = self.conv5(conv4b)                  #B x 512 x H/32 x W/32
        conv5b = self.conv5_1(conv5a)
        conv6a = self.conv6(conv5b)                  #B x 1024 x H/64 x W/64
        conv6b = self.conv6_1(conv6a)

        disp6 = self.disp6(conv6b)                                                                              #B x 1 x H/64 x W/64

        upconv5 = self.upconv5(conv6b)                                                                          #B x 512 x H/32 x W/32
        updisp6 = self.updisp6to5(disp6)                                                                        #B x 1 x H/32 x W/32
        iconv5 = self.iconv5(torch.cat((upconv5, updisp6, conv5b), 1))                                  #B x 1025 x H/32 x W/32 --  #B x 512 x H/32 x W/32
        disp5 = self.disp5(iconv5)                                                                               #B x 1 x H/32 x W/32

        upconv4 = self.upconv4(iconv5)                                                                           #B x 256 x H/16 x W/16
        updisp5 = self.updisp5to4(disp5)                                                                         #B x 1 x H/16 x W/16
        iconv4 = self.iconv4(torch.cat((upconv4, updisp5, conv4b), 1))                                           #B x 256 x H/16 x W/16
        disp4 = self.disp4(iconv4)                                                                               #B x 1 x H/16 x W/16

        upconv3 = self.upconv3(iconv4)                                                                           #B x 128 x H/8 x W/8
        updisp4 = self.updisp4to3(disp4)                                                                         #B x 1 x H/8 x W/8
        updisp4_interpolated = 2 * upsample2d_as(disp4, updisp4, mode="bilinear")
        iconv3 = self.iconv3(torch.cat((upconv3, updisp4, conv3b), 1))                                           #B x 128 x H/8 x W/8
        disp3 = self.disp3(iconv3)                                                                               #B x 1 x H/8 x W/8

        upconv2 = self.upconv2(iconv3)                                                                           #B x 64 x H/4 x W/4
        updisp3 = self.updisp3to2(disp3)                                                                         #B x 1 x H/4 x W/4
        updisp3_interpolated = 2 * upsample2d_as(disp3, updisp3, mode="bilinear")
        iconv2 = self.iconv2(torch.cat((upconv2, updisp3, conv2), 1))                                            #B x 64 x H/4 x W/4
        disp2 = self.disp2(iconv2)                                                                               #B x 1 x  H/4 x W/4

        upconv1 = self.upconv1(iconv2)                                                                           #B x 32 x H/2 x W/2
        updisp2 = self.updisp2to1(disp2)                                                                         #B x 1 x H/2 x W/2
        updisp2_interpolated = 2 * upsample2d_as(disp2, updisp2, mode="bilinear")
        iconv1 = self.iconv1(torch.cat((upconv1, updisp2, conv1), 1))                                            #B x 32 x H/2 x W/2
        disp1 = self.disp1(iconv1)                                                                               #B x 1 x  H/2 x W/2

        upconv0 = self.upconv0(iconv1)                                                                           #B x 32 x H x W
        updisp1 = self.updisp1to0(disp1)                                                                         #B x 1 x H x W
        updisp1_interpolated = 2 * upsample2d_as(disp1, updisp1, mode="bilinear")
        iconv0 = self.iconv0(torch.cat((upconv0, updisp1, left_image, right_image), 1))                          #B x 32 x H x W
        disp0 = self.disp0(iconv0)
        # disp0 = self.sigmoid_func(disp0)

        if torch.any(torch.isnan(disp0)):
            print('exit nan elements disp out')
        # disp0 = self.relu(disp0)
        # return [disp0, disp1, disp2, disp3, disp4, disp5, disp6]
        warpedLeft_img0, warpedRight_img0 = self.getwarpedimg(left_image, right_image, left_proj, right_proj, gt_proj, self.device, disp_sample=disp0)

        disp_res1, mask_res1 = self.resblock1(torch.cat((left_image, right_image, warpedLeft_img0, warpedRight_img0), dim=1), disp0, scale=2)
        warpedLeft_img1, warpedRight_img1 = self.getwarpedimg(left_image, right_image, left_proj, right_proj, gt_proj, self.device, disp_sample=disp_res1)
        merged1 = warpedLeft_img1 * mask_res1 + warpedRight_img1 * (1 - mask_res1)

        disp_res2, mask_res2 = self.resblock2(torch.cat((left_image, right_image, warpedLeft_img1, warpedRight_img1, mask_res1), dim=1), disp_res1, scale=1)
        warpedLeft_img2, warpedRight_img2 = self.getwarpedimg(left_image, right_image, left_proj, right_proj, gt_proj, self.device, disp_sample=disp_res2)
        merged2 = warpedLeft_img2 * mask_res2 + warpedRight_img2 * (1 - mask_res2)

        #refinement
        c0 = self.contextnet(left_image, left_proj, gt_proj, disp_res2)
        c1 = self.contextnet(right_image, right_proj, gt_proj, disp_res2)
        unet_output = self.unet(left_image, right_image, warpedLeft_img2, warpedRight_img2, mask_res2, disp_res2, c0, c1)
        merged_unet = unet_output[:,:3] * 2 - 1
        merged_refine = torch.clamp(merged2 + merged_unet, 0, 1)

        return [merged_refine, merged2, merged1], [disp_res2, disp_res1, disp0], [mask_res2, mask_res1]



