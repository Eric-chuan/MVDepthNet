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

def upsample2d_as(inputs, target_as, mode="bilinear"):
    _, _, h, w = target_as.size()
    return F.interpolate(inputs, [h, w], mode=mode, align_corners=True)

def get_trainable_number(variable):
    num = 1
    shape = list(variable.shape)
    for i in shape:
        num *= i
    return num

class GetVolume(nn.Module):
    """Initialization Stage Class"""

    def __init__(self, num_sample: int = 1) -> None:
        """Initialize method

        Args:
            num_sample: number of samples used in patchmatch process
        """
        super(GetVolume, self).__init__()
        self.num_sample = num_sample

    def forward(
        self,
        left_img: torch.Tensor,
        right_img: torch.Tensor,
        left_proj: torch.Tensor,
        right_proj: torch.Tensor,
        gt_proj: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Forward function for depth initialization

        Args:

            min_depth: minimum virtual depth, (B, )
            max_depth: maximum virtual depth, (B, )
            device: device on which to place tensor
            depth: current depth (B, 1, H, W)
        Returns:
            depth_sample: initialized sample depth map by randomization or local perturbation (B, Ndepth, H, W)
        """
        batch_size, _, height, width = left_img.size()
            # first iteration of Patchmatch on stage 3, sample in the inverse depth range
            # divide the range into several intervals and sample in each of them
        min_depth = torch.FloatTensor([10.]).repeat(batch_size).to(device)
        max_depth = torch.FloatTensor([40.]).repeat(batch_size).to(device)
        inverse_min_depth = 1.0 / min_depth
        inverse_max_depth = 1.0 / max_depth
        num_sample = self.num_sample
        # [B,Ndepth,H,W]
        depth_sample = torch.rand(
            size=(batch_size, num_sample, height, width), device=device
        ) + torch.arange(start=0, end=num_sample, step=1, device=device).view(
            1, num_sample, 1, 1
        )

        depth_sample = inverse_max_depth.view(batch_size, 1, 1, 1) + depth_sample / num_sample * (
            inverse_min_depth.view(batch_size, 1, 1, 1) - inverse_max_depth.view(batch_size, 1, 1, 1)
        )
        depth_sample = 1.0 / depth_sample  # [B,Ndepth,H,W]

        warped_left = image_differentiable_warping(left_img, left_proj, gt_proj, depth_sample) #(B, 3, Ndepth, H, W)
        warped_right = image_differentiable_warping(right_img, right_proj, gt_proj, depth_sample) #(B, 3, Ndepth, H, W)

        # costvolume = torch.sum(torch.abs(warped_right - warped_left), dim=1)  #(B, Ndepth, H, W)
        costvolume = torch.mean((warped_right * warped_left), dim=1)  #(B, Ndepth, H, W)

        return costvolume

class depthNet(nn.Module):
    """docstring for depthNet"""

    def __init__(self):
        super(depthNet, self).__init__()
        # input of the net is plane_sweep_volume, left_image
        # build the net
        # implement a hourglass structure with residure learning
        # encoder
        self.conv1   = conv_layer(70, 128, 7, stride=2)            #H / 2
        self.conv1_1 = conv_layer(128, 128)
        self.conv2   = conv_layer(128, 256, stride=2)              #H / 4
        self.conv2_1 = conv_layer(256, 256)
        self.conv3   = conv_layer(256, 512, stride=2)              #H / 8
        self.conv3_1 = conv_layer(512, 512)   #cost volume
        self.conv4   = conv_layer(512, 512, stride=2)              #H / 16
        self.conv4_1 = conv_layer(512, 512)

        # self.convblock = nn.Sequential(
        #     conv_layer(512, 512),
        #     conv_layer(512, 512),
        #     conv_layer(512, 512),
        #     conv_layer(512, 512),
        # )

        self.disp4 = depth_layer(512)
        # decoder and get depth
        self.upconv3 = deconv_layer(512, 512)                         #H / 8
        self.updisp4to3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.iconv3 = nn.ConvTranspose2d(1025, 512, 3, 1, 1) #input upconv3 + conv3b + updisp4
        self.disp3 = depth_layer(512)

        self.upconv2 = deconv_layer(512, 256)                         #H / 4
        self.updisp3to2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.iconv2 =  nn.ConvTranspose2d(513, 256, 3, 1, 1)  #input upconv2 + conv2 + updisp3
        self.disp2 = depth_layer(256)

        self.upconv1 = deconv_layer(256, 128)                   #H / 2
        self.updisp2to1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.iconv1 =  nn.ConvTranspose2d(257, 128, 3, 1, 1)   #input upconv1 + conv1 + udisp2 = 128 + 128 + 1 =  257
        self.disp1 = depth_layer(128)

        self.upconv0 = up_conv_layer(128, 64)                    #H
        self.updisp1to0 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.iconv0 = nn.ConvTranspose2d(65, 64, 3, 1, 1)  #input upconv0 + imgl + imgr + udisp1 = 128 + 128 + 1 =  71
        self.disp0 = depth_layer(64)

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
        self.GetVolume = GetVolume(64)


    def forward(self, left_image, right_image, left_proj, right_proj, gt_proj):
        self.device = left_image.device
        plane_sweep_volume = self.GetVolume(left_image, right_image, left_proj,
                                            right_proj, gt_proj, self.device)

        x = torch.cat((left_image, right_image, plane_sweep_volume), 1)

        conv1 = self.conv1(x)                                 #B x 128 x H/2 x W/2
        conv1b = self.conv1_1(conv1)                          #B x 128 x H/2 x W/2
        conv2 = self.conv2(conv1b)                             #B x 256 x H/4 x W/4
        conv2b = self.conv2_1(conv2)                             #B x 256 x H/4 x W/4
        conv3a = self.conv3(conv2)                            #B x 512 x H/8 x W/8
        conv3b = self.conv3_1(conv3a)
        conv4a = self.conv4(conv3b)                           #B x 512 x H/16 x W/16
        conv4b = self.conv4_1(conv4a)
        # conv4b = self.convblock(conv4b)

        disp4 = self.disp4(conv4b)

        upconv3 = self.upconv3(conv4b)                                        #B x 512 x H/8 x W/8
        updisp4 = self.updisp4to3(disp4)
        updisp4_interpolated = 2 * upsample2d_as(disp4, updisp4, mode="bilinear")
        iconv3 = self.iconv3(torch.cat((upconv3, conv3b, updisp4), 1))        #B x 512 x H/8 x W/8
        # disp3 = self.disp3(iconv3) + updisp4
        disp3 = self.disp3(iconv3) + updisp4_interpolated

        upconv2 = self.upconv2(conv3b)                                        #B x 256 x H/4 x W/4
        updisp3 = self.updisp3to2(disp3)
        updisp3_interpolated = 2 * upsample2d_as(disp3, updisp3, mode="bilinear")
        iconv2 = self.iconv2(torch.cat((upconv2, conv2b, updisp3), 1))
        # disp2 = self.disp2(iconv2) + updisp3
        disp2 = self.disp2(iconv2) + updisp3_interpolated

        upconv1 = self.upconv1(conv2)
        updisp2 = self.updisp2to1(disp2)
        updisp2_interpolated = 2 * upsample2d_as(disp2, updisp2, mode="bilinear")
        iconv1 = self.iconv1(torch.cat((upconv1, conv1b, updisp2), 1))
        # disp1 = self.disp1(iconv1) + updisp2
        disp1 = self.disp1(iconv1) + updisp2_interpolated

        upconv0 = self.upconv0(conv1)
        updisp1 = self.updisp1to0(disp1)
        updisp1_interpolated = 2 * upsample2d_as(disp1, updisp1, mode="bilinear")
        iconv0 = self.iconv0(torch.cat((upconv0, updisp1), 1))
        # disp0 = self.disp0(iconv0) + updisp1
        disp0 = self.disp0(iconv0) + updisp1_interpolated

        return [disp0, disp1, disp2, disp3]