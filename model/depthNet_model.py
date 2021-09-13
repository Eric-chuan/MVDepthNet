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


def down_conv_layer(input_channels, output_channels, kernel_size):
    return nn.Sequential(
        nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            stride=1,
            bias=False),
   nn.BatchNorm2d(output_channels),
   nn.ReLU(),
        nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            stride=2,
            bias=False),
   nn.BatchNorm2d(output_channels),
   nn.ReLU())


def conv_layer(input_channels, output_channels, kernel_size):
    return nn.Sequential(
        nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False),
  nn.BatchNorm2d(output_channels),
        nn.ReLU())


def depth_layer(input_channels):
    return nn.Sequential(
        nn.Conv2d(input_channels, 1, 3, padding=1), nn.Sigmoid())


def refine_layer(input_channels):
    return nn.Conv2d(input_channels, 1, 3, padding=1)

def up_conv_layer(input_channels, output_channels, kernel_size):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False),
  nn.BatchNorm2d(output_channels),
        nn.ReLU())


def print_error(warp_uv, transform):
    warp_np = warp_uv.cpu().data.numpy()
    print(warp_np.shape)
    print(transform.shape)
    print(warp_np[np.where(warp_np != warp_np)])

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

        warped_right = image_differentiable_warping(right_img, right_proj, left_proj, depth_sample) #(B, 3, Ndepth, H, W)

        costvolume = torch.sum(torch.abs(warped_right - left_img.unsqueeze(2).repeat(1, 1, self.num_sample, 1, 1)), dim=1)  #(B, Ndepth, H, W)

        return costvolume


class depthNet(nn.Module):
    """docstring for depthNet"""

    def __init__(self):
        super(depthNet, self).__init__()
        # input of the net is plane_sweep_volume, left_image
        # build the net
        # implement a hourglass structure with residure learning
        # encoder
        self.conv1 = down_conv_layer(67, 128, 7)
        self.conv2 = down_conv_layer(128, 256, 5)
        self.conv3 = down_conv_layer(256, 512, 3)
        self.conv4 = down_conv_layer(512, 512, 3)
        self.conv5 = down_conv_layer(512, 512, 3)

        # decoder and get depth
        self.upconv5 = up_conv_layer(512, 512, 3)
        self.iconv5 = conv_layer(1024, 512, 3)  #input upconv5 + conv4

        self.upconv4 = up_conv_layer(512, 512, 3)
        self.iconv4 = conv_layer(1024, 512, 3)  #input upconv4 + conv3
        self.disp4 = depth_layer(512)

        self.upconv3 = up_conv_layer(512, 256, 3)
        self.iconv3 = conv_layer(
            513, 256, 3)  #input upconv3 + conv2 + disp4 = 256 + 256 + 1 = 513
        self.disp3 = depth_layer(256)

        self.upconv2 = up_conv_layer(256, 128, 3)
        self.iconv2 = conv_layer(
            257, 128, 3)  #input upconv2 + conv1 + disp3 = 128 + 128 + 1 =  257
        self.disp2 = depth_layer(128)

        self.upconv1 = up_conv_layer(128, 64, 3)
        self.iconv1 = conv_layer(65, 64,
                                 3)  #input upconv1 + disp2 = 64 + 1 = 65
        self.disp1 = depth_layer(64)

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


    def forward(self, left_image, right_image, left_proj, right_proj):
        self.device = left_image.device
        plane_sweep_volume = self.GetVolume(left_image, right_image, left_proj,
                                            right_proj, self.device)

        x = torch.cat((left_image, plane_sweep_volume), 1)

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        upconv5 = self.upconv5(conv5)
        iconv5 = self.iconv5(torch.cat((upconv5, conv4), 1))

        upconv4 = self.upconv4(iconv5)
        iconv4 = self.iconv4(torch.cat((upconv4, conv3), 1))
        disp4 = 2.0 * self.disp4(iconv4)
        udisp4 = F.upsample(disp4, scale_factor=2)

        upconv3 = self.upconv3(iconv4)
        iconv3 = self.iconv3(torch.cat((upconv3, conv2, udisp4), 1))
        disp3 = 2.0 * self.disp3(iconv3)
        udisp3 = F.upsample(disp3, scale_factor=2)

        upconv2 = self.upconv2(iconv3)
        iconv2 = self.iconv2(torch.cat((upconv2, conv1, udisp3), 1))
        disp2 = 2.0 * self.disp2(iconv2)
        udisp2 = F.upsample(disp2, scale_factor=2)

        upconv1 = self.upconv1(iconv2)
        iconv1 = self.iconv1(torch.cat((upconv1, udisp2), 1))
        # print((iconv1 != iconv1).any())
        disp1 = 2.0 * self.disp1(iconv1)

        return [disp1, disp2, disp3, disp4]