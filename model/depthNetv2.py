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


class depthNet(nn.Module):
    """docstring for IDepthNet"""

    def __init__(self, resBlock=True):
        super(depthNet, self).__init__()
        # input of the net is plane_sweep_volume, left_image
        # build the net
        # implement a hourglass structure with residure learning
        # encoder
        self.block0 = IDepthBlock(6, c=240)
        self.block1 = IDepthBlock(13+2, c=150)
        self.block2 = IDepthBlock(13+2, c=90)

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


    def forward(self, left_image, right_image, left_proj, right_proj, gt_proj, scale=[4,2,1]):
        self.device = left_image.device
        disp_list = []
        merged = []
        mask_list = []
        warped_img0 = left_image
        warped_img1 = right_image
        disp = None
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if disp != None:
                disp_d, mask_d = stu[i](torch.cat((left_image, right_image, warped_img0, warped_img1, mask), 1), disp, scale=scale[i])
                disp = disp + disp_d
                mask = mask + mask_d
            else:
                disp, mask = stu[i](torch.cat((left_image, right_image), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            disp_list.append(disp)
            warped_img0 = image_differentiable_warping(left_image, left_proj, gt_proj, 1. / disp[:, :1]).squeeze(2)
            warped_img1 = image_differentiable_warping(right_image, right_proj, gt_proj, 1. / disp[:, 1:2]).squeeze(2)
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        for i in range(3):
            disp_list[i] = disp_list[i][:, :1] * mask_list[i] + disp_list[i][:, 1:2] * (1 - mask_list[i])
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
        return disp_list, merged