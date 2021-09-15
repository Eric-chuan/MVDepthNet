import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from numpy.linalg import inv
from .module import image_differentiable_warping, feature_differentiable_warping

def conv_layer(in_planes, out_planes, kernel_size=3, stride=1, batchNorm=False):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

def deconv_layer(in_planes, out_planes, kernel_size=4, stride=2, padding=1, batchNorm=False):
    if batchNorm:
        return nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1,inplace=True)
        )


def down_conv_layer(in_planes, out_planes, kernel_size=3, batchNorm=False):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=1,
                bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(
                out_planes,
                out_planes,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=2,
                bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=1,
                bias=False),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(
                out_planes,
                out_planes,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=2,
                bias=False),
            nn.LeakyReLU(0.1,inplace=True))

def up_conv_layer(in_planes, out_planes, kernel_size=3, batchNorm=False):
    if batchNorm:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True))
    else:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=False),
            nn.LeakyReLU(0.1,inplace=True))

def depth_layer(input_channels):
    return nn.Sequential(
        nn.Conv2d(input_channels, 1, 3, padding=1), nn.Sigmoid())

def predict_flow(in_planes, out_planes = 1):
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=1,padding=1,bias=False)

def refine_layer(input_channels):
    return nn.Conv2d(input_channels, 1, 3, padding=1)

def deconv_MFN(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

def get_trainable_number(variable):
    num = 1
    shape = list(variable.shape)
    for i in shape:
        num *= i
    return num

class IDepthBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IDepthBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv_layer(in_planes, c//2, 3, 2, 1),
            conv_layer(c//2, c, 3, 2, 1),
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
        self.lastconv = nn.ConvTranspose2d(c, 3, 4, 2, 1)   #out_planes depth[2] + mask[1]

    def forward(self, x, disp, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
        if disp != None:
            disp = F.interpolate(disp, scale_factor = 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, disp), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor = scale * 2, mode="bilinear", align_corners=False)
        disp = tmp[:, :2] * scale * 2
        mask = tmp[:, 2:3]
        return disp, mask

class ResBlock(nn.Module):
    def __init__(self, n_in, n_out, stride = 1, deform = False):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size = 3, padding = 1)

        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace = True)
        self.bn2 = nn.BatchNorm2d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv2d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out

class GetImgVolume(nn.Module):
    """Initialization Stage Class"""

    def __init__(self, num_sample: int = 1) -> None:
        """Initialize method

        Args:
            num_sample: number of samples used in patchmatch process
        """
        super(GetImgVolume, self).__init__()
        self.num_sample = num_sample

    def forward(
        self,
        left_img: torch.Tensor,
        right_img: torch.Tensor,
        left_proj: torch.Tensor,
        right_proj: torch.Tensor,
        gt_proj: torch.Tensor,
        device: torch.device,
        depth_sample=None
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
        batch_size, channels, height, width = left_img.size()
            # first iteration of Patchmatch on stage 3, sample in the inverse depth range
            # divide the range into several intervals and sample in each of them
        if (depth_sample == None):
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
        else:
            depth_sample = 1.0 / depth_sample  # [B,Ndepth,H,W]

        warped_left = image_differentiable_warping(left_img, left_proj, gt_proj, depth_sample) #(B, 3, Ndepth, H, W)
        warped_right = image_differentiable_warping(right_img, right_proj, gt_proj, depth_sample) #(B, 3, Ndepth, H, W)

        costvolume = torch.sum(torch.abs(warped_right - warped_left), dim=1)  #(B, Ndepth, H, W)

        return costvolume

class GetImgCorrelation(nn.Module):
    """Initialization Stage Class"""

    def __init__(self, num_sample: int = 1) -> None:
        """Initialize method

        Args:
            num_sample: number of samples used in patchmatch process
        """
        super(GetImgCorrelation, self).__init__()
        self.num_sample = num_sample

    def forward(
        self,
        left_img: torch.Tensor,
        right_img: torch.Tensor,
        left_proj: torch.Tensor,
        right_proj: torch.Tensor,
        gt_proj: torch.Tensor,
        device: torch.device,
        depth_sample = None
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
        if (depth_sample == None):
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
        else:
            depth_sample = 1.0 / depth_sample  # [B,Ndepth,H,W]

        warped_left = image_differentiable_warping(left_img, left_proj, gt_proj, depth_sample)    #(B, 3, Ndepth, H, W)
        warped_right = image_differentiable_warping(right_img, right_proj, gt_proj, depth_sample) #(B, 3, Ndepth, H, W)

        correaltion = torch.mean((warped_right * warped_left), dim=1)  #(B, Ndepth, H, W)

        costvolume = torch.sum(torch.abs(warped_right - warped_left), dim=1)  #(B, Ndepth, H, W)

        return costvolume


class GetFeatureVolume(nn.Module):
    """Initialization Stage Class"""

    def __init__(self, num_sample: int = 1) -> None:
        """Initialize method

        Args:
            num_sample: number of samples used in patchmatch process
        """
        super(GetFeatureVolume, self).__init__()
        self.num_sample = num_sample

    def forward(
        self,
        left_fea: torch.Tensor,
        right_fea: torch.Tensor,
        left_proj: torch.Tensor,
        right_proj: torch.Tensor,
        gt_proj: torch.Tensor,
        device: torch.device,
        depth_sample = None
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
        batch_size, _, height, width = left_fea.size()
            # first iteration of Patchmatch on stage 3, sample in the inverse depth range
            # divide the range into several intervals and sample in each of them
        if (depth_sample == None):
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
        else:
            depth_sample = 1.0 / depth_sample  # [B,Ndepth,H,W]

        warped_left = feature_differentiable_warping(left_fea, left_proj, gt_proj, depth_sample) #(B, C, Ndepth, H, W)
        warped_right = feature_differentiable_warping(right_fea, right_proj, gt_proj, depth_sample) #(B, C, Ndepth, H, W)

        costvolume = torch.sum(torch.abs(warped_right - warped_left), dim=1)  #(B, Ndepth, H, W)

        return costvolume


class GetFeatureCorrelation(nn.Module):
    """Initialization Stage Class"""

    def __init__(self, num_sample: int = 1) -> None:
        """Initialize method

        Args:
            num_sample: number of samples used in patchmatch process
        """
        super(GetFeatureCorrelation, self).__init__()
        self.num_sample = num_sample

    def forward(
        self,
        left_fea: torch.Tensor,
        right_fea: torch.Tensor,
        left_proj: torch.Tensor,
        right_proj: torch.Tensor,
        gt_proj: torch.Tensor,
        device: torch.device,
        depth_sample=None
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
        batch_size, _, height, width = left_fea.size()
            # first iteration of Patchmatch on stage 3, sample in the inverse depth range
            # divide the range into several intervals and sample in each of them
        if (depth_sample == None):
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
        else:
            depth_sample = 1.0 / depth_sample  # [B,Ndepth,H,W]

        warped_left = feature_differentiable_warping(left_fea, left_proj, gt_proj, depth_sample) #(B, C, Ndepth, H, W)
        warped_right = feature_differentiable_warping(right_fea, right_proj, gt_proj, depth_sample) #(B, C, Ndepth, H, W)

        correaltion = torch.mean((warped_right * warped_left), dim=1)  #(B, Ndepth, H, W)

        return correaltion