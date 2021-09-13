import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor


def down_sample(depth_image):
    return depth_image[:, :, ::2, ::2]


def build_loss(predict_depth, groundtruth_depth):

    # clamp depth_image between 0~50.0m and get inverse depth
    depth_image = groundtruth_depth.clamp(0.5, 50.0)
    depth_image = 1.0 / depth_image

    # get multi resolution depth maps
    depth_image1 = depth_image
    depth_image2 = down_sample(depth_image1)
    depth_image3 = down_sample(depth_image2)
    depth_image4 = down_sample(depth_image3)

    # build depth image loss
    loss_depth1 = F.l1_loss(predict_depth[0], depth_image)
    loss_depth2 = F.l1_loss(predict_depth[1], depth_image2)
    loss_depth3 = F.l1_loss(predict_depth[2], depth_image3)
    loss_depth4 = F.l1_loss(predict_depth[3], depth_image4)
    return loss_depth1 + loss_depth2 + loss_depth3 + loss_depth4


def build_loss_with_mask(predict_depth, groundtruth_depth):

    # clamp depth_image between 0~50.0m and get inverse depth
    vaild_depth_mask = groundtruth_depth > 0.1
    depth_image = groundtruth_depth.clamp(0.5, 50.0)
    depth_image = 1.0 / depth_image

    # get multi resolution depth maps
    depth_image1 = depth_image
    depth_image2 = down_sample(depth_image1)
    depth_image3 = down_sample(depth_image2)
    depth_image4 = down_sample(depth_image3)

    depth_mask1 = vaild_depth_mask
    depth_mask2 = down_sample(depth_mask1)
    depth_mask3 = down_sample(depth_mask2)
    depth_mask4 = down_sample(depth_mask3)

    # build depth image loss
    loss_depth1 = torch.abs(predict_depth[0] - depth_image1)
    loss_depth2 = torch.abs(predict_depth[1] - depth_image2)
    loss_depth3 = torch.abs(predict_depth[2] - depth_image3)
    loss_depth4 = torch.abs(predict_depth[3] - depth_image4)
    return loss_depth1[depth_mask1].mean() + loss_depth2[depth_mask2].mean(
    ) + loss_depth3[depth_mask3].mean() + loss_depth4[depth_mask4].mean()


def final_loss(predict_depth, groundtruth_depth):
    # clamp depth_image between 0~50.0m and get inverse depth
    vaild_depth_mask = groundtruth_depth > 0.1
    depth_image = groundtruth_depth.clamp(0.5, 50.0)
    depth_image = 1.0 / depth_image

    depth_mask1 = vaild_depth_mask

    # build depth image loss
    loss_depth1 = torch.abs(predict_depth - depth_image)
    return loss_depth1[vaild_depth_mask].mean().data[0]