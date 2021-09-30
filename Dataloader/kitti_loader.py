# -*- coding: utf-8 -*-
'''
a dataset loader for synthetic depth dataset
'''
from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from imgaug import augmenters as iaa
import cv2
from numpy.linalg import inv
import time
from torch.autograd import Variable


class KittiDataset(Dataset):
    """
	the class to load the image from room100 dataset
	"""
    def __init__(self, dataroot, category='train', use_augment=True):
        super(KittiDataset, self).__init__()
        # find all the data sets
        self.dataroot = dataroot
        self.category = category
        self.build_metas()

    def build_metas(self):
        with open(f"{self.dataroot}/trainlist.txt", "r") as f1:
            self.train_path = f1.readlines()
        with open(f"{self.dataroot}/testlist.txt", "r") as f2:
            self.val_path = f2.readlines()

        if self.category == 'train':
            self.meta_data = self.train_path
        else:
            self.meta_data = self.val_path


    def __len__(self):
        return len(self.meta_data)

    def dequantilize(self, depth):
        depth_min = 10.0
        depth_max = 40.0
        mask0 = depth == 0
        mask1 = depth != 0
        depth[mask1] = (depth[mask1] / 255.) * ((32504.0 / depth_min) - (32504.0 / depth_max)) + (32504.0 / depth_max)
        depth[mask1] = 32504.0 / depth[mask1]
        depth[mask0] = depth_min
        return depth

    def read_data(self, np_file):
        # scale 0~255 to 0~1
        i0i1gt = np_file['i0i1gt']
        i0i1gt = i0i1gt.astype(np.float32) / 255.
        left_img, right_img, gt_img = i0i1gt[0:3], i0i1gt[3:6], i0i1gt[6:9]
        _, original_h, original_w = left_img.shape

        # depth = np_file['depth']
        # depth = depth.astype(np.float32)
        # depth = self.dequantilize(depth)

        # scale_x = 320.0 / original_w
        # scale_y = 256.0 / original_h
        # left_img = cv2.resize(left_img.transpose(1, 2, 0), (320, 256))
        # right_img = cv2.resize(right_img.transpose(1, 2, 0), (320, 256))
        # gt_img = cv2.resize(gt_img.transpose(1, 2, 0), (320, 256))
        # depth = cv2.resize(depth, (320, 256))

        left_img = left_img.transpose(1, 2, 0)
        right_img = right_img.transpose(1, 2, 0)
        gt_img = gt_img.transpose(1, 2, 0)

        extrinsics = np_file['extrinsic']
        intrinsics = np_file['intrinsic']
        left_extrinsic, right_extrinsic, gt_extrinsic = extrinsics[0], extrinsics[1], extrinsics[2]
        left_intrinsic, right_intrinsic, gt_intrinsic = intrinsics[0], intrinsics[1], intrinsics[2]

        return left_img, right_img, gt_img, left_extrinsic, left_intrinsic, gt_intrinsic, right_extrinsic, right_intrinsic, gt_extrinsic

    def __getitem__(self, index):
        # get the sequence index
        data_path = f'{self.dataroot}/sequences/{self.meta_data[index].strip()}/data.npz'
        np_file = np.load(data_path)
        left_img, right_img, gt_img, \
        left_extrinsic, left_intrinsic, gt_intrinsic, right_extrinsic, right_intrinsic, gt_extrinsic = self.read_data(np_file)


        left_img = torch.from_numpy(left_img.copy()).permute(2, 0, 1)   #CxHxW
        right_img = torch.from_numpy(right_img.copy()).permute(2, 0, 1)
        gt_img = torch.from_numpy(gt_img.copy()).permute(2, 0, 1)

        left_para = np.concatenate((left_extrinsic[:3], left_intrinsic), axis=1)    #3x7  R|T|K
        right_para = np.concatenate((right_extrinsic[:3], right_intrinsic), axis=1)
        gt_para = np.concatenate((gt_extrinsic[:3], gt_intrinsic), axis=1)
        left_para = torch.from_numpy(left_para)    #3x7
        right_para = torch.from_numpy(right_para)
        gt_para = torch.from_numpy(gt_para)

        #return the sample
        return {
            'left_img': left_img,
            'right_img': right_img,
            'gt_img': gt_img,
            'left_para': left_para,
            'right_para': right_para,
            'gt_para': gt_para,
        }


if __name__ == '__main__':
    dataset_root = '/home/Eric-chuan/workspace/dataset/dataset_kitti'
    dataset_train = KittiDataset(dataset_root, 'train')
    data_dict = dataset_train[1]
    left_img = data_dict['left_img']
    # depthGT = data_dict['depth']
    # depth_image_cuda = (depthGT)
    print(left_img.shape)
    # def quantilization(depth):
    #     depth_min = 10.0
    #     depth_max = 40.0
    #     vaild_depth_mask = depth != 0.
    #     depth[vaild_depth_mask] = ((1/depth[vaild_depth_mask]) - (1/depth_max)) * 255. / ((1/depth_min)-(1/depth_max))
    #     return depth.astype('uint8')

    cv2.imwrite('left.png', (left_img.numpy().transpose(1, 2, 0) * 255))

    # data_path = f'{dataset_root}/0568_004/data.npz'
    # np_file = np.load(data_path)
    # i0i1gt = np_file['i0i1gt']
    # i0i1gt = i0i1gt.astype(np.float32) / 255.
    # left_img, right_img = i0i1gt[6:9], i0i1gt[3:6]
    # depth = np_file['depth']
    # depth = torch.from_numpy(depth.copy()).unsqueeze(0)
    # depth = depth.numpy().transpose(1, 2, 0)
    # print(depth.max(), depth.min())
    # depth_min = 10.0
    # depth_max = 40.0
    # mask = (depth != 0)
    # depth = depth.astype(np.float32)
    # depth[mask] = (depth[mask] / 255.) * ((32504.0 / depth_min) - (32504.0 / depth_max)) + (32504.0 / depth_max)
    # depth[mask] = 32504.0 / depth[mask]
    # # depth = np.ones((1080, 1920)) * 500
    # print(depth.max(), depth.min())
    # # depth = cv2.resize(, (320, 256))
    # print(depth.shape)
    # cv2.imwrite('DepthGT.png', (depth))

    # def dequantilize(depth):
    #     depth_min = 10.0
    #     depth_max = 40.0
    #     mask = (depth != 0)
    #     depth = depth.float()
    #     print(mask, mask.shape, mask.dtype, depth.dtype)
    #     depth[mask] = (depth[mask] / 255.) #* ((32504.0 / depth_min) - (32504.0 / depth_max)) + (32504.0 / depth_max)
    #     depth[mask] = 32504.0 / depth[mask]
    #     return depth

    # depth = dequantilize(depth)

    # def img2show(image):
    #     float_img = image.astype(float)
    #     print('max %f, min %f' % (float_img.max(), float_img.min()))
    #     float_img = (float_img - float_img.min()) / (
    #         float_img.max() - float_img.min()) * 255.0
    #     uint8_img = float_img.astype(np.uint8)
    #     return uint8_img

    # from torch.autograd import Variable
    # from torch import Tensor
    # import torch.nn.functional as F

    # loader = Room100Dataset('/hdd_data/datasets/DeMoN/train_data/')
    # train_loader = torch.utils.data.DataLoader(
    #     loader, batch_size=2, shuffle=True, num_workers=4)
    # begin_time = time.time()
    # for i_batch, sample in enumerate(train_loader):

    #     left_img = np.array(sample['left_img'])
    #     right_img = np.array(sample['right_img'])

    #     left_image_cuda = sample['left_img'].cuda()
    #     right_image_cuda = sample['right_img'].cuda()
    #     KRKiUV_cuda_T = sample['KRKiUV'].cuda()
    #     KT_cuda_T = sample['KT'].cuda()
    #     depth_image_cuda = sample['depth_image'].cuda()

    #     left_image_cuda = Variable(left_image_cuda, volatile=True)
    #     right_image_cuda = Variable(right_image_cuda, volatile=True)
    #     depth_image_cuda = Variable(depth_image_cuda, volatile=True)

    #     idepth_base = 1.0 / 50.0
    #     idepth_step = (1.0 / 0.5 - 1.0 / 50.0) / 63.0
    #     costvolume = Variable(
    #         torch.FloatTensor(left_img.shape[0], 64, left_img.shape[2],
    #                           left_img.shape[3]))
    #     image_height = 256
    #     image_width = 320
    #     batch_number = left_img.shape[0]

    #     normalize_base = torch.FloatTensor(
    #         [image_width / 2.0, image_height / 2.0])
    #     normalize_base = normalize_base.unsqueeze(0).unsqueeze(-1)
    #     normalize_base_v = Variable(normalize_base)

    #     KRKiUV_v = Variable(sample['KRKiUV'])
    #     KT_v = Variable(sample['KT'])
    #     for depth_i in range(64):
    #         this_depth = 1.0 / (idepth_base + depth_i * idepth_step)
    #         transformed = KRKiUV_v * this_depth + KT_v
    #         warp_uv = transformed[:, 0:2, :] / transformed[:, 2, :].unsqueeze(
    #             1)  #shape = batch x 2 x 81920
    #         warp_uv = (warp_uv - normalize_base_v) / normalize_base_v
    #         warp_uv = warp_uv.view(
    #             batch_number, 2, image_width,
    #             image_height)  #shape = batch x 2 x width x height

    #         warp_uv = warp_uv.permute(0, 3, 2,
    #                                   1)  #shape = batch x height x width x 2
    #         right_image_v = Variable(sample['right_img'])
    #         warped = F.grid_sample(right_image_v, warp_uv)
    #         costvolume[:, depth_i, :, :] = torch.sum(
    #             torch.abs(warped - Variable(sample['left_img'])),
    #             dim=1)

    #     costvolume = F.avg_pool2d(
    #         costvolume,
    #         5,
    #         stride=1,
    #         padding=2,
    #         ceil_mode=False,
    #         count_include_pad=True)
    #     np_cost = costvolume.data.numpy()
    #     winner_takes_all = np.argmin(np_cost[1, :, :, :], axis=0)
    #     print(winner_takes_all.shape)

    #     cv2.imshow('left_img',
    #                img2show(np.moveaxis(left_img[1, :, :, :], 0, -1)))
    #     cv2.imshow('right_img',
    #                img2show(np.moveaxis(right_img[1, :, :, :], 0, -1)))
    #     cv2.imshow('depth_image', img2show(winner_takes_all))

    #     if cv2.waitKey(0) == 27:
    #         break
