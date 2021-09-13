# -*- coding: utf-8 -*-
'''
a dataset loader for synthetic depth dataset
'''
from __future__ import print_function, division
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from imgaug import augmenters as iaa
import cv2
import pickle
from numpy.linalg import inv
from os import listdir
import time


class Sun3dDataset(Dataset):
    """
	the class to load the image from tum or SceneNN dataset
	"""

    def __init__(self, root_path, use_augment=True):
        super(Sun3dDataset, self).__init__()
        # find all the data sets
        self.sequence_info = []
        self.sequence_dir = []
        self.accumulate_num = []
        self.all_datapairs = 0
        files = listdir(root_path)
        for file_name in files:
            if file_name.endswith('.pkl'):
                print(file_name)
                self.sequence_dir.append(root_path + file_name[:-4] + '/')
                with open(root_path + file_name, 'rb') as input_file:
                    info_data = pickle.load(input_file)
                    self.all_datapairs += len(info_data['pairs'])
                    self.sequence_info.append(info_data)
                    self.accumulate_num.append(self.all_datapairs)
        # print(self.accumulate_num)

        #data augment
        self.use_augment = use_augment
        if self.use_augment:
            self.img_aug = iaa.SomeOf(
                (0, 2),
                [
                    iaa.AdditiveGaussianNoise(
                        loc=0,
                        scale=(0.0,
                               0.01 * 255)),  # add gaussian noise to images
                    iaa.ContrastNormalization(
                        (0.5, 2.0),
                        per_channel=0.5),  # improve or worsen the contrast
                    iaa.Multiply((0.7, 1.3), per_channel=0.5),
                    iaa.Add((-40, 40), per_channel=0.5)
                ],
                random_order=True)

        # for image warp
        self.pixel_coordinate = np.indices([320, 256]).astype(np.float32)
        self.pixel_coordinate = np.concatenate(
            (self.pixel_coordinate, np.ones([1, 320, 256])), axis=0)
        self.pixel_coordinate = np.reshape(self.pixel_coordinate, [3, -1])

    def __len__(self):
        return self.all_datapairs

    def __getitem__(self, idx):
        # get the sequence index
        this_sequence_index = 0
        this_sequence_base = 0
        while self.accumulate_num[this_sequence_index] <= idx:
            this_sequence_base = self.accumulate_num[this_sequence_index]
            this_sequence_index += 1

        this_sequence_info = self.sequence_info[this_sequence_index]
        this_sequence_path = self.sequence_dir[this_sequence_index]
        left_name = this_sequence_info['pairs'][idx
                                                - this_sequence_base]['left']
        right_name = this_sequence_info['pairs'][idx
                                                 - this_sequence_base]['right']

        #read the image and depth
        print(this_sequence_path + left_name + '_image.png')
        try:
            left_image = np.asarray(
                Image.open(this_sequence_path + left_name + '_image.png'),
                dtype=np.float32)
        except:
            print('Sun3dDataset: error read %s!' %
                  (this_sequence_path + left_name + '_image.png'))
            return self.__getitem__(1)

        try:
            right_image = np.asarray(
                Image.open(this_sequence_path + right_name + '_image.png'),
                dtype=np.float32)
        except:
            print('Sun3dDataset: error read %s!' %
                  (this_sequence_path + right_name + '_image.png'))
            return self.__getitem__(1)

        with open(this_sequence_path + left_name + '_info.pkl',
                  'rb') as input_file:
            info_data = pickle.load(input_file)
            depth_image = info_data['depth'].astype(np.float32)
            camera_K = info_data['K'].astype(np.float32)
            left_T = info_data['T'].astype(np.float32)
        with open(this_sequence_path + right_name + '_info.pkl',
                  'rb') as input_file:
            info_data = pickle.load(input_file)
            right_T = info_data['T'].astype(np.float32)

        #resize the image into 320x240
        scale_x = 320.0 / left_image.shape[1]
        scale_y = 256.0 / left_image.shape[0]
        left_image = cv2.resize(left_image, (320, 256))
        right_image = cv2.resize(right_image, (320, 256))
        depth_image = cv2.resize(depth_image, (320, 256))
        camera_K[0, :] = camera_K[0, :] * scale_x
        camera_K[1, :] = camera_K[1, :] * scale_y
        camera_K_inv = inv(camera_K)

        if self.use_augment:
            #add the image together so that it can be augmented together,
            #otherwisw some channel maybe inconsistently augmented
            togetherImage = np.append(left_image, right_image, axis=0)
            auged_together = self.img_aug.augment_image(togetherImage)
            #sperate the image
            width = int(auged_together.shape[0] / 2)
            left_image = auged_together[:width]
            right_image = auged_together[width:]

        #scale the image into 0~1.0 and then normalize
        left_image = (left_image - this_sequence_info['mean']
                      ) / this_sequence_info['std']
        right_image = (right_image - this_sequence_info['mean']
                       ) / this_sequence_info['std']

        left_in_right_Trans = np.dot(inv(right_T), left_T)

        # demon dataset uses a differen representation
        left_in_right_T = left_in_right_Trans[0:3, 3]
        left_in_right_R = left_in_right_Trans[0:3, 0:3]
        KRK_i = camera_K.dot(left_in_right_R.dot(camera_K_inv))
        KT = camera_K.dot(left_in_right_T)
        KT = np.expand_dims(KT, -1)
        KRKiUV = KRK_i.dot(self.pixel_coordinate)

        #the image should be transformed into CxHxW
        left_image = np.moveaxis(left_image, -1, 0)
        right_image = np.moveaxis(right_image, -1, 0)
        depth_image = np.expand_dims(depth_image, 0)

        left_image = left_image.astype(np.float32)
        right_image = right_image.astype(np.float32)
        depth_image = depth_image.astype(np.float32)
        KRKiUV = KRKiUV.astype(np.float32)
        KT = KT.astype(np.float32)

        #return the sample
        return {
            'left_image': left_image,
            'right_image': right_image,
            'depth_image': depth_image,
            'KRKiUV': KRKiUV,
            'KT': KT
        }


if __name__ == '__main__':

    def img2show(image):
        float_img = image.astype(float)
        print('max %f, min %f' % (float_img.max(), float_img.min()))
        float_img = (float_img - float_img.min()) / (
            float_img.max() - float_img.min()) * 255.0
        uint8_img = float_img.astype(np.uint8)
        return uint8_img

    from torch.autograd import Variable
    from torch import Tensor
    import torch.nn.functional as F

    loader = Sun3dDataset('/hdd_data/datasets/DeMoN/train_data/')
    # loader = Sun3dDataset('/home/wang/dataset/SceneNN/dataset_generator/SceneNN_train.pkl')
    # loader = Sun3dDataset('/home/wang/dataset/tum_rgbd/TUM_loader/tum_train.pkl')
    # loader = Sun3dDataset('/home/wang/dataset/tum_rgbd/TUM_loader/tum_validate.pkl')
    train_loader = torch.utils.data.DataLoader(
        loader, batch_size=2, shuffle=True, num_workers=4)
    begin_time = time.time()
    for i_batch, sample_batched in enumerate(train_loader):

        left_image = np.array(sample_batched['left_image'])
        right_image = np.array(sample_batched['right_image'])

        left_image_cuda = sample_batched['left_image'].cuda()
        right_image_cuda = sample_batched['right_image'].cuda()
        KRKiUV_cuda_T = sample_batched['KRKiUV'].cuda()
        KT_cuda_T = sample_batched['KT'].cuda()
        depth_image_cuda = sample_batched['depth_image'].cuda()

        left_image_cuda = Variable(left_image_cuda, volatile=True)
        right_image_cuda = Variable(right_image_cuda, volatile=True)
        depth_image_cuda = Variable(depth_image_cuda, volatile=True)

        idepth_base = 1.0 / 50.0
        idepth_step = (1.0 / 0.5 - 1.0 / 50.0) / 63.0
        costvolume = Variable(
            torch.FloatTensor(left_image.shape[0], 64, left_image.shape[2],
                              left_image.shape[3]))
        image_height = 256
        image_width = 320
        batch_number = left_image.shape[0]

        normalize_base = torch.FloatTensor(
            [image_width / 2.0, image_height / 2.0])
        normalize_base = normalize_base.unsqueeze(0).unsqueeze(-1)
        normalize_base_v = Variable(normalize_base)

        KRKiUV_v = Variable(sample_batched['KRKiUV'])
        KT_v = Variable(sample_batched['KT'])
        for depth_i in range(64):
            this_depth = 1.0 / (idepth_base + depth_i * idepth_step)
            transformed = KRKiUV_v * this_depth + KT_v
            warp_uv = transformed[:, 0:2, :] / transformed[:, 2, :].unsqueeze(
                1)  #shape = batch x 2 x 81920
            warp_uv = (warp_uv - normalize_base_v) / normalize_base_v
            warp_uv = warp_uv.view(
                batch_number, 2, image_width,
                image_height)  #shape = batch x 2 x width x height

            warp_uv = warp_uv.permute(0, 3, 2,
                                      1)  #shape = batch x height x width x 2
            right_image_v = Variable(sample_batched['right_image'])
            warped = F.grid_sample(right_image_v, warp_uv)
            costvolume[:, depth_i, :, :] = torch.sum(
                torch.abs(warped - Variable(sample_batched['left_image'])),
                dim=1)

        costvolume = F.avg_pool2d(
            costvolume,
            5,
            stride=1,
            padding=2,
            ceil_mode=False,
            count_include_pad=True)
        np_cost = costvolume.data.numpy()
        winner_takes_all = np.argmin(np_cost[1, :, :, :], axis=0)
        print(winner_takes_all.shape)

        cv2.imshow('left_image',
                   img2show(np.moveaxis(left_image[1, :, :, :], 0, -1)))
        cv2.imshow('right_image',
                   img2show(np.moveaxis(right_image[1, :, :, :], 0, -1)))
        cv2.imshow('depth_image', img2show(winner_takes_all))

        if cv2.waitKey(0) == 27:
            break
