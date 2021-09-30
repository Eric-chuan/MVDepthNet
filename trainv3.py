import time
import shutil
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import os
from torch.autograd import Variable
from torch import Tensor
from torch.utils.data import dataset
from torch.utils.tensorboard import SummaryWriter

from model.utils import AverageMeter
from model.depthNet_loss import *
from model.multiscaleloss import *
from Dataloader.room100_loader import *
from Dataloader.kitti_loader import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
# from Dataloader.tum_loader import TumDataset
from model.depthNetv3 import depthNet
from model.laplacian import *
from model.loss import *
from logger_utils import *
import warnings
warnings.filterwarnings("ignore")

# os.environ['CUDA_VISIBLE_DEVICES']='1'

tensorboard_path = 'train_log/tensorboard/'
model_save_path = 'train_log/models/'
epoch_num = 0
iterate_num = 0
iterate_per_epoch = 0
mini_batch_scale = 1  #so we have a minn-batch of 8 * 1 = 1
best_loss_train = -1
best_loss_validate = -1
resume_train = True
resume_train_path = 'train_log/models/checkpoint00100.pth.tar'
initialize_train = False
initialize_train_path = ''
lap_loss = LapLoss()
ssim_loss = SSIM()




def loss_v3(pred_imgs, pred_depths, gt_img):
    loss_lap1 = lap_loss(pred_imgs[0], gt_img).mean()
    loss_lap2 = lap_loss(pred_imgs[1], gt_img).mean()
    loss_lap3 = lap_loss(pred_imgs[2], gt_img).mean()
    loss_ssim1 = ssim_loss(pred_imgs[0], gt_img).mean()
    loss_ssim2 = ssim_loss(pred_imgs[1], gt_img).mean()
    loss_ssim3 = ssim_loss(pred_imgs[2], gt_img).mean()

    for i in range(3):
        pred_depths[i] = dequantilization(pred_depths[i])
    smooth_loss1 = depth_smoothness(pred_depths[0], gt_img)
    smooth_loss2 = depth_smoothness(pred_depths[1], gt_img)
    smooth_loss3 = depth_smoothness(pred_depths[2], gt_img)
    loss = 0.15*(loss_lap1+loss_lap2+loss_lap3) \
           + 0.85*(loss_ssim1+loss_ssim2+loss_ssim3)\
          + 0.5*(smooth_loss1+smooth_loss2+smooth_loss2)
    return loss


def get_learning_rate(step):
    global epoch_num, iterate_per_epoch
    if step < 1000:
        mul = step / 1000.
    else:
        mul = np.cos((step - 1000) / (iterate_per_epoch * 1000 - 1000.) * math.pi) * 0.5 + 0.5
    return (1e-5 - 1e-6) * mul + 1e-6

def main():
    global tensorboard_path, model_save_path
    global epoch_num, iterate_num, best_loss_train, best_loss_validate

    dataset_root = '/home/Eric-chuan/workspace/dataset/room100/pms_triplet/sequences/00001'
    dataset_train = Room100Dataset(dataset_root, 'train')
    train_loader = DataLoader(dataset_train, batch_size=32, num_workers=4, pin_memory=True, drop_last=True)
    dataset_val = Room100Dataset(dataset_root, 'valid')
    validate_loader = DataLoader(dataset_val, batch_size=16, pin_memory=True, num_workers=4)

    # dataset_root = '/home/Eric-chuan/workspace/dataset/dataset_kitti'
    # dataset_train = KittiDataset(dataset_root, 'train')
    # train_loader = DataLoader(dataset_train, batch_size=32, num_workers=4, pin_memory=True, drop_last=True)
    # dataset_val = KittiDataset(dataset_root, 'valid')
    # validate_loader = DataLoader(dataset_val, batch_size=16, pin_memory=True, num_workers=4)

    train_writer = SummaryWriter(tensorboard_path)
    print('train data have %d pairs' % len(train_loader))
    print('validate data have %d pairs' % len(validate_loader))

    cudnn.benchmark = True
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # model
    depthnet = depthNet()
    depthnet =  torch.nn.DataParallel(depthnet)
    depthnet = depthnet.cuda()
    if resume_train:
        pretrained_data = torch.load(resume_train_path)
        depthnet.load_state_dict(pretrained_data['state_dict'])
        epoch_num = pretrained_data['epoch_num']
        iterate_num = pretrained_data['iterate_num']
        best_loss_train = pretrained_data['best_loss_train']
        best_loss_validate = pretrained_data['best_loss_validate']
        print(
            'we start training from epoch %d, iteration %d, the best loss in training is %f and in validation is %f'
            % (epoch_num, iterate_num, best_loss_train, best_loss_validate))

    if (not resume_train) and initialize_train:
        pretrained_data = torch.load(initialize_train_path)
        depthnet.load_state_dict(pretrained_data['state_dict'])
        print('we inittialize training from epoch %d, iteration %d' %
              (epoch_num, pretrained_data['iterate_num']))

    # optimizer
    # optimizer = torch.optim.Adam(depthnet.parameters(), lr= 1e-6)
    optimizer = torch.optim.AdamW(depthnet.parameters(), lr=1e-5, weight_decay=1e-4)
    optimizer.zero_grad()


    #start the epoch
    while (epoch_num < 1000):

        loss_train = train_one_epoch(depthnet, train_loader, optimizer,
                                     train_writer)
        loss_validate = vaild_one_epoch(depthnet, validate_loader,
                                        train_writer)

        #save the checkpoint
        if (epoch_num % 10 == 0):
            checkpoint_name = model_save_path + 'checkpoint' + '%05d' % epoch_num + '.pth.tar'
            torch.save(get_state(depthnet), checkpoint_name)
        if best_loss_train < 0 or loss_train < best_loss_train:
            shutil.copyfile(checkpoint_name,
                            model_save_path + 'best_train.pth.tar')
            best_loss_train = loss_train
        if best_loss_validate < 0 or loss_validate < best_loss_validate:
            shutil.copyfile(checkpoint_name,
                            model_save_path + 'best_validate.pth.tar')
            best_loss_validate = loss_validate

        # update the index for next update
        epoch_num = epoch_num + 1
        # break


def train_one_epoch(model, train_loader, optimizer, data_writer):
    global tensorboard_path, traindata_file, test_file, model_save_path
    global epoch_num, iterate_num, best_loss_train, best_loss_validate, mini_batch_scale, iterate_per_epoch

    average_iterate = AverageMeter()
    average_loss = AverageMeter()
    average_minibatch_loss = AverageMeter()
    average_loadtime = AverageMeter()
    average_forwardtime = AverageMeter()
    average_optimizetime = AverageMeter()
    average_losstime = AverageMeter()
    data_len = len(train_loader)
    iterate_per_epoch = data_len
    train_10k = False
    if len(train_loader) > 10000 * mini_batch_scale:
        data_len = 10000 * mini_batch_scale
        train_10k = True
        print('one epoch too big! try 10k steps!')
    data_loadtime = time.time()
    iterate_begin = time.time()
    model.train()

    for i_batch, sample in enumerate(train_loader):

        # get data
        left_image_cuda = sample['left_img'].cuda()
        right_image_cuda = sample['right_img'].cuda()
        gt_image_cuda = sample['gt_img'].cuda()
        left_proj_cuda = sample['left_para'].cuda()
        right_proj_cuda = sample['right_para'].cuda()
        gt_proj_cuda = sample['gt_para'].cuda()
        depth_image_cuda = sample['depth'].cuda()

        left_image_cuda = Variable(left_image_cuda)
        right_image_cuda = Variable(right_image_cuda)
        gt_image_cuda = Variable(gt_image_cuda)
        depth_image_cuda = Variable(depth_image_cuda)
        # left_proj_cuda = Variable(left_proj_cuda)
        # right_proj_cuda = Variable(right_proj_cuda)
        # gt_proj_cuda = Variable(gt_proj_cuda)

        # time
        average_loadtime.update(time.time() - data_loadtime)

        learning_rate = get_learning_rate(iterate_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        # forward once
        forward_begin_time = time.time()
        predict_imgs, predict_depths, masks = model(left_image_cuda, right_image_cuda,
                               left_proj_cuda, right_proj_cuda, gt_proj_cuda)
        average_forwardtime.update(time.time() - forward_begin_time)
        # get the loss
        loss_begin_time = time.time()
        loss = loss_v3(predict_imgs, predict_depths, gt_image_cuda)
        # loss = build_v3soothloss_with_mask(predict_depths, depth_image_cuda)
        loss_float = loss.item()
        loss = loss / mini_batch_scale
        average_losstime.update(time.time() - loss_begin_time)
        average_loss.update(loss_float, left_image_cuda.size()[0])
        average_minibatch_loss.update(loss_float, left_image_cuda.size()[0])

        # update
        optimize_begin = time.time()
        loss.backward()
        if iterate_num % mini_batch_scale == 0:
            optimizer.step()
            optimizer.zero_grad()
        average_optimizetime.update(time.time() - optimize_begin)

        # print at each mini-batch
        if iterate_num % mini_batch_scale == 0:
            data_writer.add_scalar('train/train_loss', loss_float,
                                   iterate_num / mini_batch_scale)
            data_writer.add_scalar('train/learning_rate', learning_rate, iterate_num / mini_batch_scale)
            print(
                'train: epoch %04d, iterate %07d, epoch process %03.2f%%, loss is %3.8f, average: load %.2fs, forward %.2fs, loss %.2fs, optimize %.2fs, total %.2fs'
                % (epoch_num, iterate_num / mini_batch_scale,
                   float(i_batch) / data_len * 100.0,
                   average_minibatch_loss.avg, average_loadtime.avg,
                   average_forwardtime.avg, average_losstime.avg,
                   average_optimizetime.avg, average_iterate.avg))
            average_minibatch_loss.reset()
            average_loadtime.reset()
            average_forwardtime.reset()
            average_optimizetime.reset()
            average_losstime.reset()
            average_iterate.reset()
        if iterate_num % 5 == 0:
            print(predict_depths[0][-1].max(), predict_depths[0][-1].min())
            PredictDepth1 = ((predict_depths[0][-1] * 255.).detach().cpu().numpy()).astype('uint8')
            GTDepth = quantilization(depth_image_cuda[-1].detach().cpu().numpy()).astype('uint8')
            PredictImage1 =  (predict_imgs[0][-1].detach().cpu().numpy()[[2, 1, 0], :, :] * 255).astype('uint8')
            GTImage = (gt_image_cuda[-1].detach().cpu().numpy()[[2, 1, 0], :, :] * 255).astype('uint8')
            mask = ((masks[0][-1] * 255.).detach().cpu().numpy()).astype('uint8')
            depths = np.concatenate((GTDepth, PredictDepth1), axis=1)
            images = np.concatenate((GTImage, PredictImage1), axis=1)
            data_writer.add_image('train/Depth', depths, iterate_num)
            data_writer.add_image('train/Image', images, iterate_num)
            data_writer.add_image('train/Mask', mask, iterate_num)

        # if iterate_num % 10 == 0:
        #     cv2.imwrite('DepthGT.png', quantilization(depth_image_cuda[-1].detach().cpu().numpy().transpose(1, 2, 0)).astype('uint8'))
        #     print('depth shape==========', depth_image_cuda[-1].detach().cpu().numpy().transpose(1, 2, 0).shape)

        iterate_num = iterate_num + 1

        # prepare for the next iterate
        average_iterate.update(time.time() - iterate_begin)
        data_loadtime = time.time()
        iterate_begin = time.time()

        if train_10k and i_batch == data_len:
            break

    return average_loss.avg


def vaild_one_epoch(model, vaild_loader, data_writer):
    global tensorboard_path, traindata_file, test_file, model_save_path
    global epoch_num, iterate_num, best_loss_train, best_loss_validate

    validate_iterate = 0
    average_loss = AverageMeter()
    average_loadtime = AverageMeter()
    average_forwardtime = AverageMeter()
    data_len = len(vaild_loader)
    data_loadtime = time.time()
    model.eval()
    loss_vector = []

    for i_batch, sample in enumerate(vaild_loader):

        # get data
        left_image_cuda = sample['left_img'].cuda()
        right_image_cuda = sample['right_img'].cuda()
        gt_image_cuda = sample['gt_img'].cuda()
        left_proj_cuda = sample['left_para'].cuda()
        right_proj_cuda = sample['right_para'].cuda()
        gt_proj_cuda = sample['gt_para'].cuda()
        depth_image_cuda = sample['depth'].cuda()

        left_image_cuda = Variable(left_image_cuda, volatile=True)
        right_image_cuda = Variable(right_image_cuda, volatile=True)
        gt_image_cuda = Variable(gt_image_cuda, volatile=True)
        depth_image_cuda = Variable(depth_image_cuda, volatile=True)
        # left_proj_cuda = Variable(left_proj_cuda, volatile=True)
        # right_proj_cuda = Variable(right_proj_cuda, volatile=True)
        # gt_proj_cuda = Variable(gt_proj_cuda, volatile=True)


        # time
        average_loadtime.update(time.time() - data_loadtime)

        # forward once
        forward_begin_time = time.time()
        predict_imgs, predict_depths, masks = model(left_image_cuda, right_image_cuda,
                               left_proj_cuda, right_proj_cuda, gt_proj_cuda)
        forward_time = time.time() - forward_begin_time
        average_forwardtime.update(forward_time)

        # get the loss
        # loss = build_v3loss_with_mask(predict_depths, depth_image_cuda)
        loss = loss_v3(predict_imgs, predict_depths, gt_image_cuda)
        loss_float = loss.item()

        loss_vector.append(loss_float)
        average_loss.update(loss_float, left_image_cuda.size()[0])
        psnr_list = []
        for j in range(gt_image_cuda.shape[0]):
            psnr = -10 * math.log10(torch.mean((gt_image_cuda[j] - predict_imgs[0][j]) * (gt_image_cuda[j] - predict_imgs[0][j])).cpu().data)
            psnr_list.append(psnr)

        # print
        if validate_iterate % 10 == 0:
            print(
                'validate: iterate %07d, batch process %03.2f%%, train loss is %3.3f, average load time %.2f ms, average forward time %.2f ms'
                % (validate_iterate, float(i_batch) / data_len * 100.0,
                   loss_float, average_loadtime.avg * 1000.0,
                   average_forwardtime.avg * 1000.0))
            average_loadtime.reset()
            average_forwardtime.reset()

        validate_iterate += 1
        # prepare for the next iterate
        data_loadtime = time.time()

    data_writer.add_scalar('valid/validate_loss', average_loss.avg,
                           epoch_num)
    data_writer.add_scalar('valid/psnr', np.array(psnr_list).mean(), epoch_num)
    # data_writer.add_histogram(
    #     'validate_loss',
    #     np.asarray(loss_vector),
    #     global_step=epoch_num,
    #     bins=np.arange(0.00, 4.00, 0.0001))
    print(np.asarray(loss_vector))
    return average_loss.avg


# set the learning rate
def learning_rate_set(optimizer):
    global tensorboard_path, traindata_file, test_file, model_save_path
    global epoch_num, iterate_num, best_loss_train, best_loss_validate

    pass


# get the current train state for save
def get_state(model):
    global tensorboard_path, traindata_file, test_file, model_save_path
    global epoch_num, iterate_num, best_loss_train, best_loss_validate

    return {
        'epoch_num': epoch_num + 1,
        'iterate_num': iterate_num,
        'best_loss_train': best_loss_train,
        'best_loss_validate': best_loss_validate,
        'state_dict': model.state_dict()
    }

def quantilization(depth):
    depth_min = 10.0
    depth_max = 40.0
    # depth = depth.float()
    vaild_depth_mask = depth != 0
    depth[vaild_depth_mask] = ((1/depth[vaild_depth_mask]) - (1/depth_max)) * 255. / ((1/depth_min)-(1/depth_max))
    return depth

def dequantilization(depth):
    depth_min = 10.0
    depth_max = 40.0000
    depth = (depth) * ((32504.0 / depth_min) - (32504.0 / depth_max)) + (32504.0 / depth_max)
    mask = depth != 0
    depth[mask] = 32504.0 / depth[mask]
    return depth



if __name__ == '__main__':
    main()

