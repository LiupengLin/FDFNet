# python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/10 15:23
# @Author  : Liupeng Lin
# @Email   : linliupeng@whu.edu.cn
# Copyright (C) 2021 Liupeng Lin. All Rights Reserved.


import os
import gc
import re
import time
import glob
import h5py
import torch
import argparse
import warnings
import numpy as np
import torch.optim as optim
from torch import nn
from dataset import *
from fdfnet import FDFNet
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Pytorch FDFNet')
parser.add_argument('--model', default='FDFNet', type=str, help='choose path of model')
parser.add_argument('--batchsize', default=64, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int, help='number of train epoch')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate for Adam')
parser.add_argument('--train_data', default='data/train/train_hv_vv_y4r.h5', type=str, help='path of train data')
parser.add_argument('--nFeat', default=64, type=int, help='the number of feature maps')
parser.add_argument('--nDense', type=int, default=6, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
parser.add_argument('--ncha_dualsar', default=2, type=int, help='the number of SinSAR channels')
parser.add_argument('--ncha_polsar', default=9, type=int, help='the number of PolSAR channels')
parser.add_argument('--ncha_pd', default=4, type=int, help='the number of polarimetric decomposition images channels')
parser.add_argument('--ncha_diff', default=2, type=int, help='the number of differential images channels')
parser.add_argument('--ratio', default=8, type=int, help='the ratio of channel attention module')
parser.add_argument('--gpu', default='0,1', type=str, help='gpu id')
args = parser.parse_args()

cuda = torch.cuda.is_available()
nGPU = torch.cuda.device_count()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
savedir = os.path.join('model', args.model)


log_header = [
    'epoch',
    'iteration',
    'train/loss',
]

if not os.path.exists(savedir):
    os.mkdir(savedir)

if not os.path.exists(os.path.join(savedir, 'log.csv')):
            with open(os.path.join(savedir, 'log.csv'), 'w') as f:
                f.write(','.join(log_header) + '\n')


def find_checkpoint(savedir):
    file_list = glob.glob(os.path.join(savedir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for m in file_list:
            result = re.findall(".*model_(.*).pth.*", m)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def main():
    # load train data
    print('==> Data loading')
    hf = h5py.File(args.train_data, 'r+')
    lr_polsar = np.float32(hf['data1'])
    hr_dualsar = np.float32(hf['data2'])
    hr_diff = np.float32(hf['data3'])
    lr_pd = np.float32(hf['data4'])
    hr_polsar = np.float32(hf['label'])

    lr_polsar = torch.from_numpy(lr_polsar).view(-1, args.ncha_polsar, 20, 20)
    hr_dualsar = torch.from_numpy(hr_dualsar).view(-1, args.ncha_dualsar, 40, 40)
    hr_diff = torch.from_numpy(hr_diff).view(-1, args.ncha_diff, 40, 40)
    lr_pd = torch.from_numpy(lr_pd).view(-1, args.ncha_pd, 20, 20)
    hr_polsar = torch.from_numpy(hr_polsar).view(-1, args.ncha_polsar, 40, 40)

    train_set = FDFNetDataset(lr_polsar, hr_dualsar, hr_diff, lr_pd, hr_polsar)
    train_loader = DataLoader(dataset=train_set, num_workers=8, drop_last=True, batch_size=64, shuffle=True, pin_memory=True)

    print('==> Model building')
    model = FDFNet(args)
    criterion1 = nn.L1Loss(size_average=True)
    criterion2 = nn.L1Loss(size_average=True)
    criterion3 = nn.L1Loss(size_average=True)

    if cuda:
        print('==> GPU setting')
        model = model.cuda()
        criterion1 = criterion1.cuda()
        criterion2 = criterion2.cuda()
        criterion3 = criterion3.cuda()

    print('==> Optimizer setting')
    optimizer = optim.Adam(model.parameters(), lr=args.lr,  weight_decay=0)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    initial_epoch = find_checkpoint(savedir=savedir)
    if initial_epoch > 0:
        print('==> Resuming by loading epoch %03d' % initial_epoch)
        model = torch.load(os.path.join(savedir, 'model_%03d.pth' % initial_epoch))

    if len(args.gpu) > 1:
        model = torch.nn.DataParallel(model)

    for epoch in range(initial_epoch, args.epoch):
        scheduler.step(epoch)
        epoch_loss = 0
        start_time = time.time()

        # train
        model.train()
        num_train = len(train_loader.dataset)
        for iteration, batch in enumerate(train_loader):
            lr_polsar_batch, hr_dualsar_batch, hr_diff_batch, lr_pd_batch, hr_polsar_batch = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3]), Variable(batch[4])
            if cuda:
                lr_polsar_batch = lr_polsar_batch.cuda()
                hr_dualsar_batch = hr_dualsar_batch.cuda()
                hr_diff_batch = hr_diff_batch.cuda()
                lr_pd_batch = lr_pd_batch.cuda()
                hr_polsar_batch = hr_polsar_batch.cuda()
            optimizer.zero_grad()
            out = model(lr_polsar_batch, hr_dualsar_batch, hr_diff_batch, lr_pd_batch)
            out_hh = out[:, 0, :, :].view(-1, 1, 40, 40)
            out_hv = (0.5 * out[:, 5, :, :]).view(-1, 1, 40, 40)
            out_vv = out[:, 8, :, :].view(-1, 1, 40, 40)
            hr_sar_hv_batch = hr_dualsar_batch[:, 0, :, :].view(-1, 1, 40, 40)
            hr_sar_vv_batch = hr_dualsar_batch[:, 1, :, :].view(-1, 1, 40, 40)

            loss1 = criterion1(out, hr_polsar_batch)
            loss2 = criterion2(out_hv, hr_sar_hv_batch)
            loss3 = criterion3(out_vv, hr_sar_vv_batch)

            lambda1 = loss1.data / (loss1.data + loss2.data + loss3.data)
            lambda2 = loss2.data / (loss1.data + loss2.data + loss3.data)
            lambda3 = loss3.data / (loss1.data + loss2.data + loss3.data)
            loss = lambda1 * loss1 + lambda2 * loss2 + lambda3 * loss3

            epoch_loss += loss.data/num_train
            print('%4d %4d / %4d loss = %2.6f' % (epoch + 1, iteration, train_set.hr_polsar.size(0)/args.batchsize, loss.data))
            loss.backward()
            optimizer.step()
            with open(os.path.join(savedir, 'log.csv'), 'a') as file:
                log = [epoch, iteration] + [loss.data.item()]
                log = map(str, log)
                file.write(','.join(log) + '\n')
        if len(args.gpu) > 1:
            torch.save(model.module, os.path.join(savedir, 'model_%03d.pth' % (epoch + 1)))
        else:
            torch.save(model, os.path.join(savedir, 'model_%03d.pth' % (epoch + 1)))
        gc.collect()
        elapsed_time = time.time() - start_time
        print('epcoh = %4d , time is %4.4f s' % (epoch + 1, elapsed_time))


if __name__ == '__main__':
    main()
