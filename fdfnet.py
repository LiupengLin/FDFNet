# python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/11 20:36
# @Author  : Liupeng Lin
# @Email   : linliupeng@whu.edu.cn
# Copyright (C) 2021 Liupeng Lin. All Rights Reserved.


import torch
from torch import nn
import torch.nn.init as init


class make_dense(nn.Module):
    def __init__(self, nFeat, growthRate):
        super(make_dense, self).__init__()
        self.conv_dense = nn.Sequential(
            nn.Conv2d(nFeat, growthRate, kernel_size=3, padding=1, bias=True), nn.PReLU())

    def forward(self, x):
        out1 = self.conv_dense(x)
        out = torch.cat((x, out1), 1)
        return out


class RDB(nn.Module):
    def __init__(self, nFeat, nDense, growthRate):
        super(RDB, self).__init__()
        nFeat_ = nFeat
        modules = []
        for i in range(nDense):
            modules.append(make_dense(nFeat_, growthRate))
            nFeat_ += growthRate
            self.dense_layers = nn.Sequential(*modules)
            self.conv_1x1 = nn.Conv2d(nFeat_, nFeat, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out1 = self.conv_1x1(self.dense_layers(x))
        out = torch.add(x, out1)
        return out


class CA(nn.Module):
    def __init__(self, nFeat, ratio=8):
        super(CA, self).__init__()
        self.conv1_ca = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3,  stride=1, padding=1, bias=True), nn.PReLU())
        self.conv2_ca = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3,  stride=1, padding=1, bias=True), nn.PReLU())
        self.avg_pool_ca = nn.AdaptiveAvgPool2d(1)
        self.fc1_ca = nn.Sequential(
            nn.Conv2d(nFeat, nFeat//ratio, kernel_size=1, padding=0, bias=True), nn.PReLU())
        self.fc2_ca = nn.Sequential(
            nn.Conv2d(nFeat//ratio, nFeat, kernel_size=1, padding=0, bias=True), nn.Sigmoid())

    def forward(self, x):
        out1 = self.conv1_ca(x)
        avg_weight_ca = self.fc2_ca(self.fc1_ca(self.avg_pool_ca(out1)))
        out = self.conv2_ca(torch.mul(x, avg_weight_ca))
        return out


class SA(nn.Module):
    def __init__(self, nFeat):
        super(SA, self).__init__()
        self.conv1_sa = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3,  stride=1, padding=1, bias=True), nn.PReLU())
        self.conv2_sa = nn.Sequential(
            nn.Conv2d(nFeat, 1, kernel_size=3, stride=1, padding=1, bias=True), nn.Sigmoid())
        self.conv3_sa = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU())

    def forward(self, x):
        out1 = self.conv1_sa(x)
        weight_sa = self.conv2_sa(out1)
        out = self.conv3_sa(torch.mul(x, weight_sa))
        return out


class PDA(nn.Module):
    def __init__(self, nFeat):
        super(PDA, self).__init__()
        self.deconv_pda = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nFeat, out_channels=nFeat, kernel_size=4, stride=2, padding=1, bias=True), nn.Sigmoid())
        self.conv1_pda = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3,  stride=1, padding=1, bias=True), nn.PReLU())

    def forward(self, hr_sar_fea, lr_pd_fea):
        out = self.conv1_pda(torch.mul(hr_sar_fea, self.deconv_pda(lr_pd_fea)))
        return out


class RCSA(nn.Module):
    def __init__(self, nFeat):
        super(RCSA, self).__init__()
        self.ca_rcsa = CA(nFeat)
        self.sa_rcsa = SA(nFeat)
        self.conv1_rcsa = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU())

    def forward(self, x):
        out1 = self.conv1_rcsa(torch.add(self.ca_rcsa(x), self.sa_rcsa(x)))
        out = torch.add(x, out1)
        return out


class HDSA(nn.Module):
    def __init__(self, args):
        super(HDSA, self).__init__()
        ncha_dualsar = args.ncha_dualsar
        nFeat = args.nFeat
        self.conv1_hrsa = nn.Sequential(
            nn.Conv2d(ncha_dualsar, nFeat, kernel_size=3,  stride=1, padding=1, bias=True), nn.PReLU())
        self.conv2_hrsa = nn.Sequential(
            nn.Conv2d(nFeat, 1, kernel_size=3, padding=1, bias=True), nn.Sigmoid())
        self.conv3_hrsa = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3,  stride=1, padding=1, bias=True), nn.PReLU())

    def forward(self, lr_polsar_fea, hr):
        out1 = self.conv2_hrsa(self.conv1_hrsa(hr))
        out = self.conv3_hrsa(torch.mul(lr_polsar_fea, out1))
        return out


class LPCA(nn.Module):
    def __init__(self, args):
        ratio = args.ratio
        nFeat = args.nFeat
        ncha_dualsar = args.ncha_dualsar
        super(LPCA, self).__init__()
        self.conv1_lpca = nn.Sequential(
            nn.Conv2d(ncha_dualsar, nFeat, kernel_size=3,  stride=1, padding=1, bias=True), nn.PReLU())
        self.conv2_lpca = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU())
        self.avg_pool_lpca = nn.AdaptiveAvgPool2d(1)
        self.fc1_lpca = nn.Sequential(
            nn.Conv2d(nFeat, nFeat//ratio, kernel_size=1, padding=0, bias=True), nn.PReLU())
        self.fc2_lpca = nn.Sequential(
            nn.Conv2d(nFeat//ratio, nFeat, kernel_size=1, padding=0, bias=True), nn.Sigmoid())

    def forward(self, lr_polsar_fea, hr):
        out1 = self.conv1_lpca(hr)
        avg_weight_lpca = self.fc2_lpca(self.fc1_lpca(self.avg_pool_lpca(lr_polsar_fea)))
        out = self.conv2_lpca(torch.mul(out1, avg_weight_lpca))
        return out


class MCroAM(nn.Module):
    def __init__(self, args):
        super(MCroAM, self).__init__()
        nFeat = args.nFeat
        self.hdsa_mcroam = HDSA(args)
        self.lpca_mcroam = LPCA(args)
        self.conv1_mcroam = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU())

    def forward(self, lr, hr):
        out = self.conv1_mcroam(torch.add(self.hdsa_mcroam(lr, hr), self.lpca_mcroam(lr, hr)))
        return out


class ComplexBlock(nn.Module):
    def __init__(self, nFeat):
        super(ComplexBlock, self).__init__()
        nFeat_ = nFeat//2
        self.convs1 = nn.Sequential(
            nn.Conv2d(1, nFeat_, kernel_size=3, padding=1, bias=True), nn.PReLU())
        self.convs2 = nn.Sequential(
            nn.Conv2d(2, nFeat_, kernel_size=3, padding=1, bias=True), nn.PReLU())
        self.convs3 = nn.Sequential(
            nn.Conv2d(2, nFeat_, kernel_size=3, padding=1, bias=True), nn.PReLU())
        self.convs4 = nn.Sequential(
            nn.Conv2d(1, nFeat_, kernel_size=3, padding=1, bias=True), nn.PReLU())
        self.convs5 = nn.Sequential(
            nn.Conv2d(2, nFeat_, kernel_size=3, padding=1, bias=True), nn.PReLU())
        self.convs6 = nn.Sequential(
            nn.Conv2d(1, nFeat_, kernel_size=3, padding=1, bias=True), nn.PReLU())
        self.conv_cb = nn.Sequential(
            nn.Conv2d(nFeat_*6, nFeat, kernel_size=3, padding=1, bias=True), nn.PReLU())

    def forward(self, x):
        conv1 = self.convs1(torch.index_select(x, 1, torch.LongTensor([0]).cuda()))
        conv2 = self.convs2(torch.index_select(x, 1, torch.LongTensor([1, 2]).cuda()))
        conv3 = self.convs3(torch.index_select(x, 1, torch.LongTensor([3, 4]).cuda()))
        conv4 = self.convs4(torch.index_select(x, 1, torch.LongTensor([5]).cuda()))
        conv5 = self.convs5(torch.index_select(x, 1, torch.LongTensor([6, 7]).cuda()))
        conv6 = self.convs6(torch.index_select(x, 1, torch.LongTensor([8]).cuda()))
        out = self.conv_cb(torch.cat((conv1, conv2, conv3, conv4, conv5, conv6), 1))
        return out


class LPSR(nn.Module):
    def __init__(self, args):
        super(LPSR, self).__init__()
        nDense = args.nDense
        nFeat = args.nFeat
        growthRate = args.growthRate
        self.cb_lpsr = ComplexBlock(nFeat)
        self.deconv_lpsr = nn.Sequential(
            nn.ConvTranspose2d(nFeat, nFeat, kernel_size=4, stride=2, padding=1, bias=True), nn.PReLU())
        self.rdb1_lpsr = RDB(nFeat, nDense, growthRate)

    def forward(self, x):
        out1 = self.deconv_lpsr(self.cb_lpsr(x))
        out2 = self.rdb1_lpsr(out1)
        out = torch.add(out1, out2)
        return out


class DIM(nn.Module):
    def __init__(self, args):
        super(DIM, self).__init__()
        ncha_dualsar = args.ncha_dualsar
        ncha_diff = args.ncha_diff
        nDense = args.nDense
        nFeat = args.nFeat
        growthRate = args.growthRate
        self.conv1_dim = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True), nn.PReLU())
        self.conv2_dim = nn.Sequential(
            nn.Conv2d(ncha_dualsar, nFeat, kernel_size=3, padding=1, bias=True), nn.Sigmoid())
        self.conv3_dim = nn.Sequential(
            nn.Conv2d(ncha_diff, nFeat, kernel_size=3, padding=1, bias=True), nn.Sigmoid())
        self.rdb1_dim = RDB(nFeat, nDense, growthRate)
        self.rdb2_dim = RDB(nFeat, nDense, growthRate)
        self.rdb3_dim = RDB(nFeat, nDense, growthRate)
        self.rcsa1_dim = RCSA(nFeat)
        self.rcsa2_dim = RCSA(nFeat)
        self.rcsa3_dim = RCSA(nFeat)
        self.conv1_1x1_dim = nn.Conv2d(nFeat * 3, nFeat, kernel_size=1, padding=0, bias=True)

    def forward(self, lr_polsar_fea, hr_sar, hr_diff):
        out1 = torch.mul(lr_polsar_fea, self.conv2_dim(hr_sar))
        out2 = torch.mul(lr_polsar_fea, self.conv3_dim(hr_diff))
        out3 = self.rdb1_dim(self.rcsa1_dim(out2))
        out4 = self.rdb2_dim(self.rcsa2_dim(out3))
        out5 = self.rdb3_dim(self.rcsa3_dim(out4))
        out6 = self.conv1_1x1_dim(torch.cat((out3, out4, out5), 1))
        out = torch.sub(out1, out6)
        return out


class RDMA(nn.Module):
    def __init__(self, args):
        super(RDMA, self).__init__()
        nFeat = args.nFeat
        nDense = args.nDense
        growthRate = args.growthRate
        self.rcsa1_rdma = RCSA(nFeat)
        self.rcsa2_rdma = RCSA(nFeat)
        self.rcsa3_rdma = RCSA(nFeat)
        self.rdb1_rdma = RDB(nFeat, nDense, growthRate)
        self.rdb2_rdma = RDB(nFeat, nDense, growthRate)
        self.rdb3_rdma = RDB(nFeat, nDense, growthRate)
        self.conv1_1x1_rdma = nn.Conv2d(nFeat * 3, nFeat, kernel_size=1, padding=0, bias=True)

    def forward(self, fea):
        out1 = self.rdb1_rdma(self.rcsa1_rdma(fea))
        out2 = self.rdb2_rdma(self.rcsa2_rdma(out1))
        out3 = self.rdb3_rdma(self.rcsa3_rdma(out2))
        out = self.conv1_1x1_rdma(torch.cat((out1, out2, out3), 1))
        return out


class FDFNet(nn.Module):
    def __init__(self, args):
        super(FDFNet, self).__init__()
        ncha_polsar = args.ncha_polsar
        ncha_dualsar = args.ncha_dualsar
        ncha_pd = args.ncha_pd
        nFeat = args.nFeat
        self.conv1_fdfnet = nn.Sequential(
            nn.Conv2d(ncha_dualsar, nFeat, kernel_size=3, padding=1, bias=True), nn.PReLU())
        self.conv2_fdfnet = nn.Sequential(
            nn.Conv2d(ncha_pd, nFeat, kernel_size=3, padding=1, bias=True), nn.PReLU())
        self.conv3_fdfnet = nn.Sequential(
            nn.Conv2d(nFeat, ncha_polsar, kernel_size=3, padding=1, bias=True), nn.PReLU())
        self.mcroam1_fdfnet = MCroAM(args)
        self.pda1_fdfnet = PDA(nFeat)
        self.rdma_fdfnet = RDMA(args)
        self.lpsr_fdfnet = LPSR(args)
        self.dim_fdfnet = DIM(args)
        self.conv1_1x1_fdfnet = nn.Conv2d(nFeat * 4, nFeat, kernel_size=1, padding=0, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, lr_polsar, hr_sar, hr_diff, lr_pd):
        out1 = self.lpsr_fdfnet(lr_polsar)
        out2 = self.mcroam1_fdfnet(out1, hr_sar)
        out3 = self.dim_fdfnet(out1, hr_sar, hr_diff)
        out4 = self.pda1_fdfnet(self.conv1_fdfnet(hr_sar), self.conv2_fdfnet(lr_pd))
        out5 = self.rdma_fdfnet(self.conv1_1x1_fdfnet(torch.cat((out1, out2, out3, out4), 1)))
        out = self.conv3_fdfnet(torch.add(out1, out5))
        return out
