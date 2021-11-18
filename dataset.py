# python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/11 20:35
# @Author  : Liupeng Lin
# @Email   : linliupeng@whu.edu.cn
# Copyright (C) 2021 Liupeng Lin. All Rights Reserved.


from torch.utils.data import Dataset


class FDFNetDataset(Dataset):
    def __init__(self, lr_polsar, hr_dualsar, hr_diff, lr_pd, hr_polsar):
        super(FDFNetDataset, self).__init__()
        self.lr_polsar = lr_polsar
        self.hr_dualsar = hr_dualsar
        self.hr_diff = hr_diff
        self.lr_pd = lr_pd
        self.hr_polsar = hr_polsar

    def __getitem__(self, index):
        batch_lr_polsar = self.lr_polsar[index]
        batch_hr_dualsar = self.hr_dualsar[index]
        batch_hr_diff = self.hr_diff[index]
        batch_lr_pd = self.lr_pd[index]
        batch_hr_polsar = self.hr_polsar[index]
        return batch_lr_polsar.float(), batch_hr_dualsar.float(), batch_hr_diff.float(), batch_lr_pd.float(), batch_hr_polsar.float()

    def __len__(self):
        return self.hr_polsar.size(0)



