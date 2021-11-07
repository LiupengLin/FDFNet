from torch.utils.data import Dataset


class FDFNetDataset(Dataset):
    def __init__(self, lr_polsar, hr_sar, hr_sar_diff, lr_pd, hr_polsar):
        super(FDFNetDataset, self).__init__()
        self.lr_polsar = lr_polsar
        self.hr_sar = hr_sar
        self.hr_sar_diff = hr_sar_diff
        self.lr_pd = lr_pd
        self.hr_polsar = hr_polsar

    def __getitem__(self, index):
        batch_lr_polsar = self.lr_polsar[index]
        batch_hr_sar = self.hr_sar[index]
        batch_hr_sar_diff = self.hr_sar_diff[index]
        batch_lr_pd = self.lr_pd[index]
        batch_hr_polsar = self.hr_polsar[index]
        return batch_lr_polsar.float(), batch_hr_sar.float(), batch_hr_sar_diff.float(), batch_lr_pd.float(), batch_hr_polsar.float()

    def __len__(self):
        return self.hr_polsar.size(0)


