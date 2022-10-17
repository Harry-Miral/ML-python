# -*- coding: UTF-8 -*-
from torch.utils.data import Dataset
from load_data import Load


class MyDataset(Dataset):
    def __init__(self, k: int):
        self.img, self.y = Load(k)

    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, item):
        return self.img[item], self.y[item]
