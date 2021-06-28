from torch.utils import data
from dataset import patches_generation


class TestDataset(data.Dataset):
    def __init__(self, opt):
        self.patches = patches_generation(opt=opt, mode='test')

    def __getitem__(self, item):
        res = self.patches[item]
        return res

    def __len__(self):
        return len(self.patches)
