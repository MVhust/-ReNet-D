from torch.utils import data
from dataset import patches_generation


class TrainDataset(data.Dataset):
    def __init__(self, opt):
        self.patches = patches_generation(opt=opt, mode='train')

    def __getitem__(self, item):
        res = self.patches[item]
        return res

    def __len__(self):
        return len(self.patches)
