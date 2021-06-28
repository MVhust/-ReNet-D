from torch.utils import data
import os
import cv2


class TextileData(data.Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]

    def __getitem__(self, item):
        img_path = self.imgs[item]
        img = cv2.imread(img_path, 0)
        return img

    def __len__(self):
        return len(self.imgs)
