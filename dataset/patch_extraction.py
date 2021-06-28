from dataset import TextileData
import numpy as np


def patches_generation(opt, root='', patch_size=32, stride=4, mode='train'):
    """
    :param opt:
    :param root:
    :param patch_size:
    :param stride:
    :param mode: 'train' or 'test'
    :return: patches
    """
    if mode == 'train':
        root = opt.train_raw_data_root + opt.pattern_index + '/'
        patch_size = opt.patch_size
        stride = opt.train_patch_stride
    elif mode == 'test':
        root = opt.test_raw_data_root + opt.pattern_index + '/'
        patch_size = opt.patch_size
        stride = opt.test_patch_stride

    raw_dataset = TextileData(root)
    N, W, H, C = len(raw_dataset), 0, 0, 0
    if mode == 'train':
        W = opt.train_width
        H = opt.train_height
        C = opt.channel
    elif mode == 'test':
        W = opt.test_width
        H = opt.test_height
        C = opt.channel

    raw_data = np.ones([N, C, H, W])

    for index, item in enumerate(raw_dataset):
        raw_data[index] = item

    if mode == 'test':
        raw_data = np.pad(raw_data, ((0,), (0,), (stride,), (stride,)), mode='edge')
        #https://www.cnblogs.com/hezhiyao/p/8177541.html 

    w_scale = (W - patch_size)/stride + 1
    h_scale = (H - patch_size)/stride + 1
    if mode=='test':
        w_scale += 2
        h_scale += 2
    total_patches = w_scale * h_scale * N
    total_patches = int(total_patches)

    patches = np.ones([total_patches, C, patch_size, patch_size])
    index = 0

    if mode == 'train':
        for i in range(int(h_scale)):
            for j in range(int(w_scale)):
                section = raw_data[:, :, i*stride:i*stride + patch_size, j*stride:j*stride + patch_size]
                patches[index*N:(index+1)*N] = section
                index += 1
    elif mode == 'test':
        pass
        for n in range(N):
            for i in range(int(h_scale)):
                for j in range(int(w_scale)):
                    section = raw_data[n, :, i*stride:i*stride + patch_size, j*stride:j*stride + patch_size]
                    patches[index] = section
                    index += 1

    return patches

