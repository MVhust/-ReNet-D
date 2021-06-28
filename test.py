# -*- coding: utf-8 -*-
from torch.autograd import Variable
from torch.utils.data import DataLoader
import models
import numpy as np
import os
from dataset import TestDataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from logger import Logger
import cv2
import scipy.io as oi
import torch
import time


def reconstruction(pieces, opt):
    res = np.ones([opt.test_height, opt.test_width])
    step = int(opt.test_patch_stride)
    res = np.pad(res, ((step,), (step,)), mode='edge')
    w_scale = int((opt.test_width - opt.patch_size) / opt.test_patch_stride + 3)
    h_scale = int((opt.test_height - opt.patch_size) / opt.test_patch_stride + 3)
    for i in range(h_scale):
        i_ = i * step
        for j in range(w_scale):
            j_ = j * step
            res[i_:i_ + opt.patch_size, j_:j_ + opt.patch_size] += pieces[i * w_scale + j, 0]

    res = res[step:step + opt.test_height, step:step + opt.test_width]
    return res


def visualize_recon(origin, generated, sub, idx, save_dir):
    sub = sub ** 2
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path1 = os.path.join(save_dir, str(idx) + '1' + '_recon.png')
    save_path2 = os.path.join(save_dir, str(idx) + '2' + '_recon.png')
    save_path3 = os.path.join(save_dir, str(idx) + '3' + '_recon.png')
    cv2.imwrite(save_path1, origin)
    cv2.imwrite(save_path2, generated)
    cv2.imwrite(save_path3, sub)


def visualize_patch_jet(residual, idx, save_dir, row, col, patch_size):
    row = int(row / patch_size)
    col = int(col / patch_size)
    mse = np.zeros((row, col))
    for row_idx in range(row):
        for col_idx in range(col):
            patch = residual[row_idx * patch_size:row_idx * patch_size + patch_size,
                    col_idx * patch_size:col_idx * patch_size + patch_size]
            patch = patch ** 2
            patch = np.mean(patch)
            mse[row_idx][col_idx] = patch
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    mat_path = os.path.join(save_dir, str(idx) + '.mat')
    mat = np.zeros([row, col])
    oi.savemat(mat_path, {'name': mat})


def test(opt, load_model_path='default_config'):
    if load_model_path == 'default_config':
        data_info = opt.model + '_patch_size_' + str(opt.patch_size)
        training_parameter = '_batch_size_' + str(opt.train_batch_size) + '_epoch_' + str(opt.max_epoch)
        # checkpoint = opt.load_model_path + data_info + training_parameter + '.pth'
        checkpoint = opt.load_model_path + 'CAE8_patch_size_32_batch_size_512_epoch_4000.pth'

        img_show_path = opt.img_show_path + data_info + training_parameter + '/'
    else:
        img_show_path = opt.img_show_path + load_model_path.replace('.pth', '/')

        index_of_CAE = load_model_path.find('CAE')
        index_of_patch_size = load_model_path.find('patch_size')
        print(img_show_path)

        model = load_model_path[index_of_CAE]
        patch_size = int(load_model_path[index_of_patch_size])
        test_patch_stride = int(patch_size / 2)
        new_config = {'model': model,
                      'patch_size': patch_size,
                      'test_patch_stride': test_patch_stride}
        opt.parse(new_config)

        data_path = opt.train_raw_data_root + '/'
        for item in os.listdir(data_path):
            data_path += item
            break

        img = io.imread(data_path, as_grey=True)
        height = img.shape[0]
        width = img.shape[1]
        new_config = {'train_height': height, 'train_width': width,
                      'test_height': height, 'test_width': width}
        opt.parse(new_config)

        checkpoint = opt.load_model_path + load_model_path

    if os.path.exists(img_show_path):
        pass
    else:
        os.makedirs(img_show_path)

    net = getattr(models, opt.model)()
    net.load(checkpoint)
    if opt.use_gpu:
        net.cuda()

    testDataset = TestDataset(opt=opt)
    w_scale = (opt.test_width - opt.patch_size) / opt.test_patch_stride + 3
    h_scale = (opt.test_height - opt.patch_size) / opt.test_patch_stride + 3
    s = int(w_scale * h_scale)
    test_dataloader = DataLoader(testDataset, batch_size=s, shuffle=False)

    for index, item in enumerate(test_dataloader, 0):
        inputs = item.float()
        original_img = inputs.numpy()
        original_img = reconstruction(original_img, opt)
        print(original_img.shape)
        inputs = Variable(inputs)
        if opt.use_gpu:
            inputs = inputs.cuda()
        start_time = time.time()
        outputs = net(inputs)
        end_time = time.time()
        print("sig_time: ", end_time - start_time, " s")
        if opt.use_gpu:
            outputs = outputs.cuda()
        outputs = outputs.data.cpu().numpy()

        generated_img = reconstruction(outputs, opt)
        residual_img = generated_img - original_img
        visualize_recon(original_img, generated_img, residual_img, index, img_show_path)
