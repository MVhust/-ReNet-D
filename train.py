import torch.optim as opti
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import init
import os
from logger import Logger
from dataset import TrainDataset
import models
import cv2
import numpy as np
import math
import myLOSS


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal(m.weight)
        init.constant(m.bias, 0)


def train(opt, initialize=True):
    net = getattr(models, opt.model)()
    data_info = opt.model + '_patch_size_' + str(opt.patch_size)
    training_parameter = '_batch_size_' + str(opt.train_batch_size) + '_epoch_' + str(opt.max_epoch)
    log_dir = opt.log_dir + data_info + training_parameter + '/'
    if os.path.exists(log_dir):
        pass
    else:
        os.makedirs(log_dir)
    logger = Logger(log_dir)
    fo = open(log_dir+'prepare.bat', 'w')
    fo.write("cd %~dp0\ntensorboard --logdir ./")
    fo.close()
    fo = open(log_dir+'show_logs.bat', 'w')
    fo.write("start chrome.exe http://RFX:6006")
    fo.close()
    if initialize:
        pass
        # net.apply(weights_init)
    else:
        net.load(opt.load_model_path + data_info + training_parameter + '.pth')
    if opt.use_gpu:
        net.cuda()

    #criterion = nn.L1Loss()
    criterion = myLOSS.SSIM()
    #criterion = nn.MSELoss(size_average=False)
    optimizer = opti.Adam(net.parameters(), lr=opt.lr)
    trainDataset = TrainDataset(opt=opt)
    train_dataloader = DataLoader(trainDataset, batch_size=opt.train_batch_size, shuffle=True)
    loss_now_val = 1
    running_loss = 0.
    for epoch in range(opt.max_epoch):
        if epoch % 300 == 0:
            #logger.scalar_summary(opt.model+'TLOSS', running_loss/(opt.patch_size ** 2 * len(trainDataset)), epoch)
            print('epoch' + str(epoch) + ':  loss= %.8f' % (running_loss/(1 * len(trainDataset))))
            net.save(opt.load_model_path + data_info + training_parameter + str(epoch) +'.pth')

        running_loss = 0.0
        for index, item in enumerate(train_dataloader, 1):  # train_dataloader 的一个对象是BATCH
            inputs = item.float()
            inputs = Variable(inputs)
            if opt.use_gpu:
                inputs = inputs.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if index == len(train_dataloader):
                print('epoch' + str(epoch+1) + ' loss= %.8f' % (running_loss / (1 * len(trainDataset))))  # 一个epoch的损失

                loss_val = running_loss / (1 * len(trainDataset))
                if loss_val < loss_now_val:
                    loss_now_val = loss_val
                    net.save(opt.load_model_path + data_info + training_parameter + '.pth')
                else:
                    loss_now_val = loss_val
                    print('epoch' + str(epoch+1) + ': loss没有下降')


