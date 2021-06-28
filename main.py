import os
from train import train
from test import test
from config import DefaultConfig

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

opt = DefaultConfig()

train(opt, initialize=True)
print('train_done!')
test(opt)


