import torch.nn as nn
import torch.nn.functional as F
import torch
import time
from config import DefaultConfig

opt = DefaultConfig()

#网络结构参考FCN
#（3,64 3x3）
#（64,128 3x3）
#
#

class CAE8(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.model_name = str(type(self))

        self.encoder_conv1 = nn.Conv2d(3, 10, kernel_size=5, padding=2, stride=1)
        self.encoder_conv2 = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=1)
        self.encoder_conv3 = nn.Conv2d(20, 30, kernel_size=5, padding=2, stride=1)
        self.encoder_conv4 = nn.Conv2d(30, 40, kernel_size=5, padding=2, stride=1)
        self.encoder_conv5 = nn.Conv2d(40, 50, kernel_size=5, padding=2, stride=1)

        self.decoder_deconv5 = nn.ConvTranspose2d(50, 40, kernel_size=5, padding=2, stride=1)
        self.decoder_deconv4 = nn.ConvTranspose2d(40, 30, kernel_size=5, padding=2, stride=1)
        self.decoder_deconv3 = nn.ConvTranspose2d(30, 20, kernel_size=5, padding=2, stride=1)
        self.decoder_deconv2 = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=1)
        self.decoder_deconv1 = nn.ConvTranspose2d(10, 3, kernel_size=5, padding=2, stride=1)

        self.decoder_deconv6 = nn.ConvTranspose2d(3, 3, kernel_size=5, padding=2, stride=1)
        self.decoder_deconv7 = nn.ConvTranspose2d(3, 3, kernel_size=1)


    def forward(self, x):
        #x = F.relu(self.encoder_conv1(x))
        #x = F.relu(self.encoder_conv2(x))
        #F.max_pool2d()
        x, indices1 = F.max_pool2d(F.relu(self.encoder_conv1(x)), kernel_size=2, stride=2, return_indices=True)

        #x = F.relu(self.encoder_conv4(x))
        #x = F.relu(self.encoder_conv4(x))
        x, indices2 = F.max_pool2d(F.relu(self.encoder_conv2(x)), kernel_size=2, stride=2, return_indices=True)

        #x = F.relu(self.encoder_conv4(x))
        #x = F.relu(self.encoder_conv4(x))
        x, indices3 = F.max_pool2d(F.relu(self.encoder_conv3(x)), kernel_size=2, stride=2, return_indices=True)

        x = F.relu(self.encoder_conv4(x))
        x = F.relu(self.encoder_conv5(x))
        #x = F.relu(self.encoder_conv4(x))


        #x = F.relu(self.decoder_deconv4(x))
        x = F.relu(self.decoder_deconv5(x))
        x = F.max_unpool2d(F.relu(self.decoder_deconv4(x)), indices=indices3, kernel_size=2, stride=2)
        #x = F.relu(self.decoder_deconv4(x))
        #x = F.relu(self.decoder_deconv4(x))
        x = F.max_unpool2d(F.relu(self.decoder_deconv3(x)), indices=indices2, kernel_size=2, stride=2)
        #x = F.relu(self.decoder_deconv4(x))
        #x = F.relu(self.decoder_deconv4(x))
        x = F.max_unpool2d(F.relu(self.decoder_deconv2(x)), indices=indices1, kernel_size=2, stride=2)
        #x = F.relu(self.decoder_deconv2(x))
        #leaky_relu(input, negative_slope=0.01, inplace=False)
        x = F.relu(self.decoder_deconv1(x))
        x = F.relu(self.decoder_deconv6(x))
        x = F.relu(self.decoder_deconv7(x))

        return x

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        torch.save(self.state_dict(), name)
