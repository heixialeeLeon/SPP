import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from spp_layer import spatial_pyramid_pool


class SPP_NET(nn.Module):
    '''
    A CNN model which adds spp layer so that we can input multi-size tensor
    '''

    def __init__(self, input_nc, ndf=64, gpu_ids=[]):
        super(SPP_NET, self).__init__()
        self.gpu_ids = gpu_ids
        self.output_num = (4,2,1)

        self.conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False)
        self.LReLU1 =nn.ReLU()

        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 1, 1, bias=False)
        self.BN1 = nn.BatchNorm2d(ndf * 2)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 1, 1, bias=False)
        self.BN2 = nn.BatchNorm2d(ndf * 4)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=False)
        self.BN3 = nn.BatchNorm2d(ndf * 8)

        self.conv5 = nn.Conv2d(ndf * 8, 64, 4, 1, 0, bias=False)
        self.fc1 = nn.Linear(2688, 1024)
        self.fc2 = nn.Linear(1024, 256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.LReLU1(x)

        x = self.conv2(x)
        x = F.leaky_relu(self.BN1(x))

        x = self.conv3(x)
        x = F.leaky_relu(self.BN2(x))

        x = self.conv4(x)
        # x = F.leaky_relu(self.BN3(x))
        # x = self.conv5(x)
        spp = spatial_pyramid_pool(x, self.output_num)

        # print(spp.size())
        fc1 = self.fc1(spp)
        fc2 = self.fc2(fc1)
        s = nn.Sigmoid()
        output = s(fc2)
        return output

if __name__ == "__main__":
    input = Variable(torch.randn(8, 3, 200, 200))
    net = SPP_NET(input_nc=3, ndf=16)
    output = net(input)
    print(output.shape)

    input = Variable(torch.randn(8, 3, 300, 300))
    net = SPP_NET(input_nc=3, ndf=16)
    output = net(input)
    print(output.shape)