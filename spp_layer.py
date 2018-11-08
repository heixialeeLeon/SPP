import math
import torch
import torch.nn as nn
from torch.autograd import Variable

def spatial_pyramid_pool(x, out_side):
    out = None
    for n in out_side:
        w_r, h_r = map(lambda s: math.ceil(s / n), x.size()[2:])  # Receptive Field Size
        s_w, s_h = map(lambda s: math.floor(s / n), x.size()[2:])  # Stride
        max_pool = nn.MaxPool2d(kernel_size=(w_r, h_r), stride=(s_w, s_h))
        y = max_pool(x)
        if out is None:
            out = y.view(y.size()[0], -1)
        else:
            out = torch.cat((out, y.view(y.size()[0], -1)), 1)
    return out

class SpatialPyramidPool2D(nn.Module):
    """
    Args:
        out_side (tuple): Length of side in the pooling results of each pyramid layer.

    Inputs:
        - `input`: the input Tensor to invert ([batch, channel, width, height])
    """

    def __init__(self, out_side):
        super(SpatialPyramidPool2D, self).__init__()
        self.out_side = out_side

    def forward(self, x):
        out = None
        for n in self.out_side:
            w_r, h_r = map(lambda s: math.ceil(s / n), x.size()[2:])  # Receptive Field Size
            s_w, s_h = map(lambda s: math.floor(s / n), x.size()[2:])  # Stride
            max_pool = nn.MaxPool2d(kernel_size=(w_r, h_r), stride=(s_w, s_h))
            y = max_pool(x)
            if out is None:
                out = y.view(y.size()[0], -1)
            else:
                out = torch.cat((out, y.view(y.size()[0], -1)), 1)
        return out

if __name__ == "__main__":
    input = Variable(torch.randn(8,3, 50, 50))
    output = spatial_pyramid_pool(input, (4,2))
    print(output.shape)

    input = Variable(torch.randn(8, 3, 30, 30))
    output = spatial_pyramid_pool(input, (4,2))
    print(output.shape)

    input = Variable(torch.randn(8, 3, 80, 80))
    output = spatial_pyramid_pool(input, (4,2))
    print(output.shape)