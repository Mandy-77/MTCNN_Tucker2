#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from numpy import linalg as la
import torch.nn as nn

class Tkd2Conv(nn.Module):

    def __init__(self, conv_nn_module, rc, rf):

        def tucker2decomposition(conv_nn_module, rc, rf):
            bias = conv_nn_module.bias
            stride = conv_nn_module.stride
            padding = conv_nn_module.padding
            conv = conv_nn_module.weight.detach().data.numpy()
            conv = conv.transpose((2,3,1,0))
            dim_tensor = conv.shape
            mode3_matrix = np.transpose(conv, (2, 0, 1, 3)).reshape([dim_tensor[2], -1])
            u3, sigma3, vt3 = la.svd(mode3_matrix)
            conv1_matrix = u3[:, 0:rc]
            mode4_matrix = np.transpose(conv, (3, 0, 1, 2)).reshape([dim_tensor[3], -1])
            u4, sigma4, vt4 = la.svd(mode4_matrix)
            conv3_matrix = u4[:, 0:rf]
            conv2 = np.dot(conv1_matrix.transpose(), mode3_matrix).reshape(rc, dim_tensor[0],
                                 dim_tensor[1], dim_tensor[3]).transpose([1, 2, 0, 3])
            conv2 = np.transpose(conv2, (3, 0, 1, 2)).reshape([dim_tensor[3], -1])
            conv2 = np.dot(conv3_matrix.transpose(), conv2).reshape(rf, dim_tensor[0],
                                 dim_tensor[1], rc).transpose([1, 2, 3, 0])
            conv1 = conv1_matrix.reshape([1, 1, dim_tensor[2], rc])
            conv3 = conv3_matrix.transpose().reshape([1, 1, rf, dim_tensor[3]])
            return conv1, conv2, conv3, bias, stride, padding


        super(Tkd2Conv,self).__init__()
        conv1, conv2, conv3, bias, stride, padding = tucker2decomposition(conv_nn_module, rc, rf)
        size1 = conv1.shape
        size2 = conv2.shape
        size3 = conv3.shape
        conv1_weight = torch.from_numpy(conv1).permute(3, 2, 0, 1).float()
        conv2_weight = torch.from_numpy(conv2).permute(3, 2, 0, 1).float()
        conv3_weight = torch.from_numpy(conv3).permute(3, 2, 0, 1).float()
        self.conv1 = nn.Conv2d(size1[2], size1[3], size1[0], bias=False)
        self.conv2 = nn.Conv2d(size2[2], size2[3], size2[0], stride = stride,
                               padding = padding, bias=False)
        self.conv3 = nn.Conv2d(size3[2], size3[3], size3[0], bias=False)
        self.conv1.weight = nn.Parameter(data=conv1_weight, requires_grad=True)
        self.conv2.weight = nn.Parameter(data=conv2_weight, requires_grad=True)
        self.conv3.weight = nn.Parameter(data=conv3_weight, requires_grad=True)
        # self.conv3.bias = nn.Parameter(data=bias, requires_grad=True)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.conv3(x)
        return out

# test
def main():
    conv_nn_module = torch.nn.Conv2d(16, 32, 5, 5)
    rc = 4
    rf = 4
    decomposed_conv = Tkd2Conv(conv_nn_module, rc, rf)
    x = torch.rand([1, 16, 32, 32])
    out = decomposed_conv(x)
    out2 = conv_nn_module(x)

    error = out - out2
    norm_n = np.linalg.norm(error.data.numpy().reshape(-1), ord=2)
    norm_d = np.linalg.norm(out2.data.numpy().reshape(-1), ord=2)
    result = norm_n / norm_d

    relative = error/out2
    result2 = np.linalg.norm(relative.data.numpy().reshape(-1), ord=1)/(1*32*28*28)
    return out, out2, result, result2

if __name__ == "__main__":
    out, out2, result, result2 = main()
    print(out, out2, result, result2)








