#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from numpy import linalg as la
import torch.nn as nn
from scipy.optimize import minimize_scalar
import time

class Tkd2ConvVbmf(nn.Module):
    def __init__(self, conv_nn_module, cacb1, sigma2_1, cacb2, sigma2_2):

        def VBMF(Y, cacb, sigma2=None, H=None):
            L,M = Y.shape
            if H is None:
                H = L
            U,s,V = np.linalg.svd(Y)
            U = U[:,:H]
            s = s[:H]
            V = V[:H].T
            residual = 0.
            if H<L:
                residual = np.sum(np.sum(Y**2)-np.sum(s**2))
            if sigma2 is None:
                upper_bound = (np.sum(s**2)+ residual)/(L+M)

                if L==H:
                    lower_bound = s[-1]**2/M
                else:
                    lower_bound = residual/((L-H)*M)

                sigma2_opt = minimize_scalar(VBsigma2, args=(L,M,cacb,s,residual), bounds=[lower_bound, upper_bound], method='Bounded')
                sigma2 = sigma2_opt.x
                print ("Estimated sigma2: ", sigma2)
            thresh_term = (L+M + sigma2/cacb**2)/2
            threshold = np.sqrt( sigma2 * (thresh_term + np.sqrt(thresh_term**2 - L*M) ))
            pos = np.sum(s>threshold)
            d = np.multiply(s[:pos],
                            1 - np.multiply(sigma2/(2*s[:pos]**2),
                                            L+M+np.sqrt( (M-L)**2 + 4*s[:pos]**2/cacb**2 )))
            post = {}
            zeta = sigma2/(2*L*M) * (L+M+sigma2/cacb**2 - np.sqrt((L+M+sigma2/cacb**2)**2 - 4*L*M))
            post['ma'] = np.zeros(H)
            post['mb'] = np.zeros(H)
            post['sa2'] = cacb * (1-L*zeta/sigma2) * np.ones(H)
            post['sb2'] = cacb * (1-M*zeta/sigma2) * np.ones(H)
            delta = cacb/sigma2 * (s[:pos]-d- L*sigma2/s[:pos])
            post['ma'][:pos] = np.sqrt(np.multiply(d, delta))
            post['mb'][:pos] = np.sqrt(np.divide(d, delta))
            post['sa2'][:pos] = np.divide(sigma2*delta, s[:pos])
            post['sb2'][:pos] = np.divide(sigma2, np.multiply(delta, s[:pos]))
            post['sigma2'] = sigma2
            post['F'] = 0.5*(L*M*np.log(2*np.pi*sigma2) + (residual+np.sum(s**2))/sigma2 - (L+M)*H
                       + np.sum(M*np.log(cacb/post['sa2']) + L*np.log(cacb/post['sb2'])
                                + (post['ma']**2 + M*post['sa2'])/cacb + (post['mb']**2 + L*post['sb2'])/cacb
                                + (-2 * np.multiply(np.multiply(post['ma'], post['mb']), s)
                                   + np.multiply(post['ma']**2 + M*post['sa2'],post['mb']**2 + L*post['sb2']))/sigma2))
            return U[:,:pos], np.diag(d), V[:,:pos], post

        def VBsigma2(sigma2,L,M,cacb,s,residual):
            H = len(s)
            thresh_term = (L+M + sigma2/cacb**2)/2
            threshold = np.sqrt( sigma2 * (thresh_term + np.sqrt(thresh_term**2 - L*M) ))
            pos = np.sum(s>threshold)
            d = np.multiply(s[:pos],
                            1 - np.multiply(sigma2/(2*s[:pos]**2),
                                            L+M+np.sqrt( (M-L)**2 + 4*s[:pos]**2/cacb**2 )))
            zeta = sigma2/(2*L*M) * (L+M+sigma2/cacb**2 - np.sqrt((L+M+sigma2/cacb**2)**2 - 4*L*M))
            post_ma = np.zeros(H)
            post_mb = np.zeros(H)
            post_sa2 = cacb * (1-L*zeta/sigma2) * np.ones(H)
            post_sb2 = cacb * (1-M*zeta/sigma2) * np.ones(H)
            delta = cacb/sigma2 * (s[:pos]-d- L*sigma2/s[:pos])
            post_ma[:pos] = np.sqrt(np.multiply(d, delta))
            post_mb[:pos] = np.sqrt(np.divide(d, delta))
            post_sa2[:pos] = np.divide(sigma2*delta, s[:pos])
            post_sb2[:pos] = np.divide(sigma2, np.multiply(delta, s[:pos]))
            F = 0.5*(L*M*np.log(2*np.pi*sigma2) + (residual+np.sum(s**2))/sigma2 - (L+M)*H
                       + np.sum(M*np.log(cacb/post_sa2) + L*np.log(cacb/post_sb2)
                                + (post_ma**2 + M*post_sa2)/cacb + (post_mb**2 + L*post_sb2)/cacb
                                + (-2 * np.multiply(np.multiply(post_ma, post_mb), s)
                                   + np.multiply(post_ma**2 + M*post_sa2,post_mb**2 + L*post_sb2))/sigma2))
            return F

        def tucker2decomposition(conv_nn_module, cacb1, sigma2_1, cacb2, sigma2_2):
            #bias = conv_nn_module.bias
            bias = 0
            stride = conv_nn_module.stride
            padding = conv_nn_module.padding
            conv = conv_nn_module.weight.cpu().detach().data.numpy()
            conv = conv.transpose((2,3,1,0))
            dim_tensor = conv.shape

            mode3_matrix = np.transpose(conv, (2, 0, 1, 3)).reshape([dim_tensor[2], -1])
            start = time.clock()
            u3, sigma3, vt3, _ = VBMF(mode3_matrix, cacb1, sigma2_1)
            end = time.clock()
            conv1_matrix = u3
            rc = len(sigma3)
            print('The time of VBMF is: {}'.format(end - start))
            print('The rank of mode-k1 is: {}'.format(rc))

            mode4_matrix = np.transpose(conv, (3, 0, 1, 2)).reshape([dim_tensor[3], -1])
            start = time.clock()
            u4, sigma4, vt4, _ = VBMF(mode4_matrix, cacb2, sigma2_2)
            end = time.clock()
            conv3_matrix = u4
            rf = len(sigma4)
            print('The time of VBMF is: {}'.format(end - start))
            print('The rank of mode-k2 is: {}'.format(rf))
            print('\n')

            conv2 = np.dot(conv1_matrix.transpose(), mode3_matrix).reshape(rc, dim_tensor[0],
                                 dim_tensor[1], dim_tensor[3]).transpose([1, 2, 0, 3])
            conv2 = np.transpose(conv2, (3, 0, 1, 2)).reshape([dim_tensor[3], -1])
            conv2 = np.dot(conv3_matrix.transpose(), conv2).reshape(rf, dim_tensor[0],
                                 dim_tensor[1], rc).transpose([1, 2, 3, 0])
            conv1 = conv1_matrix.reshape([1, 1, dim_tensor[2], rc])
            conv3 = conv3_matrix.transpose().reshape([1, 1, rf, dim_tensor[3]])
            return conv1, conv2, conv3, bias, stride, padding


        super(Tkd2ConvVbmf, self).__init__()
        conv1, conv2, conv3, bias, stride, padding = tucker2decomposition(conv_nn_module, cacb1, sigma2_1,
                                                                          cacb2, sigma2_2)
        size1 = conv1.shape
        size2 = conv2.shape
        size3 = conv3.shape
        conv1_weight = torch.from_numpy(conv1).permute(3, 2, 0, 1).float()
        conv2_weight = torch.from_numpy(conv2).permute(3, 2, 0, 1).float()
        conv3_weight = torch.from_numpy(conv3).permute(3, 2, 0, 1).float()
        self.conv1 = nn.Conv2d(size1[2], size1[3], size1[0], bias=False)
        self.conv2 = nn.Conv2d(size2[2], size2[3], size2[0], stride = stride, padding = padding, bias=False)
        #self.conv3 = nn.Conv2d(size3[2], size3[3], size3[0], bias=True)
        self.conv3 = nn.Conv2d(size3[2], size3[3], size3[0], bias=False)
        self.conv1.weight = nn.Parameter(data=conv1_weight, requires_grad=True)
        self.conv2.weight = nn.Parameter(data=conv2_weight, requires_grad=True)
        self.conv3.weight = nn.Parameter(data=conv3_weight, requires_grad=True)
        #self.conv3.bias = nn.Parameter(data=bias, requires_grad=True)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.conv3(x)
        return out

if __name__ == "__main__":
    model = torchvision.models.resnet50(pretrained=True, progress=True) 
    Tkd2ConvVbmf(model.layer4[2].conv2, cacb1=1, sigma2_1=None, cacb2=1, sigma2_2=None)
