from cgi import test
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utils import *

import time
from datetime import datetime
from readData import readtoArray
import os, sys
from colorMap import parula
import argparse

from sklearn.decomposition import PCA
from sklearn.preprocessing import * ###

torch.manual_seed(0)
np.random.seed(0)

################################################################
# PCANN network
################################################################
# class pcann(torch.nn.Module):
#     def __init__(self, params):
#         super(pcann, self).__init__()
#         self.params = params
#         self.linear = nn.ModuleList()
#         self.listt   = params["layers"]
#         for i in range(len(self.listt) - 1):
#             self.linear.append(nn.Linear(self.listt[i], self.listt[i+1]))

#     def forward(self, x):     
#         for layer in self.linear:
#             x = F.selu(layer(x))
        
#         return x


class pcann(nn.Module):

    def __init__(self, in_features: int, out_features: int, p_drop=0.2, use_selu: bool=True):
        super(pcann, self).__init__()

        activation = nn.SELU() if use_selu else nn.ReLU()
        dropout = nn.AlphaDropout(p=p_drop) if use_selu else nn.Dropout(p=p_drop)

        self.net = nn.Sequential( # flatten input image from batchx1x28x28 to batchx784
            nn.Linear(in_features=in_features, out_features=500),
            activation,
            dropout,
            nn.Linear(in_features=500, out_features=1000),
            activation,
            dropout,
            nn.Linear(in_features=1000, out_features=2000),
            activation,
            dropout,
            nn.Linear(in_features=2000, out_features=1000),
            activation,
            dropout,
            nn.Linear(in_features=1000, out_features=500),
            activation,
            dropout,
            nn.Linear(in_features=500, out_features=out_features)
        )

        if use_selu:
            for param in self.net.parameters():
                # biases zero
                if len(param.shape) == 1:
                    nn.init.constant_(param, 0)
                # others using lecun-normal initialization
                else:
                    nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        return self.net(x)



################################################################
# PCALin network
################################################################
class pcalin(torch.nn.Module):
    def __init__(self, params):
        super(pcalin, self).__init__()
        self.params = params
        self.listt   = params["layers"]
        self.linear = nn.Linear(self.listt[0], self.listt[-1])

    def forward(self, x):
        x = self.linear(x)
        
        return x