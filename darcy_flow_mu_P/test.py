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
class pcann(torch.nn.Module):
    def __init__(self, params):
        super(pcann, self).__init__()
        self.params = params
        self.listt   = params["layers"]
        self.linear = nn.Linear(self.listt[0], self.listt[-1])

    def forward(self, x):
        x = self.linear(x)
        
        return x
################################################################
# configs
################################################################


print("torch version is ",torch.__version__)
ntrain = 1000#
ntest = 5000#


parser = argparse.ArgumentParser(description='parse batch_size, epochs and resolution')
parser.add_argument('--bs',  default=100, type = int, help='batch-size')#
parser.add_argument('--res', default=512, type = int, help='resolution')#


parser.add_argument('--lr', default=1e-3, type = float, help='learning rate')#
parser.add_argument('--gm', default=0.1, type = float, help='gamma')#
parser.add_argument('--ss', default=10000, type = int, help='step size')#

parser.add_argument('--ep', default=8000, type = int, help='epochs')#
parser.add_argument('--wd', default=1.5e-2, type = float, help='weight decay')#
parser.add_argument('--dX', default=500, type = int, help='input reduced dimension')
parser.add_argument('--dY', default=500, type = int, help='output reduced dimension')
parser.add_argument('--algo', default='adam', type = str, help='optim algo used: adam or sgd')
parser.add_argument('--data', default='darcy_pwc', type = str, help='data used: darcy_pwc, darcy_lognorm, poisson')

args = parser.parse_args()

batch_size = args.bs #100
res = args.res + 1

learning_rate = args.lr
gamma = args.gm
step_size = args.ss

epochs = args.ep #500
wd = args.wd
dX = args.dX
dY = args.dX#args.dY
algo = args.algo
data = args.data

#print("\nUsing batchsize = %s, epochs = %s, and resolution = %s\n"%(batch_size, epochs, res))
params = {}
params["xmin"] = 0
params["ymin"] = 0
params["xmax"] = 1
params["ymax"] = 1

params["layers"] = [dX , 500, 1000, 2000, 1000, 500, dY]
################################################################
# load data and data normalization
################################################################
if data == 'darcy_pwc':
    PATH = "/localdata/Derick/stuart_data/Darcy_421/new_aUP_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
if data == 'darcy_lognorm':
    PATH = "/localdata/Derick/stuart_data/Darcy_421/aUL_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
if data == 'poisson':
    PATH = "/localdata/Derick/stuart_data/Darcy_421/fUG_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"



#Read Datasets
X_train, Y_train, X_test, Y_test = readtoArray(PATH, 1024, 5000, Nx = 512, Ny = 512)

print ("Converting dataset to numpy array.")
tt = time.time()
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test  = np.array(X_test )
Y_test  = np.array(Y_test )
print ("    Conversion completed after %.2f minutes"%((time.time()-tt)/60))

print ("Subsampling dataset to the required resolution.")
tt = time.time()
X_train = SubSample(X_train, res, res)
Y_train = SubSample(Y_train, res, res)
X_test  = SubSample(X_test , res, res)
Y_test  = SubSample(Y_test , res, res)
print ("    Subsampling completed after %.2f minutes"%((time.time()-tt)/60))

print ("Taking out the required train/test size.")
tt = time.time()
x_train = torch.from_numpy(X_train[:ntrain, :, :].reshape(ntrain, -1)).float()
y_train = torch.from_numpy(Y_train[:ntrain, :, :].reshape(ntrain, -1)).float()
x_test  = torch.from_numpy(X_test [ :ntest, :, :].reshape(ntest , -1)).float()
y_test  = torch.from_numpy(Y_test [ :ntest, :, :].reshape(ntest , -1)).float()
print ("    Taking completed after %s seconds"%(time.time()-tt))
print("...")

print ("Obtaining the pre-PCA standardise functions")   
tt = time.time()   
scalerX = StandardScaler().fit(x_train)  ###
scalerY = StandardScaler().fit(y_train)  ### 
print ("    Obtaining the pre-PCA standardise functions done after %.4f seconds"%(time.time()-tt))

x_train_stand = scalerX.transform(x_train) ###
y_train_stand = scalerY.transform(y_train) ###
# x_test_stand  = scalerX.transform(x_test) ###
# y_test_stand  = scalerY.transform(y_test) ###

print ("Obtaining the PCA functions")   
tt = time.time()   
pcaX = PCA(n_components = dX, svd_solver = 'full').fit(x_train)
pcaY = PCA(n_components = dY, svd_solver = 'full').fit(y_train)   
pcaX_stand = PCA(n_components = dX, svd_solver = 'full').fit(x_train_stand)
pcaY_stand = PCA(n_components = dY, svd_solver = 'full').fit(y_train_stand)      
print ("    Obtaining the PCA functions done after %s seconds"%(time.time()-tt)) 

#var= pcaX.explained_variance_ratio_

#Cumulative Variance explains
var1 = np.cumsum(np.round(pcaX.explained_variance_ratio_, decimals=4)*100)
var2 = np.cumsum(np.round(pcaY.explained_variance_ratio_, decimals=4)*100)
var1_stand = np.cumsum(np.round(pcaX_stand.explained_variance_ratio_, decimals=4)*100)
var2_stand = np.cumsum(np.round(pcaY_stand.explained_variance_ratio_, decimals=4)*100)


plt.plot(var1, label = 'pcaX unstandardised')
plt.plot(var2, label = 'pcaY unstandardised')
plt.plot(var1_stand, label = 'pcaX standardised', linestyle = '-.' )
plt.plot(var2_stand, label = 'pcaY standardised', linestyle = '-.' )
plt.xlabel('Principal Component')#, fontsize=16, labelpad=15)
plt.ylabel('Cumulative Proportion of Variance Explained')
plt.grid()
plt.legend(loc = 'lower right')
if data == 'darcy_pwc':
    plt.title('Darcy PWC')
    plt.savefig('figures/darcy pwc data 1000 full svd.png',dpi=500)
if data == 'darcy_lognorm':
    plt.title('Darcy LogNorm')
    plt.savefig('figures/darcy lognorm data 1000 full svd.png',dpi=500)
if data == 'poisson':
    plt.title('Poisson')
    plt.savefig('figures/poisson data 1000 full svd.png',dpi=500)