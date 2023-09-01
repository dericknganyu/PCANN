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
parser.add_argument('--bs',  default=1000, type = int, help='batch-size')#
parser.add_argument('--res', default=512, type = int, help='resolution')#


parser.add_argument('--lr', default=2.6e-5, type = float, help='learning rate')#
parser.add_argument('--gm', default=0.1, type = float, help='gamma')#
parser.add_argument('--ss', default=100, type = int, help='step size')#

parser.add_argument('--ep', default=8000, type = int, help='epochs')#
parser.add_argument('--wd', default=0.0870, type = float, help='weight decay')#
parser.add_argument('--dX', default=100, type = int, help='input reduced dimension')
parser.add_argument('--dY', default=100, type = int, help='output reduced dimension')
parser.add_argument('--algo', default='sgd', type = str, help='optim algo used: adam or sgd')
parser.add_argument('--pdrop', default=0.001, type = float, help='drop out')#0.005
parser.add_argument('--fac', default=0.05, type = float, help='factor')#0.005
parser.add_argument('--pat', default=10, type = int, help='patience')#

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
p_drop = args.pdrop
factor = args.fac
patience = args.pat

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
PATH = "/localdata/Derick/stuart_data/Darcy_421/new_aUP_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
resultPATH = "pcalin-inv/"
modelPATH  = "pcalin-inv/"
#Read Datasets
Y_train, X_train, _, _ = readtoArray(PATH, 1024, 5000, Nx = 512, Ny = 512)

print ("Converting dataset to numpy array.")
tt = time.time()
X_train = np.array(X_train)
Y_train = np.array(Y_train)
print ("    Conversion completed after %.2f minutes"%((time.time()-tt)/60))

print ("Subsampling dataset to the required resolution.")
tt = time.time()
X_train = SubSample(X_train, res, res)
Y_train = SubSample(Y_train, res, res)
print ("    Subsampling completed after %.2f minutes"%((time.time()-tt)/60))

print ("Taking out the required train/test size.")
tt = time.time()
x_train = torch.from_numpy(X_train[:ntrain, :, :].reshape(ntrain, -1)).float()
y_train = torch.from_numpy(Y_train[:ntrain, :, :].reshape(ntrain, -1)).float()
print ("    Taking completed after %.4f seconds"%(time.time()-tt))
print("...")

print ("Obtaining the pre-PCA standardise functions")   
tt = time.time()   
#scalerX = GaussianNormalizer1(x_train) #UnitGaussianNormalizer1(x_train) #StandardScaler().fit(x_train)  #MinMaxScaler().fit(x_train) #StandardScaler().fit(x_train)  ###
#scalerY = GaussianNormalizer1(y_train) #UnitGaussianNormalizer1(y_train) # StandardScaler().fit(y_train)  #MinMaxScaler().fit(y_train) #StandardScaler().fit(y_train)  ###
print ("    Obtaining the pre-PCA standardise functions done after %.4f seconds"%(time.time()-tt))

# x_train = scalerX.transform(x_train) ###scalerX.encode(x_train) #
# y_train = scalerY.transform(y_train) ###scalerY.encode(y_train) #
# x_test  = scalerX.transform(x_test) ###scalerX.encode(x_test) #
# y_test  = scalerY.transform(y_test) ###scalerY.encode(y_test) #

print ("Obtaining the PCA functions")   
tt = time.time()   
pcaX = PCA(n_components = dX, random_state = 0).fit(x_train) #PCA(n_components = dX, svd_solver = 'full').fit(x_train) # 
pcaY = PCA(n_components = dY, random_state = 0).fit(y_train) #  PCA(n_components = dY, svd_solver = 'full').fit(y_train) # 
print ("    Obtaining the PCA functions done after %.4f seconds"%(time.time()-tt))


###  

x_train = torch.from_numpy(pcaX.transform(x_train)).float().to(device)
y_train = torch.from_numpy(pcaY.transform(y_train)).float().to(device) 


x_normalizer = UnitGaussianNormalizer(x_train)
#x_train = x_normalizer.encode(x_train)

y_normalizer = UnitGaussianNormalizer(y_train)
#y_train = y_normalizer.encode(y_train)

fichier_out = CheckExist(resultPATH+'models')
#x_train = x_train.reshape(ntrain,res,res,1)
#x_test = x_test.reshape(ntest,res,res,1)

step_size = 2500
gamma = 0.01
wd = 0.15
batch_size = 200
learning_rate = 0.001
p_drop= 0
test_l2 = 0.220275
TIMESTAMP = '20221017-224747-902802'

#last_model_sgd_513~res_0.007366~RelL2TestError_150~rd_0~pdrop_1000~ntrain_5000~ntest_500~BatchSize_0.000000500~LR_0.1000~Reg_0.5000~gamma_2000~Step_20000~epochs_20221011-181648-522053
################################################################
# training and evaluation
################################################################
model = pcann(params).cuda()
print("Model has %s parameters"%(count_params(model)))

if algo == 'adam':    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
if algo == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99, dampening=0, weight_decay=wd, nesterov=True)


myloss = LpLoss(size_average=False)
y_normalizer.cuda()



ModelInfos = "_%03d"%(res)+"~res_"+str(np.round(test_l2,6))+"~RelL2TestError_"+str(dX)+"~rd_"+str(ntrain)+"~ntrain_"+str(ntest)+"~ntest_"+str(batch_size)+"~BatchSize_"+str(learning_rate)+\
                                "~LR_"+str(wd)+"~Reg_"+str(gamma)+"~gamma_"+str(step_size)+"~Step_"+str(epochs)+"~epochs_"+TIMESTAMP


model.load_state_dict(torch.load(modelPATH+"models/last_model"+ModelInfos+".pt"))
model.eval()

print()
print()

#Just a file containing data sampled in same way as the training and test dataset
fileName = "new_aUP_Square_TrainData=1_TestData=1_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
U_train, F_train, U_test, F_test= readtoArray(fileName, 1, 1, 512, 512)
F_train = SubSample(F_train, res, res)
U_train = SubSample(U_train, res, res)
ff = np.array(F_train[0])

print("Starting the Verification with Sampled Example")
U_FDM = np.array(U_train[0])

for i in range(25):
    print("      Doing PCANN on Example...")
    tt = time.time()

    inPCANN = ff.reshape(1, -1)
    #inPCANN = scalerX.transform(inPCANN) ###
    inPCANN = pcaX.transform(inPCANN)
    inPCANN = torch.from_numpy(inPCANN).float().to(device)
    inPCANN = model(x_normalizer.encode(inPCANN))
    inPCANN = y_normalizer.decode(inPCANN)
    inPCANN = inPCANN.detach().cpu().numpy()

    inPCANN = pcaY.inverse_transform(inPCANN) ###
    #inPCANN = scalerY.inverse_transform(inPCANN)
    U_PCANN = inPCANN.reshape(res, res) 
    print("            PCANN completed after %s"%(time.time()-tt))

myLoss = LpLoss(size_average=False)
print()
print("Ploting comparism of FDM and PCANN Simulation results")
fig = plt.figure(figsize=((5+2)*4, 5))

fig.suptitle("Plot of $- \Delta u = f(x, y)$ on $\Omega = ]0,1[ x ]0,1[$ with $u|_{\partial \Omega}  = 0.$")

colourMap = parula() #plt.cm.jet #plt.cm.coolwarm

plt.subplot(1, 4, 1)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("Input")
plt.imshow(F_train[0], cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

plt.subplot(1, 4, 2)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("Truth")
plt.imshow(U_FDM, cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

plt.subplot(1, 4, 3)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("PCALin")
plt.imshow(U_PCANN, cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

plt.subplot(1, 4, 4)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("Truth-PCALin, RelL2Err = "+str(round(myLoss.rel_single(U_PCANN, U_FDM).item(), 3)))
plt.imshow(np.abs(U_FDM - U_PCANN), cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

plt.savefig(resultPATH+'figures/compare'+ModelInfos+'.png',dpi=500)

fig = plt.figure(figsize=((5+1)*2, 5))

plt.subplot(1, 2, 1)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("PCALin")
plt.imshow(U_PCANN, cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

plt.subplot(1, 2, 2)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("Truth-PCALin, RelL2Err = "+str(round(myLoss.rel_single(U_PCANN, U_FDM).item(), 3)))
plt.imshow(np.abs(U_FDM - U_PCANN), cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

fig.tight_layout()
plt.savefig(resultPATH+'figures/PCALin-DarcyPWC-Inverse-UP>png',dpi=500)

