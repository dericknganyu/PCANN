import sys
sys.path.append('../')
from pca_models import *

torch.manual_seed(0)
np.random.seed(0)


################################################################
# configs
################################################################


print("torch version is ",torch.__version__)
ntrain = 1000#
ntest = 5000#


parser = argparse.ArgumentParser(description='parse batch_size, epochs and resolution')
parser.add_argument('--bs',  default=1000, type = int, help='batch-size')#
parser.add_argument('--res', default=63, type = int, help='resolution')#


parser.add_argument('--lr', default=2.6e-5, type = float, help='learning rate')#
parser.add_argument('--gm', default=0.1, type = float, help='gamma')#
parser.add_argument('--ss', default=100, type = int, help='step size')#

parser.add_argument('--ep', default=10000, type = int, help='epochs')#
parser.add_argument('--wd', default=0.0870, type = float, help='weight decay')#
parser.add_argument('--dX', default=70, type = int, help='input reduced dimension')
parser.add_argument('--dY', default=70, type = int, help='output reduced dimension')
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
PATH = "../../../../../../localdata/Derick/stuart_data/Darcy_421/StructuralMechanics_TrainData=1000_TestData=5000_Resolution=41X41_Domain=[0,1]X[0,1].hdf5"
resultPATH = "pcann-inv/"#"/localdata/Derick/stuart_data/Darcy_421/poisson/--inv2/RN/"
#Read Datasets
Y_train, X_train, Y_test, X_test = readtoArray(PATH, 1024, 5000, Nx = 512, Ny = 512)

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
print ("    Taking completed after %.4f seconds"%(time.time()-tt))
print("...")

# print ("Obtaining the pre-PCA standardise functions")   
# tt = time.time()   
# scalerX = StandardScaler().fit(x_train)  #GaussianNormalizer1(x_train) #UnitGaussianNormalizer1(x_train) #StandardScaler().fit(x_train)  #MinMaxScaler().fit(x_train) #StandardScaler().fit(x_train)  ###
# scalerY = StandardScaler().fit(y_train)  #GaussianNormalizer1(y_train) #UnitGaussianNormalizer1(y_train) # StandardScaler().fit(y_train)  #MinMaxScaler().fit(y_train) #StandardScaler().fit(y_train)  ###
# print ("    Obtaining the pre-PCA standardise functions done after %.4f seconds"%(time.time()-tt))

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
x_test  = torch.from_numpy(pcaX.transform(x_test)).float().to(device)
y_test  = torch.from_numpy(pcaY.transform(y_test)).float().to(device)


x_normalizer = UnitGaussianNormalizer(x_train) #RangeNormalizer(x_train) #
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train) #RangeNormalizer(y_train) #
y_train = y_normalizer.encode(y_train)

fichier_out = CheckExist(resultPATH+'models')
#x_train = x_train.reshape(ntrain,res,res,1)
#x_test = x_test.reshape(ntest,res,res,1)
for step_size in [2500]:#[2500, 5000, 10000]:
    for gamma in [0.1]:#[1, 1e-1, 1e-2, 1e-3, 1e-4]:
        for wd in [0.1]:#0.25, 0.1, 0.5, 0.05]:#[0.25]:#[0.1, 0.15, 0.01, 0.015, 0.001]:#[1.5e-2, 1e-1, 1.5e-1]: #
            for batch_size in [500]:#[500, 1000]:#[100, 500, 1000]:#[100, 200, 500, 1000]:
                for learning_rate in [1e-5]:#1e-3, 5e-3, 1e-4, 5e-4] + [1e-5, 5e-5, 1e-6, 5e-6, 1e-7, 5e-7, 1e-8, 5e-8]:#[1e-6, 5e-6, 1e-7, 5e-7]:#[1e-5]:#[1e-6, 1e-5]:#[1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                    for p_drop in [0.001]:# 0]:#[0, 0.01, 0.05, 0.001]:
# for step_size in [2500]:#[2500, 5000, 10000]:
#     for gamma in [0.1]:#[1, 1e-1, 1e-2, 1e-3, 1e-4]:
#         for wd in [0.1]:#[0.1, 0.15, 0.01, 0.015, 0.001]:#[1.5e-2, 1e-1, 1.5e-1]: #
#             for batch_size in [500]:#[100, 500, 1000]:#[100, 200, 500, 1000]:
#                 for learning_rate in [1e-5]:#[1e-4, 2.5e-4, 1e-3]:#[1e-5, 2.5e-5, 1e-6]:#[1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:#[1e-3, 2.5e-3, 1e-2]:
#                     for p_drop in [0.001]:#[0, 0.01, 0.05, 0.001]:
                        torch.manual_seed(0)
                        np.random.seed(0)
                        testInfos = "RelL2TestError_"+str(dX)+"~rd_"+str(ntrain)+"~ntrain_"+str(ntest)+"~ntest_"+str(batch_size)+"~BatchSize_"+str(learning_rate)+\
                                "~LR_"+str(wd)+"~Reg_"+str(gamma)+"~gamma_"+str(step_size)+"~Step_"+str(epochs)+"~epochs"
                        if testInfos in fichier_out:
                            print ("Already exists, skipping")
                            continue
                        if gamma == 1 and step_size !=2500:
                            print ("Repetition, skipping")
                            continue

                        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=False)#shuffle=True)
                        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

                        ################################################################
                        # training and evaluation
                        ################################################################
                        model = pcann(in_features=dX, out_features=dY, p_drop=p_drop, use_selu=True).cuda()#pcann(modes, modes, width).cuda()
                        print("Model has %s parameters"%(count_params(model)))

                        if algo == 'adam':    
                            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
                        if algo == 'sgd':
                            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99, dampening=0, weight_decay=wd, nesterov=True)
                        
                        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
                        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience, verbose=True)

                        myloss = LpLoss(size_average=False)
                        y_normalizer.cuda()
                        TIMESTAMP = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')#time.strftime("%Y%m%d-%H%M%S-%f")
                        if os.path.isfile(resultPATH+'files/lossData_'+TIMESTAMP+'.txt'):
                            os.remove(resultPATH+'files/lossData_'+TIMESTAMP+'.txt')
                        if os.path.isfile(resultPATH+'files/lossTestData_'+TIMESTAMP+'.txt'):
                            os.remove(resultPATH+'files/lossTestData_'+TIMESTAMP+'.txt')   
                        for ep in range(epochs):
                            model.train()
                            t1 = default_timer()
                            train_l2 = 0


                            for x, y in train_loader:
                                x, y = x.cuda(), y.cuda()

                                optimizer.zero_grad()
                                out = model(x)#.reshape(batch_size, res, res)
                                out = y_normalizer.decode(out)
                                y = y_normalizer.decode(y)

                                loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
                                
                                if loss.isnan() or loss.isinf():
                                    print('NaN or Inf loss found, skipping | loss is %s'%(loss.item()))
                                    break

                                loss.backward()

                                optimizer.step()
                                
                                #>>>>>>
                                if ep == epochs-1:
                                    out = pcaY.inverse_transform(out.detach().cpu().numpy())
                                    #out = scalerY.inverse_transform(out)
                                    out = torch.from_numpy(out).float().to(device)

                                    y = pcaY.inverse_transform(y.detach().cpu().numpy())
                                    #y = scalerY.inverse_transform(y)
                                    y = torch.from_numpy(y).float().to(device)

                                    loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
                                #>>>>>>
                                train_l2 += loss.item()

                            if loss.isnan() or loss.isinf():
                                break

                            scheduler.step(loss)

                            model.eval()
                            test_l2 = 0.0
                            with torch.no_grad():

                                for x, y in test_loader:
                                    x, y = x.cuda(), y.cuda()

                                    out = model(x)#.reshape(batch_size, res, res)
                                    out = y_normalizer.decode(out)
                                    #>>>>>>
                                    if ep == epochs-1:
                                        out = pcaY.inverse_transform(out.detach().cpu().numpy())
                                        #out = scalerY.inverse_transform(out)
                                        out = torch.from_numpy(out).float().to(device)

                                        y = pcaY.inverse_transform(y.detach().cpu().numpy())
                                        #y = scalerY.inverse_transform(y)
                                        y = torch.from_numpy(y).float().to(device)
                                    #>>>>>>

                                    test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1))#.item()
                            #scheduler.step(test_l2)
                            train_l2/= ntrain
                            test_l2 /= ntest
                            test_l2 = test_l2.item()

                            t2 = default_timer()
                            if ep%10 == 10-1:
                                print("epoch: %s, completed in %.4f seconds. Training Loss: %.4f and Test Loss: %.4f"%(ep+1, t2-t1, train_l2, test_l2))

                                file = open(resultPATH+'files/lossData_'+TIMESTAMP+'.txt',"a")
                                file.write(str(ep+1)+" "+str(train_l2)+" "+str(test_l2)+"\n")

                            if algo == 'adam':    
                                ModelInfos = "_inv_adam_%03d"%(res)+"~res_%.6f"%(test_l2)+"~RelL2TestError_"+str(dX)+"~rd_"+str(p_drop)+"~pdrop_"+str(ntrain)+"~ntrain_"+str(ntest)+"~ntest_"+str(batch_size)+"~BatchSize_%.10f"%(learning_rate)+\
                                        "~LR_%.4f"%(wd)+"~Reg_%.4f"%(gamma)+"~gamma_%s"%(step_size)+"~Step_"+str(epochs)+"~epochs_"+datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f') 

                            if algo == 'sgd':
                                ModelInfos = "_inv_sgd_%03d"%(res)+"~res_%.6f"%(test_l2)+"~RelL2TestError_"+str(dX)+"~rd_"+str(p_drop)+"~pdrop_"+str(ntrain)+"~ntrain_"+str(ntest)+"~ntest_"+str(batch_size)+"~BatchSize_%.10f"%(learning_rate)+\
                                        "~LR_%.4f"%(wd)+"~Reg_%.4f"%(gamma)+"~gamma_%s"%(step_size)+"~Step_"+str(epochs)+"~epochs_"+datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f') 

                        if loss.isnan() or loss.isinf():
                            continue
                        
                        torch.save(model.state_dict(), resultPATH+"models/last_model"+ModelInfos+".pt")
                        os.rename(resultPATH+'files/lossData_'+TIMESTAMP+'.txt', resultPATH+'files/lossData'+ModelInfos+'.txt')

                        dataLoss = np.loadtxt(resultPATH+'files/lossData'+ModelInfos+'.txt')
                        
                        stepTrain = dataLoss[:,0] #Reading Epoch                  
                        errorTrain = dataLoss[:,1] #Reading erros
                        errorTest  = dataLoss[:,2]

                        print("Ploting Loss VS training step...")
                        fig = plt.figure(figsize=(15, 10))
                        plt.yscale('log')
                        plt.plot(stepTrain, errorTrain, label = 'Training Loss')
                        plt.plot(stepTrain , errorTest , label = 'Test Loss')
                        plt.xlabel('epochs')#, fontsize=16, labelpad=15)
                        plt.ylabel('Loss')
                        plt.legend(loc = 'upper right')
                        plt.title("lr = %s test error = %s"%(learning_rate, str(np.round(test_l2,6))))
                        plt.savefig(resultPATH+"figures/Error_VS_TrainingStep"+ModelInfos+".png", dpi=500)


                        #def use_model():#(params, model,device,nSample,params):

                        model.load_state_dict(torch.load(resultPATH+"models/last_model"+ModelInfos+".pt"))
                        model.eval()


                        print()
                        print()

                        #Just a file containing data sampled in same way as the training and test dataset
                        fileName = "StructuralMechanics_TrainData=1_TestData=1_Resolution=41X41_Domain=[0,1]X[0,1].hdf5"
                        U_train, F_train, U_test, F_test= readtoArray(fileName, 1, 1, 512, 512)
                        F_train = SubSample(F_train, res, res)
                        U_train = SubSample(U_train, res, res)
                        ff = np.array(F_train[0])

                        print("Starting the Verification with Sampled Example")
                        U_FDM = np.array(U_train[0])

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
                        U_PCANN = inPCANN.reshape(res, res) #scalerY.decode(inPCANN).reshape(res, res) #
                        print("            PCANN completed after %s"%(time.time()-tt))

                        myLoss = LpLoss(size_average=False)
                        print()
                        print("Ploting comparism of FDM and PCANN Simulation results")
                        fig = plt.figure(figsize=((5+1)*4, 5))
                        fig.set_tight_layout(True)

                        #fig.suptitle("Plot of $- \Delta u = f(x, y)$ on $\Omega = ]0,1[ x ]0,1[$ with $u|_{\partial \Omega}  = 0.$")

                        colourMap = plt.cm.magma # #plt.cm.jet #plt.cm.coolwarm

                        plt.subplot(1, 4, 1)
                        plt.xlabel('x')#, fontsize=16, labelpad=15)
                        plt.ylabel('y')#, fontsize=16, labelpad=15)
                        plt.title("Input")
                        plt.imshow(F_train[0], cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
                        plt.colorbar()#format=OOMFormatter(-5))

                        plt.subplot(1, 4, 2)
                        plt.xlabel('x')#, fontsize=16, labelpad=15)
                        plt.ylabel('y')#, fontsize=16, labelpad=15)
                        plt.title("FDM")
                        plt.imshow(U_FDM, cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
                        plt.colorbar()#format=OOMFormatter(-5))

                        plt.subplot(1, 4, 3)
                        plt.xlabel('x')#, fontsize=16, labelpad=15)
                        plt.ylabel('y')#, fontsize=16, labelpad=15)
                        plt.title("PCANN")
                        plt.imshow(U_PCANN, cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
                        plt.colorbar()#format=OOMFormatter(-5))

                        plt.subplot(1, 4, 4)
                        plt.xlabel('x')#, fontsize=16, labelpad=15)
                        plt.ylabel('y')#, fontsize=16, labelpad=15)
                        plt.title("FDM-PCANN, RelL2Err = "+str(round(myLoss.rel_single(U_PCANN, U_FDM).item(), 3)))
                        plt.imshow(np.abs(U_FDM - U_PCANN), cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto', norm=matplotlib.colors.LogNorm())#)#, vmin=0, vmax=1, )
                        plt.colorbar()#format=OOMFormatter(-5))

                        plt.savefig(resultPATH+'figures/compare'+ModelInfos+'.png',dpi=500)

                        #plt.show()

