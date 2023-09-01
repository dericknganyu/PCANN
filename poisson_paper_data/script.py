#runnung fno.py for different reolutions on different gpus

from ast import Str
import subprocess
import argparse

resolution = [64, 128, 256, 512] #[512] #

for res in resolution:
    reduced_dim = [200]#[50, 70, 150, 200] #[50, 70, 100, 150, 200, 250] #[70, 50]#[30, 50, 70, 100, 150, 200, 250] #[30, 50, 70, 100]#
    gpu_infos   = [0]#[5, 6, 7, 4]#[0 , 1 , 2  , 3  ] #[0, 7]#[1 , 2 , 3 , 4  , 5  , 6  , 7  ] #[0, 1 , 2 , 3] #[1 , 2 , 3 , 4  , 5  , 6  , 7  ] #

    parser = argparse.ArgumentParser(description='parse mode')
    parser.add_argument('--mode' , default ='fwd', type = str, help='fwd for forward, inv for inverse')#
    parser.add_argument('--algo', default='sgd', type = str, help='optim algo used: adam or sgd')
    #parser.add_argument('--norm' , default='UG'   , type = str, help='G for Gaussian, UG for unit gaussian')#
    #parser.add_argument('--res' , default=64   , type = int, help='resolutions factors of 512')#

    args = parser.parse_args()
    #res  = args.res

    if args.mode == 'fwd':
        name_screen    = 'pcann-snn2_poisson_algo5_%s_res_%s_rd_'%(args.algo, res)
        command_screen = 'python pcann-snn2.py --algo %s --res %s --dX '%(args.algo, res)

    if args.mode == 'inv':
        name_screen    = 'inv-pcann_poisson_algo2_%s_res_%s_rd_'%(args.algo, res)
        command_screen = 'python inv-pcann-snn2.py --algo %s --res %s --dX '%(args.algo, res)

    if args.mode == 'fwd-lin':
        name_screen    = 'linear_poisson_algo_%s_res_%s_rd_'%(args.algo, res)
        command_screen = 'python linear.py --algo %s --res %s --dX '%(args.algo, res)

    if args.mode == 'inv-lin':
        name_screen    = 'inv-linear_poisson_algo_%s_res_%s_rd_'%(args.algo, res)
        command_screen = 'python inv-linear.py --algo %s --res %s --dX '%(args.algo, res)

    for rd, gpu_in in zip(reduced_dim, gpu_infos):

        screen_name = name_screen + str(rd)
        command     = command_screen + '%s'%(rd) #'python pcann-gauss-norm.py --dX %s'%(rd) #
        subprocess.run('screen -dmS '+screen_name, shell=True)
        
        subprocess.run('screen -r  %s -p 0 -X stuff "%s^M"'%(screen_name, 'conda activate base'), shell=True)
        subprocess.run('screen -r  %s -p 0 -X stuff "%s^M"'%(screen_name, 'export CUDA_VISIBLE_DEVICES=%s'%(gpu_in)), shell=True)
        subprocess.run('screen -r  %s -p 0 -X stuff "%s^M"'%(screen_name, command), shell=True)

