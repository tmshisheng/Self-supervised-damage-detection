# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 21:13:54 2022

@author: Sheng Shi
"""


import torch
import numpy as np
from utils_MyNet_train.MyNet_train3 import MyNet,Test_MyNet
import os
import random

# Hyper Parameters
window_size            = 50  # 50/100/200/400/800
batch_size             = 128
dt                     = 0.05

## Input and Output Dimension
in_channel = [4,5,12,13]
outputdim  = len(in_channel)
lspace2    = int(3) # 3/6/9


def DataTransf(data,window_size,window_size_step = 1):
    step_skip = int(window_size/window_size_step)
    train = data
    nw = int(train.shape[0]/window_size_step-step_skip)
    data = np.zeros([outputdim*nw,window_size])
    train_y = []
    for i in range(nw):
        for j in range(outputdim):
            data[i*outputdim+j,:] = train[i*window_size_step: (i*window_size_step+window_size), j]
            train_y.append(j)
    data = np.array(data).astype('float64')
    train_x = torch.from_numpy(data)
    train_y = torch.tensor(train_y)
    return train_x[:-1,:],train_x[:-1,:],train_y[:-1],train_x[1:,-1].unsqueeze(1)

def DamageOcccurance(feature):
    indicator = []
    for i in range(int(feature.shape[0]/4)):
        matrix_temp = feature[(i)*4:(i+1)*4,:]
        indicator_temp = matrix_temp==np.max(matrix_temp,axis=1)
        indicator_temp = indicator_temp==[[True,False,False,False],
                                         [False,True,False,False],
                                         [False,False,True,False],
                                         [False,False,False,True]]
        if indicator_temp.all():
            indicator.append(0)
        else:
            indicator.append(1)
    return indicator

# Train Predictor
mad0_1 = []; mad0_2 = []; mad0_3 = []; mad1 = []; mad2 = []; mad3 = []; mad4 = []; mad5 = []; mad6 = []
RL_all_0_1=[]; RL_all_0_2=[]; RL_all_0_3=[]; RL_all_1=[]; RL_all_2=[]; RL_all_3=[]; RL_all_4=[]; RL_all_5=[]; RL_all_6=[];
PL_all_0_1=[]; PL_all_0_2=[]; PL_all_0_3=[]; PL_all_1=[]; PL_all_2=[]; PL_all_3=[]; PL_all_4=[]; PL_all_5=[]; PL_all_6=[];
LF_all_0_1=[]; LF_all_0_2=[]; LF_all_0_3=[]; LF_all_1=[]; LF_all_2=[]; LF_all_3=[]; LF_all_4=[]; LF_all_5=[]; LF_all_6=[];
for noise_level in [0]:
    print('Loading trained predictor ...')
    data0_1 = np.loadtxt('{}/train.csv'.format(noise_level),dtype="float",delimiter=',')[:,in_channel]
    scale = max(np.sqrt(np.mean(data0_1**2,axis=0)))
    data0_2 = np.loadtxt('{}/validation.csv'.format(noise_level),dtype="float",delimiter=',')[:,in_channel]
    data0_1 = data0_1/scale
    data0_2 = data0_2/scale
    Train_x,Train1_y,Train2_y,Train3_y = DataTransf(data0_1,window_size,1)
    Valid_x,Valid1_y,Valid2_y,Valid3_y = DataTransf(data0_2,window_size,1)
    best_net = torch.load('Trained_models/%d/predictor4_%d_%d.pt' %(noise_level,window_size,lspace2))

    ## Undamaged feature for training 
    print('Generating features for undamaged training data ...')
    RL0_1,CL0_1,PL0_1,LF0_1,LFdots0_1 = Test_MyNet(best_net,Train_x,Train1_y,Train2_y,Train3_y)
    
    ## Undamaged feature for validation 
    print('Generating features for undamaged validation data ...')
    RL0_2,CL0_2,PL0_2,LF0_2,LFdots0_2 = Test_MyNet(best_net,Valid_x,Valid1_y,Valid2_y,Valid3_y)
    
    ## Undamaged feature for testing 
    print('Generating features for undamaged testing data')
    data0_3 = np.loadtxt('{}/test0.csv'.format(noise_level),dtype="float",delimiter=',')[:,in_channel]
    data0_3 = data0_3/scale
    Test0_x,Test0_y1,Test0_y2,Test0_y3 = DataTransf(data0_3,window_size)
    RL0_3,CL0_3,PL0_3,LF0_3,LFdots0_3 = Test_MyNet(best_net,Test0_x,Test0_y1,Test0_y2,Test0_y3)
    
    ## Damaged feature for testing (DP 1)
    print('Generating features for DP1')
    data1 = np.loadtxt('{}/test{}.csv'.format(noise_level,1),dtype="float",delimiter=',')[:,in_channel]
    data1 = data1/scale
    Test1_x,Test1_y1,Test1_y2,Test1_y3= DataTransf(data1,window_size)
    RL1,CL1,PL1,LF1,LFdots1 = Test_MyNet(best_net,Test1_x,Test1_y1,Test1_y2,Test1_y3)
    
    ## Damaged feature for testing (DP 2)
    print('Generating features for DP2')
    data2 = np.loadtxt('{}/test{}.csv'.format(noise_level,2),dtype="float",delimiter=',')[:,in_channel]
    data2 = data2/scale
    Test2_x,Test2_y1,Test2_y2,Test2_y3= DataTransf(data2,window_size)
    RL2,CL2,PL2,LF2,LFdots2 = Test_MyNet(best_net,Test2_x,Test2_y1,Test2_y2,Test2_y3)
    
    ## Damaged feature for testing (DP 3)
    print('Generating features for DP3')
    data3 = np.loadtxt('{}/test{}.csv'.format(noise_level,3),dtype="float",delimiter=',')[:,in_channel]
    data3 = data3/scale
    Test3_x,Test3_y1,Test3_y2,Test3_y3 = DataTransf(data3,window_size)
    RL3,CL3,PL3,LF3,LFdots3 = Test_MyNet(best_net,Test3_x,Test3_y1,Test3_y2,Test3_y3)

    ## Damaged feature for testing (DP 4)
    print('Generating features for DP4')
    data4 = np.loadtxt('{}/test{}.csv'.format(noise_level,4),dtype="float",delimiter=',')[:,in_channel]
    data4 = data4/scale
    Test4_x,Test4_y1,Test4_y2,Test4_y3 = DataTransf(data4,window_size)
    RL4,CL4,PL4,LF4,LFdots4 = Test_MyNet(best_net,Test4_x,Test4_y1,Test4_y2,Test4_y3)

    ## Damaged feature for testing (DP 5)
    print('Generating features for DP5')
    data5 = np.loadtxt('{}/test{}.csv'.format(noise_level,5),dtype="float",delimiter=',')[:,in_channel]
    data5 = data5/scale
    Test5_x,Test5_y1,Test5_y2,Test5_y3 = DataTransf(data5,window_size)
    RL5,CL5,PL5,LF5,LFdots5 = Test_MyNet(best_net,Test5_x,Test5_y1,Test5_y2,Test5_y3)

    ## Damaged feature for testing (DP 6)
    print('Generating features for DP6')
    data6 = np.loadtxt('{}/test{}.csv'.format(noise_level,6),dtype="float",delimiter=',')[:,in_channel]
    data6 = data6/scale
    Test6_x,Test6_y1,Test6_y2,Test6_y3 = DataTransf(data6,window_size)
    RL6,CL6,PL6,LF6,LFdots6 = Test_MyNet(best_net,Test6_x,Test6_y1,Test6_y2,Test6_y3)
    
    # Detection of damage occurance
    print('Calculating mads ...' )
    ## for CL0_1
    indicator0_1 = DamageOcccurance(CL0_1)
    mad0_1_temp       = sum(indicator0_1)/len(indicator0_1)
    ## for CL0_2
    indicator0_2 = DamageOcccurance(CL0_2)
    mad0_2_temp       = sum(indicator0_2)/len(indicator0_2)
    ## for CL0_3
    indicator0_3 = DamageOcccurance(CL0_3)
    mad0_3_temp       = sum(indicator0_3)/len(indicator0_3)
    ## for CL1
    indicator1 = DamageOcccurance(CL1)
    mad1_temp       = sum(indicator1)/len(indicator1)
    ## for CL2
    indicator2 = DamageOcccurance(CL2)
    mad2_temp       = sum(indicator2)/len(indicator2)
    ## for CL3
    indicator3 = DamageOcccurance(CL3)
    mad3_temp       = sum(indicator3)/len(indicator3)
    ## for CL4
    indicator4 = DamageOcccurance(CL4)
    mad4_temp       = sum(indicator4)/len(indicator4)
    ## for CL5
    indicator5 = DamageOcccurance(CL5)
    mad5_temp       = sum(indicator5)/len(indicator5)
    ## for CL6
    indicator6 = DamageOcccurance(CL6)
    mad6_temp       = sum(indicator6)/len(indicator6)
    
    mad0_1.append(mad0_1_temp)
    mad0_2.append(mad0_2_temp)
    mad0_3.append(mad0_3_temp)
    mad1.append(mad1_temp)
    mad2.append(mad2_temp)
    mad3.append(mad3_temp)
    mad4.append(mad4_temp)
    mad5.append(mad5_temp)
    mad6.append(mad6_temp)
    
    RL_all_0_1.append(RL0_1)
    RL_all_0_2.append(RL0_2)
    RL_all_0_3.append(RL0_3)
    RL_all_1.append(RL1)
    RL_all_2.append(RL2)
    RL_all_3.append(RL3)
    RL_all_4.append(RL4)
    RL_all_5.append(RL5)
    RL_all_6.append(RL6)
    
    PL_all_0_1.append(PL0_1)
    PL_all_0_2.append(PL0_2)
    PL_all_0_3.append(PL0_3)
    PL_all_1.append(PL1)
    PL_all_2.append(PL2)
    PL_all_3.append(PL3)
    PL_all_4.append(PL4)
    PL_all_5.append(PL5)
    PL_all_6.append(PL6)
    
    LF_all_0_1.append(LF0_1)
    LF_all_0_2.append(LF0_2)
    LF_all_0_3.append(LF0_3)
    LF_all_1.append(LF1)
    LF_all_2.append(LF2)
    LF_all_3.append(LF3)
    LF_all_4.append(LF4)
    LF_all_5.append(LF5)
    LF_all_6.append(LF6)
    
mad = np.array([mad0_1,mad0_2,mad0_3,mad1,mad2,mad3,mad4,mad5,mad6]).T
RL  = np.array([RL_all_0_1,RL_all_0_2,RL_all_0_3,RL_all_1,RL_all_2,RL_all_3,RL_all_4,RL_all_5,RL_all_6]).T
PL  = np.array([PL_all_0_1,PL_all_0_2,PL_all_0_3,PL_all_1,PL_all_2,PL_all_3,PL_all_4,PL_all_5,PL_all_6]).T
LF  = np.array([LF_all_0_1,LF_all_0_2,LF_all_0_3,LF_all_1,LF_all_2,LF_all_3,LF_all_4,LF_all_5,LF_all_6]).T