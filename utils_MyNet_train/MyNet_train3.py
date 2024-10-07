# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 15:33:47 2021
At University of Toronto
@author: Sheng Shi
"""

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from tqdm import tqdm

# model -----------------------------------------------------------------------
class MyNet(nn.Module):
    def __init__(self,predict_out,k,lspace2):
        super(MyNet,self).__init__()
        lspace1    = int((k+lspace2)/2)
        lspace3    = int((lspace2+4)/2)
        lspace4    = int((lspace2+1)/2)
        # encoder
        self.encoder1 = nn.Linear(k,int(lspace1))
        self.encoder2 = nn.Linear(int(lspace1),int(lspace2))
        # decoder
        self.decoder1 = nn.Linear(int(lspace2),int(lspace1))
        self.decoder2 = nn.Linear(int(lspace1),k)
        # classifier
        self.classifier1 = nn.Linear(int(lspace2),int(lspace3))
        self.classifier2 = nn.Linear(int(lspace3),predict_out)
        # predictior
        self.predictor1  = nn.Linear(int(lspace2),int(lspace4))
        self.predictor2  = nn.Linear(int(lspace4),1)
    def forward(self,input):
        # out1
        out1 = input
        # encoder
        out1 = self.encoder1(out1)
        out1 = nn.LeakyReLU(negative_slope=0.05)(out1)
        out1 = self.encoder2(out1)
        out1 = nn.LeakyReLU(negative_slope=0.05)(out1)
        # decoder
        out2 = self.decoder1(out1)
        out2 = nn.LeakyReLU(negative_slope=0.05)(out2)
        out2 = self.decoder2(out2)
        # classifier
        out3 = self.classifier1(out1)
        out3 = nn.LeakyReLU(negative_slope=0.05)(out3)
        out3 = self.classifier2(out3)
        # predictor
        out4 = self.predictor1(out1)
        out4 = nn.LeakyReLU(negative_slope=0.05)(out4)
        out4 = self.predictor2(out4)
        return out2,out3,out4
    
# test ------------------------------------------------------------------------
def Test_MyNet(net,test_x,test_y1,test_y2,test_y3):
    batch_size = 128
    test_loader = Data.DataLoader(
        dataset = Data.TensorDataset(test_x, test_y1, test_y2, test_y3),
        batch_size = batch_size,
        shuffle = False
        )
    net.eval()
    test_loop = tqdm(test_loader)
    test_loop.set_description('testing')
    lossfun1 = torch.nn.MSELoss()
    lossfun2 = torch.nn.Softmax(1)
    lossfun3 = torch.nn.CrossEntropyLoss()
    i = 1
    with torch.no_grad():
        for test_data_x,test_data1_y,test_data2_y,test_data3_y in test_loop:
            out1,out2,out3 = net(test_data_x)
            test_output1 = lossfun1(out1,test_data1_y)
            test_output2 = lossfun2(out2)
            test_output3 = lossfun1(out3,test_data3_y)
            test_output4 = lossfun3(out2,test_data2_y)
            if i:
                test_output_all1 = test_output1
                test_output_all2 = test_output2
                test_output_all3 = test_output3
                test_output_all4 = test_output4
                i = 0
            else:
                test_output_all1 = np.append(test_output_all1,test_output1)
                test_output_all2 = np.append(test_output_all2,test_output2,0)
                test_output_all3 = np.append(test_output_all3,test_output3)
                test_output_all4 = np.append(test_output_all4,test_output4)
    return np.mean(test_output_all1),test_output_all2,np.mean(test_output_all3),np.mean(test_output_all4),(test_output_all1+test_output_all3+test_output_all4)/3
