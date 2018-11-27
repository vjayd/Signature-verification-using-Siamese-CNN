#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 20:01:10 2018

@author: vijay
"""

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from Config import Config as Config
from SigDataset import SigDataset as sg
import Process as p



class SignCNN(nn.Module):
    def __init__(self):
        #DId not added Dropout and Local response layer..
        #The input is resized to 512 * 512  using bilinear interpolation 
        super(SignCNN, self).__init__()
        
        self.conv2 = nn.Conv2d(1, 16 , 3, stride = 2, padding = 2)#150
        #76,76,128
        self.batch2 = nn.BatchNorm2d(16)
        
        self.conv3 = nn.Conv2d(16, 32, 3, stride = 2, padding = 1)
        #38,38,32
        self.batch3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 3, stride = 2, padding = 1)
        #19,19,64
        self.conv5 = nn.Conv2d(64, 12, 3, stride = 1, padding = 1)
        #19,19,12
        self.batch4 = nn.BatchNorm2d(12)
        #self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv6 = nn.Conv2d(12, 8, 3, stride = 2, padding = 1)
        #10,10,8
        
        self.fc1 = nn.Linear(800,500)
        self.fc2 = nn.Linear(500,300)
        self.fc3 = nn.Linear(300, 216)
        self.fc4 = nn.Linear(216, 188)
        self.fc5 = nn.Linear(188, 70)
        
    def forward_once(self, x):
        #x = F.relu(self.pool(self.batch1(self.conv1(x))))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.batch3(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.batch4(self.conv5(x)))
        x = F.relu(self.conv6(x))
        
        
        x = x.view(-1, 8 *10 *10)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return x
    

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
    
    
######################################################### TRAIN DATASET ############################################3
folder_dataset = dset.ImageFolder(root = Config.training_dir)
siamese_sigdataset = sg(folder_dataset, Config.training_dir, transform = transforms.Compose([transforms.Resize((150,150)),
                                                        transforms.ToTensor()]), should_invert = False)
train_dataloader = DataLoader(siamese_sigdataset, shuffle = True,
                              num_workers = 8, batch_size = Config.batch_size)

model = SignCNN()


########################################################## TEST CODE #############################################


folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
siamese_test_dataset = sg(folder_dataset_test, Config.testing_dir, transform=transforms.Compose([transforms.Resize((150,150)),
                                                                     transforms.ToTensor()]),should_invert=False)
test_dataloader = DataLoader(siamese_test_dataset,num_workers=8,batch_size=Config.batch_size,shuffle=True)

####################################################### PROCESS THE DATA FOR TRAINING AND TESTING #########################



def process():
    p.train(train_dataloader, model)
    p.test(test_dataloader, model)
    
    
    
    
    
    
process()    
