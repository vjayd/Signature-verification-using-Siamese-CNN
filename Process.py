#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 12:31:46 2018

@author: vijay
"""
from Config import Config as Config
import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
import gc


class ContrastiveLoss(torch.nn.Module):
   
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive




criterion = ContrastiveLoss()





def show_plot(iteration,loss,i):
    if(i==1):
        plt.xlabel('iteration')
        plt.ylabel('loss')
    elif(i==2):
        plt.xlabel('threshold(d)')
        plt.ylabel('accuracy')
    else:
        plt.xlabel('threshold(d)')
        plt.ylabel('accuracy=(TPR+TNR/2)')
    
    plt.plot(iteration,loss)
    plt.show()




def train( train_dataloader, model):
    counter =[]
    loss_history = []
    iteration_number = 0

    optimizer = optim.Adam(model.parameters(), Config.learning_rate)
    for epoch in range(0,Config.train_epoch):
        for i, data in enumerate(train_dataloader,0):
            img0, img1 , label = data  
            img0, img1 , label = img0, img1 , label
            optimizer.zero_grad()
            #output1,output2 = model(Variable(img0.type(torch.FloatTensor),Variable(img1.type(torch.FloatTensor))))
            output1,output2 = model(img0,img1)
           
            loss_contrastive = criterion(output1,output2,label)                                                                                                                                     
            loss_contrastive.backward()
            optimizer.step()
            if i %50 == 0 :
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
    show_plot(counter,loss_history,1)



def custom_replace(tensor, on_great, on_less, i):
    # we create a copy of the original tensor, 
    # because of the way we are replacing them.
    res = tensor.clone()
    res[tensor>=i] = on_great
    res[tensor<i] = on_less
    return res



list_d = []
list_acc =[]
list_acc_tpn = []

def accuracy_statistics(result, label):
    
    max_value , indices = torch.max(result,0)
    
    for i in np.arange(0, int(3), 0.1):
        cloned_result = result.clone()
        predicted = custom_replace(cloned_result, on_great =1, on_less =0, i =i)
        total_correct = predicted.eq(label).sum()
        accuracy = (100 * total_correct)/Config.batch_size
        list_d.append(i)
        list_acc.append(accuracy)
        
        true_positive = (predicted[:,0]==0).sum()
        true_negative = (predicted[:,0]==1).sum()
        mod_similar = (label[:,0]==0).sum()
        mod_dissimilar =(label[:,0]==1).sum()
        TPR = true_positive/ mod_similar
        TNR = true_negative/ mod_dissimilar
        accuracyt =  (TPR + TNR)/2
        list_acc_tpn.append(accuracyt)
        
    #plot the graph   
    show_plot(list_d, list_acc,2)
    
    print("Accuracy Attained ",max(list_acc).numpy(),"%")
    





def test(test_dataloader, model):
    
    for i , data in enumerate(test_dataloader, 0):
        print(i)
        img0, img1, label = data
        output1, output2 = model(img0, img1)
        e_distance = F.pairwise_distance(output1, output2).reshape(180,1)
        accuracy_statistics(result = e_distance, label = label)
  
    