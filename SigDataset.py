#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 12:01:30 2018

@author: vijay
"""


import os
import torchvision.datasets as dset
import numpy as np
import random
from PIL import Image
import torch
import PIL.ImageOps    
from torch.utils.data import Dataset



#Class to create a custom datset according to pytorch practices
class SigDataset(Dataset):
    
    def __init__(self,imageFolderDataset,dirname, transform=None,should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.directoryname = dirname
        self.should_invert = should_invert
     
        
        
    def random_folder(self, directoryname1):
        folder_name = random.choice(os.listdir(directoryname1))
        return os.path.join(directoryname1, folder_name)
    
    
    
    
        #__get_item is a skeleton provided by the pytorch Dataset we need to override 
    def __getitem__(self,index):
        
        
        folder = self.random_folder(self.directoryname)
        subfolderdataset = dset.ImageFolder(root = folder)
        

        img0_tuple = random.choice(subfolderdataset.imgs)
        
        
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            
            while True:
            #keep looping till the same class image is found
                img1_tuple = random.choice(subfolderdataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                #keep looping till a different class image is found
                
                img1_tuple = random.choice(subfolderdataset.imgs) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)      
    
