#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 14:32:52 2018

@author: kowshik
"""

import random

import torch
from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from   torchvision import datasets, transforms
import numpy as np

class AutoEncoder(nn.Module):
    
    def __init__(self, code_size,n,n_principal_components):
        super().__init__()
        self.code_size = code_size
        self.projection_size=code_size
        self.n_principal_components=n_principal_components
        self.n=n
        # Encoder specification
        self.enc_cnn_1 = nn.Conv2d(1, 10, kernel_size=5)
        self.enc_cnn_2 = nn.Conv2d(10, 20, kernel_size=5)
        self.enc_linear_1 = nn.Linear(4 * 4 * 20, 50)
        self.enc_linear_2 = nn.Linear(50, self.code_size)
        
        # Decoder specification
        self.dec_linear_1 = nn.Linear(self.projection_size, 160)
        self.dec_linear_2 = nn.Linear(160, IMAGE_SIZE)
        
    def forward(self, images):
        code = self.encode(images)
        subspaces,projections= self.get_subspaces_projection(code,n,n_principal_components)
        
        out = self.decode(projections)
        return out, code
    
    def encode(self, images):
        code = self.enc_cnn_1(images)
        code = F.selu(F.max_pool2d(code, 2))
        
        code = self.enc_cnn_2(code)
        code = F.selu(F.max_pool2d(code, 2))
        
        code = code.view([images.size(0), -1])
        code = F.selu(self.enc_linear_1(code))
        code = self.enc_linear_2(code)
       
        return code
    def get_subspaces_projection(self,code,n,n_principal_components):
        Subspaces=[]
        Projections=[]
       
        
        for i in range(code.shape[0]):

            code_indv= code[i].view(n,n)
            
            #code_indv=StandardScaler().fit_transform(code_indv)
    
            [U,S,V]=torch.svd(code_indv)
    
            Subspaces.append(U[:,:n_principal_components])
            projection=torch.matmul(U[:,:n_principal_components],torch.transpose(U[:,:n_principal_components],1,0))
            
            
            Projections.append(projection)
        Projections=torch.stack(Projections).view(-1,n*n)
      
        return Subspaces,Projections # projectins will be a tensor of batch_size *n^2

        
    def decode(self, Projections):
        
        out = F.selu(self.dec_linear_1(Projections))
        out = F.sigmoid(self.dec_linear_2(out))
     
        out = out.view([-1, 1, IMAGE_WIDTH, IMAGE_HEIGHT])
        return out
    
    
    

IMAGE_SIZE = 784
IMAGE_WIDTH = IMAGE_HEIGHT = 28

# Hyperparameters
code_size = 25# let it be a square 
n=5;#sqrt(code_size)
n_principal_components=2
num_epochs = 5
batch_size = 128
lr = 0.002
optimizer_cls = optim.Adam

# Load data
train_data = datasets.MNIST('mnist/', train=True , transform=transforms.ToTensor())
test_data  = datasets.MNIST('mnist/', train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=True)

# Instantiate model
autoencoder = AutoEncoder(code_size,n,n_principal_components)
loss_fn = nn.BCELoss()
optimizer = optimizer_cls(autoencoder.parameters(), lr=lr)

# Training loop
for epoch in range(num_epochs):
    print("Epoch %d" % epoch)
    
    for i, (images, _) in enumerate(train_loader):    # Ignore image labels
        out, code = autoencoder(Variable(images))
        
        optimizer.zero_grad()
        loss = loss_fn(out, images)
        loss.backward()
        optimizer.step()
        
    print("Loss = %.3f" % loss.data[0])

#%% Try reconstructing on test data
test_image = random.choice(test_data.test_data)
test_image =torch.tensor(test_image.view([1, 1, IMAGE_WIDTH, IMAGE_HEIGHT]),dtype=torch.float32)
test_reconst, _ = autoencoder(test_image)

torchvision.utils.save_image(test_image.data, 'orig1.png')
torchvision.utils.save_image(test_reconst.data, 'recons1t1.png')
