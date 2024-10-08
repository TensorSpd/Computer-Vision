## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # we are the feeding the images with pixels 224x224
        self.conv1 = nn.Conv2d(1, 32, 5)
        #our output will have pixel size of (W-K+2P)S+1 = 220 with 32 feature maps
        
        #maxpooling
        self.pool1 = nn.MaxPool2d(2,2)
        #this conv will produce the pixel size of (W-K)S+1 = 110 
        #output tensor shape (32,110,110)
        
        #adding batch normalization to avoid overfitting
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 36, 5)
        #(W-K+2p)/s+1 = 106, layer output --> (36,106,106)
        self.pool2 = nn.MaxPool2d(2,2)
        #(W-K)/s+1 = 53, layer ouput --> (36, 53, 53)
        self.bn2 = nn.BatchNorm2d(36)
        
        self.conv3 = nn.Conv2d(36, 48, 5)
        #layer output --> (48, 49,  49)
        self.pool3 = nn.MaxPool2d(2,2)
        #layer output --> (48, 24, 24)
        self.bn3 = nn.BatchNorm2d(48)
        
        self.conv4 = nn.Conv2d(48, 64, 3)
        #layer output --> (64, 22, 22)
        self.pool4 = nn.MaxPool2d(2, 2)
        #layer output --> (64, 11, 11)
        #self.conv_dropout1 = nn.Dropout(p=0.2)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 128, 3)
        #layer output --> (128, 9, 9)
        self.pool5 = nn.MaxPool2d(2,2)
        #layer output --> (128, 4, 4)
        #self.conv_dropout2 = nn.Dropout(p=0.2)
        self.bn5 = nn.BatchNorm2d(128)
        
        #Linear layer with dropout probability = 0.4
        
        self.fc1 = nn.Linear(128*4*4, 136)
        
        #self.linear_drop = nn.Dropout(p =0.4)
                  
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        
        ##flattening the output for linear layer
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
