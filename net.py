import numpy as np
import torch
import torch.nn as nn

# https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
# Some good implementation examples

# Convolutional unit consisting of convolution, batch norm, ReLU
# Input : (b, h, w, c_in) -> Output: (b, h, w, c_out)
class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels, stride=1, padding=1)
        self.bn   = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        
        output = self.conv(input)
        output = self.relu(output)
        output = self.bn(output)

        return output
    
# https://discuss.pytorch.org/t/flatten-layer-of-pytorch-build-by-sequential-container/5983/2
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
# Dense unit consisting of flatten, relu, linear
# Input : (b, d1, d2, d3, ...) -> Output : (b, out)
# -- in_features = d1 * d2 * d3 * ...
class DenseBlock(nn.Module):
    
    def __init__(self, in_features, out_features):
        super(DenseBlock, self).__init__()
        
        self.flat = Flatten()
        self.relu = nn.ReLU()
        self.lin  = nn.Linear(in_features=in_features, out_features=out_features)
        
    def forward(self, input):
        
        output = self.flat(input)
        output = self.relu(output)
        output = self.lin(output)
        
        return output

# Neural Net with num_conv convolutional units and num_lin linear units
class SimpleNet(nn.Module):
    
    def __init__(self, num_conv=2, num_channels=32, num_classes=10):
        
        super(SimpleNet, self).__init__()
        self.num_channels = num_channels

        layers = []
        for i in range(3):
            for j in range(num_conv):

                # First layer is should have input channels be equal to dim of dataset
                if i == 0 and j == 0:
                    layers.append(ConvBlock(in_channels=3, out_channels=num_channels))
                else:
                    layers.append(ConvBlock(in_channels=num_channels, out_channels=num_channels))
                    
            layers.append(nn.MaxPool2d(3))
        
        # linear layer to go from convolutional blocks to linear blocks     
        layers.append(DenseBlock(in_features=num_channels, out_features=num_classes))
        
        # Append all blocks sequentially in neural net
        self.net = nn.Sequential(*layers)
        
    def forward(self, input):

        return self.net(input)
