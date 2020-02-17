import numpy as np
import torch
import torch.nn as nn

# Basic Layer unit consisting of convolution, batch norm, and RELU
class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output


# Neural Net with num_layers number of units
# Each unit has num_channels number of channels
class SimpleNet(nn.Module):
    def __init__(self,num_layers=1,num_channels=32,num_classes=10):
        super(SimpleNet,self).__init__()
        self.num_channels=num_channels

        layers = []
        for i in range(num_layers):

            # First layer is should have input channels be equal to dim of dataset
            if i == 0:
                layers.append(Unit(in_channels=3, out_channels=num_channels))
            else:
                layers.append(Unit(in_channels=num_channels, out_channels=num_channels))


        # Need to apply pooling equal to the dimension of num channels (to reduce dimensionality)
        # nn.AvgPool2d might work better here
        layers.append(nn.MaxPool2d(kernel_size=num_channels))

        # Append all blocks sequentially in neural net
        self.net = nn.Sequential(*layers)

        # Last linear layer to go from output features to number of classes
        self.fc = nn.Linear(in_features=num_channels, out_features=num_classes)


    def forward(self, input):

        output = self.net(input)
        output = output.view(-1,self.num_channels)
        output = self.fc(output)
        return output
