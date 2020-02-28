import time
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt

from net import *


# load saved net
def load_model(path):
    
    # unpack path to get model parameters
    _, conv_str, ch_str = path.split('_')
    conv = int(conv_str[5:6])
    ch = int(ch_str[3:5])
    
    # instantiate model
    model = SimpleNet(num_conv=conv, num_channels=ch, num_classes=10)
    
    # load model state dict
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)
    
    return model

# get losses of a NN on a set of images
def get_losses(model, dataloader, loss_fn):
    
    # move to GPU if possible
    if torch.cuda.is_available():
        model.cuda()
        
    model.eval()
        
    losses = []
    
    for i, (images, labels) in enumerate(dataloader):
    
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            
        # Predict classes using images from the test set
        with torch.no_grad():
            outputs = model(images)
        
        # Compute the loss based on the predictions and actual labels
        loss = loss_fn(outputs, labels)
        losses.append(float(loss))
        
    return losses

def representative_subset(losses, Y, num_classes, percentiles):

    subset = []
    
    # rank the images by loss
    ids = np.argsort(losses)
    
    # reorder the class labels to match the new ranking
    Y = Y[ids]

    # loop over classes
    for c in range(num_classes):
    
        # get indices for images of this class
        ids_class = ids[Y == c]
        
        # extract one image at each of the specified percentiles
        for p in percentiles:
            i = int(p * (len(ids_class) - 1))
            subset.append(ids_class[i])
            
    return subset
    
    
# Predicts a given image
def predict_image(image_tensor):
    print("Prediction in progress")

    if torch.cuda.is_available():
        image_tensor.cuda()

    # Turn the input into a Variable
    input = Variable(image_tensor)

    # Predict the class of the image
    output = model(input)
    print(output)
    index = output.data.numpy().argmax()

    return index
    