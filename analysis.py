import time
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
    
def load_human_anns():
    '''
    Load human trial results in dict. format
    
    Output format:
    {
        <participant id> : {
            <image id> : {
                "label": <label>,
                "gt": <gt>,
                "time": <time>
            }
        }
    }
    '''
    
    anns = {}

    participants = [3, 4, 6, 9, 10, 11, 12, 13, 14, 15, 16]
    
    for p_id in participants:
    
        anns[p_id] = {}
    
        csv = pd.read_csv("results/%d_study_100ms.csv" % p_id)
        filenames = list(csv["stimFile"])
        labels = list(csv["rating.response"])
        times = list(csv["rating.rt"])
        
        for f, l, t in zip(filenames, labels, times):
        
            f = f.replace("stims/", '').replace(".png", '')
            img_id_str, gt_str = f.split('_')
            img_id = int(img_id_str)
            gt = int(gt_str)
            
            anns[p_id][img_id] = {
                "label": l,
                "gt": gt,
                "time": round(t, 3)
            }
    
    return anns

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
    
# get predictions of a NN on a set of images
def get_preds(model, dataloader):
    
    # move to GPU if possible
    if torch.cuda.is_available():
        model.cuda()
        
    model.eval()
        
    predictions = []
    
    for i, (images, labels) in enumerate(dataloader):
    
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            
        # Predict classes using images from the test set
        with torch.no_grad():
            outputs = model(images)
        
        # Take maximum entry as prediction
        pred = outputs.cpu().data.numpy().argmax()
        predictions.append(int(pred))
        
    return predictions
    
def load_NN_anns():
    '''
    Load NN test results in dict. format
    
    Output format:
    {
        <net name> : {
            <image id> : {
                "label": <label>,
                "gt": <gt>,
                "loss": <loss>
            }
        }
    }
    '''
    
    anns = {}
    
    convs = [1, 2, 3]
    chs = [16, 32, 64]
    net_names = ["SimpleNet_conv=%d_ch=%d" % (i, j) for i in convs for j in chs]
    
    # get human annotation ids from csv
    image_ids = np.loadtxt("human_trials.csv", dtype=int, delimiter=',')
    
    transform = transforms.Compose([
        # Must convert all images to tensors first to be processed.
        transforms.ToTensor(),
        # Normalize images to mean 0, variance 1 (improves training)
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # get dataloader
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           transform=transform, download=True)
    test_set_raw = CIFAR10(root="data", train=False, download=True)
    testlabels = np.array([y for (img, y) in test_set_raw])
    # Create a temp dataset loader just to load images, with batch size 1
    testloader = DataLoader(testset, batch_size=1,shuffle=False, num_workers=1)
    loss_fn = nn.CrossEntropyLoss()
    
    for name in net_names:
    
        anns[name] = {}
        
        # load network
        path = "models/" + name + "/49.pth"
        model = load_model(path)
        
        # get predictions and losses
        losses = get_losses(model, testloader, loss_fn)
        preds  = get_preds(model, testloader)
        
        # create entries
        for img_id in image_ids:
        
            label = preds[img_id]
            gt = testlabels[img_id]
            loss = losses[img_id]
            
            anns[name][img_id] = {
                "label": label,
                "gt": gt,
                "loss": loss
            }
    
    return anns
    
def feature_representation(model, img):
    # TODO: test this

    model = nn.Sequential(*list(model.children())[:-1])
    
    img_t = transforms.to_tensor(img)
    img_t = torch.unsqueeze(img_t, 0)
    img_t = img_t.to(Config.device)

    features = model(img_t)
    return features.cpu().data.numpy()[0]
    
def principal_components(A, k):
    # returns the first k singular vectors in the decomposition of A
    # each row in the output is a singular vector
    
    pca = PCA(n_components=k)
    pca.fit(A)
    
    return pca.components_
    
def confusion_matrix(labels, predictions, n_classes):
    # rows are true class, cols are predicted class
    # labels  is (n_images,)
    # predictions is (n_images, n_classes), with rows summing to 1
    confusion = np.zeros((n_classes, n_classes))
    counts = np.zeros((n_classes))
    
    for true, pred in zip(labels, predictions):
        
        confusion[true] += pred
        counts[true] += 1
        
    for c in range(n_classes):
        
        confusion[c] /= counts
        confusion[c, c] = 0
    
    return confusion

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
    