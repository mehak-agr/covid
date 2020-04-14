#import files from Nathan Densenet start
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from pydicom.contrib.pydicom_PIL import show_PL
import pathlib
from sklearn.metrics import roc_auc_score

#Pytorch packages
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func
#import from Nathan Densent ends
import csv
import os
import os.path
import tarfile
from urllib.parse import urlparse

import torch.utils.data as data
from PIL import Image

from medinet import util

# reproducability
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def find_classes(dir):
    targets = ['No Pneumonia','Pneumonia']
    # assign a label to each class
    classes = dict()
    index=0
    for x in targets:
        classes[x]=index
        index+=1
    return classes

class ChestXRay(data.Dataset):

    def __init__(self, csv, set, transform=None, target_transform=None): # image_list_file refers to the csv file with images names + labels

        self.csv = csv #link to csv directory
        self.set = set #link to type TRAIN/VAL
        self.transform = transform
        self.target_transform = target_transform
        self.path_images = ''
        print("info", self.csv, self.set)
        self.classes = find_classes(self.csv) #Basically name all objects in image, 'No Pneumonia and Pneumonia' to 0 and 1
        
        df = pd.read_csv(csv)
        df = df.drop_duplicates(subset=['path'])
        df = df[~df.path.str.contains('lateral')]
        self.image_names, self.labels = df['path'].tolist(), df['target'].tolist()

        print('[dataset] ChestXRay set=%s  number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.image_names)))


    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(os.path.join(self.path_images, image_name)).convert('RGB')
        #image = pydicom.dcmread(os.path.join(self.path_images,image_name))
        #make a [3,320,320] array to match with RGB input requirement
        label = self.labels[index]
        label = torch.from_numpy(np.array(label))

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is  not None:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.image_names)

    def get_number_classes(self):
        return len(self.classes)
