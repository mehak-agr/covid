#import files from Nathan Densenet start
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from pydicom.contrib.pydicom_PIL import show_PL
import pathlib
from PIL import Image
from sklearn.metrics import roc_auc_score

#Pytorch packages
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
#import from Nathan Densenet ends

import argparse

import torch
import torch.nn as nn

from medinet.engine import MulticlassEngine
from medinet.chestxray import ChestXRay

from medinet.models import resnet101_wildcat
from medinet.models import densenet121_wildcat
from medinet.models import vgg_wildcat

from medinet.models import resnet101_weldon
from medinet.models import densenet121_weldon
from medinet.models import vgg_weldon

from medinet.util import AveragePrecisionMeter, Warp
from sklearn.metrics import roc_auc_score
import glob

#reproducability
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='BASELINE/WILDCAT/WELDON Training')
parser.add_argument('--data',
                    help='path to dataset')
parser.add_argument('--model-dir', default='./models/', type=str, metavar='MODELPATH',
                    help='path to model directory (default: none)')
parser.add_argument('--image-size', '-i', default=320, type=int,
                    metavar='N', help='image size (default: 320)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 0)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--k', default=0, type=float,
                    metavar='N', help='number of regions (default: 0)')
parser.add_argument('--alpha', default=0, type=float,
                    metavar='N', help='weight for the min regions (default: 0)')
parser.add_argument('--maps', default=0, type=int,
                    metavar='N', help='number of maps per class (default: 0)')
parser.add_argument('--adam', default=1, type=int,
                    metavar='A', help='Use Adam')
parser.add_argument('--arch', default=0, type=int,
                    metavar='w', help='Use Baseline/Wildcat/Weldon')
parser.add_argument('--variant', default=0, type=int,
                    metavar='w', help='Use Densenet/Resnet/VGG')
parser.add_argument('--train_csv', default='train.csv', type=str,
                    metavar='r', help='Give train csv')
parser.add_argument('--val_csv', default='val.csv', type=str,
                    help='Give validation csv')
parser.add_argument('--sigmoid', default='1', type=int,
                    help='Specify if you need sigmoid activation in baseline')
parser.add_argument('--balanced', default='1', type=int,
                    help='Specify if you need balanced sampling or not (Default:BalancedSampling)')
parser.add_argument('--loss', default='0', type=int,
                    help='Specify which criterion you need (default:BCELoss)')

                    
class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.vgg19 = torchvision.models.vgg19(pretrained=True)
        num_ftrs = self.vgg19.classifier[0].in_features
        self.vgg19.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid())
    
    def forward(self, x):
        x = self.vgg19(x)
        return x

class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid())
    
    def forward(self, x):
        x = self.densenet121(x)
        return x

class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101, self).__init__()
        self.resnet101 = torchvision.models.resnet101(pretrained=True)
        num_ftrs = self.resnet101.fc.in_features
        self.resnet101.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid())
    
    def forward(self, x):
        x = self.resnet101(x)
        return x

def main_chestxray():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    adam_str = 'adam_' if args.adam else ''
    
    train_dataset = ChestXRay(args.data + args.train_csv, 'TRAIN') #COVID-Add link to proper training dataset here 
    val_dataset = ChestXRay(args.data + args.val_csv, 'VAL') #COVID-Add link to proper validation dataset here
    
    
    data_str = ''
    if '_cur' in args.train_csv:
        data_str = data_str + 'cur_'
    if 'chex' in args.train_csv:
        data_str = data_str + 'chex_'
    if 'mimic' in args.train_csv:
        data_str = data_str + 'mimic_'
    if 'rsna' in args.train_csv:
        data_str = data_str + 'rsna_'
    if 'all' in args.train_csv:
        data_str = data_str + 'chex_mimic_rsna_'
        
    if args.resume == '':
        check_str = ''
    else:
        check_str = 'from_'
        if '_cur' in args.resume:
            check_str = check_str + 'cur_'
        if 'chex' in args.resume:
            check_str = check_str + 'chex_'
        if 'mimic' in args.resume:
            check_str = check_str + 'mimic_'
        if 'rsna' in args.resume:
            check_str = check_str + 'rsna_'
        check_str = check_str + 'to_'
        

    num_classes = 2

    # load model
    if args.arch == 2:
        model_str = 'weldon'
        para_str = 'k' + str(args.k) + '_'
        if args.variant == 2:
            model = vgg_weldon(num_classes, pretrained=True, kmax=args.k)
            model_str += '_vgg'
        if args.variant == 1:
            model = resnet101_weldon(num_classes, pretrained=True, kmax=args.k)
            model_str += '_resnet'
        if args.variant == 0:
            model = densenet121_weldon(num_classes, pretrained=True, kmax=args.k)
            model_str += '_densenet'
        
    if args.arch == 1:
        model_str = 'wildcat'
        para_str = 'k' + str(args.k) + '_maps' + str(args.maps) + '_alpha' + str(args.alpha) + '_'
        if args.variant == 2:
            model = vgg_wildcat(num_classes, pretrained=True, kmax=args.k, alpha=args.alpha, num_maps=args.maps)
            model_str += '_vgg'
        if args.variant == 1:
            model = resnet101_wildcat(num_classes, pretrained=True, kmax=args.k, alpha=args.alpha, num_maps=args.maps)
            model_str += '_resnet'
        if args.variant == 0:
            model = densenet121_wildcat(num_classes, pretrained=True, kmax=args.k, alpha=args.alpha, num_maps=args.maps)
            model_str += '_densenet'
            
    if args.arch == 0:
        model_str = 'baseline'
        para_str = ''
        if args.variant == 2:
            model = VGG19().cuda()
            model_str += '_vgg'
        if args.variant == 1:
            model = ResNet101().cuda()
            model_str += '_resnet'
        if args.variant == 0:
            model = DenseNet121().cuda()
            model_str += '_densenet'
            
    if args.balanced == 0:
        model_str = model_str + '_unbalanced'
    else:
        model_str = model_str + '_balanced'
        
    mod_name = 'lr{}_lrp{}_{}epochs{}_{}{}{}{}'.format(args.lr, args.lrp, adam_str, args.epochs, para_str, check_str, data_str, model_str)
    print(mod_name)

    # define loss function (criterion)
    if args.loss == 0:
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # define optimizer
    if args.adam == 0:
        print("ITS SGD")
        if args.wild == 1:
            optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    #TODO add functionality for for ADAM optimizer
    else:
        print("ITS ADAM")
        if args.arch != 0:
            print("using config optim")
            optimizer = torch.optim.Adam(model.get_config_optim(args.lr, args.lrp),
                                    lr=args.lr,betas=(0.9,0.999))
        else:
            print("using param")
            optimizer = torch.optim.Adam(model.parameters(),
                                    lr=args.lr,betas=(0.9,0.999))

    
    state = {'workers': args.workers, 
             'batch_size': args.batch_size, 
             'image_size': args.image_size, 
             'max_epochs': args.epochs,
             'evaluate': args.evaluate, 
             'resume': args.resume, 
             'balanced':args.balanced, 
             'losstype':args.loss, 
             'print_freq':args.print_freq}
    state['difficult_examples'] = False
    state['save_model_path'] = args.model_dir + mod_name

    engine = MulticlassEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)

if __name__ == '__main__':
    main_chestxray()