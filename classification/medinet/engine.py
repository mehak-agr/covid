import os
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
from tqdm import tqdm
import random
# from ignite.handlers import EarlyStopping

from medinet.util import AveragePrecisionMeter, Warp
from sklearn.metrics import roc_auc_score

# reproducability
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max
        
        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1]*len(self.keys)
        for label in self.dataset:
            print("length of {}, {}".format(label, len(self.dataset[label])))
            
    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1]*len(self.keys)
    
    def _get_label(self, dataset, idx, labels = None):
        if dataset.labels is not None:
            return dataset.labels[idx]
        else:
            # Trying guessing
            dataset_type = type(dataset)
            if is_torchvision_installed and dataset_type is torchvision.datasets.MNIST:
                return dataset.train_labels[idx].item()
            elif is_torchvision_installed and dataset_type is torchvision.datasets.ImageFolder:
                return dataset.imgs[idx][1]
            else:
                raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        return self.balanced_max*len(self.keys)


class Engine(object):
    def __init__(self, state={}):
        self.state = state
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()

        if self._state('image_size') is None:
            self.state['image_size'] = 224

        if self._state('batch_size') is None:
            self.state['batch_size'] = 16

        if self._state('workers') is None:
            self.state['workers'] = 4

        if self._state('multi_gpu') is None:
            self.state['multi_gpu'] = False

        if self._state('device_ids') is None:
            self.state['device_ids'] = [0, 1]

        if self._state('evaluate') is None:
            self.state['evaluate'] = False

        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 50

        if self._state('epoch_step') is None:
            self.state['epoch_step'] = []
        
        if self._state('balanced') is None:
            self.state['balanced'] = 1
            
        if self._state('losstype') is None:
            self.check['losstype'] = 0
            
        if self._state('use_pb') is None:
            self.state['use_pb'] = True
            
        if self._state('print_freq') is None:
            self.state['print_freq'] = 50

        # meters
        self.state['meter_loss'] = tnt.meter.AverageValueMeter()
        
        # time measure
        self.state['batch_time'] = tnt.meter.AverageValueMeter()
        self.state['data_time'] = tnt.meter.AverageValueMeter()

        self.state['break'] = False
        
    def _state(self, name):
        if name in self.state:
            return self.state[name]
        
    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        loss = self.state['meter_loss'].value()[0]
        if display:
            if training:
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}'.format(self.state['epoch'], loss=loss))
            else:
                print('Test: \t Loss {loss:.4f}'.format(loss=loss))
        return loss
    
    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        pass

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        # record loss
        #self.state['loss_batch'] = self.state['loss'].data[0]
        self.state['loss_batch'] = self.state['loss'].data
        self.state['meter_loss'].add(self.state['loss_batch'].cpu())

        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                   batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):

        input_var = torch.autograd.Variable(self.state['input'])
        target_var = torch.autograd.Variable(self.state['target'])

        # compute output
        if not training:
            # with torch.no_grad():
            with torch.set_grad_enabled(False):
                self.state['output'] = model(input_var)
                self.state['loss'] = criterion(self.state['output'], target_var)

        if training:
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                self.state['output'] = model(input_var)
                self.state['loss'] = criterion(self.state['output'], target_var)
                self.state['loss'].backward()
                optimizer.step()

    def init_learning(self, model, criterion):

        if self._state('train_transform') is None:
            try:
                normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
                print("using model normalize [WILDCAT]")
            except:
                normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                print("using generic normlize [REGULAR]")

            self.state['train_transform'] = transforms.Compose([
                Warp(self.state['image_size']),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        if self._state('val_transform') is None:
            try:
                normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
                print("using model normalize [WILDCAT]")
            except:
                normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                print("using generic normlize [REGULAR]")

            self.state['val_transform'] = transforms.Compose([
                Warp(self.state['image_size']),
                transforms.ToTensor(),
                normalize,
            ])

        self.state['best_score'] = 0

    def learning(self, model, criterion, train_dataset, val_dataset, optimizer=None):

        self.init_learning(model, criterion)

        # define train and val transform
        train_dataset.transform = self.state['train_transform']
        train_dataset.target_transform = self._state('train_target_transform')
        val_dataset.transform = self.state['val_transform']
        val_dataset.target_transform = self._state('val_target_transform')

        # data loading code
        
        if self._state('balanced') == 0:
            print('Performing balanced weighted sampling')
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.state['batch_size'], num_workers=self.state['workers'], drop_last=True)
        else:  
            train_loader = torch.utils.data.DataLoader(
                train_dataset,sampler=BalancedBatchSampler(train_dataset), batch_size=self.state['batch_size'], num_workers=self.state['workers'], drop_last=True)
        
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,batch_size=self.state['batch_size'], shuffle=False, num_workers=self.state['workers'], drop_last=True)

        # optionally resume from a checkpoint
        if self._state('resume') is not None:
            if os.path.isfile(self.state['resume']):
                print("=> loading checkpoint '{}'".format(self.state['resume']))
                checkpoint = torch.load(self.state['resume'])
                self.state['start_epoch'] = checkpoint['epoch']
                self.state['best_score'] = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.state['evaluate'], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.state['resume']))


        if self.state['use_gpu']:
            train_loader.pin_memory = True
            val_loader.pin_memory = True
            cudnn.benchmark = True

            if self.state['multi_gpu']:
                model = torch.nn.DataParallel(model, device_ids=self.state['device_ids']).cuda()
            else:
                model = torch.nn.DataParallel(model).cuda()

            criterion = criterion.cuda()

        if self.state['evaluate']:
            self.validate(val_loader, model, criterion)
            return

        # callbacks = [EarlyStopping(monitor='best_score', patience=5)]
        starting_epoch = 0
        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            self.state['epoch'] = epoch
            # self.adjust_learning_rate(optimizer)

            # train for one epoch
            self.train(train_loader, model, criterion, optimizer, epoch, val_loader)

            # evaluate on validation set
            auc = self.validate(val_loader, model, criterion)
            is_best = auc > self.state['best_score']
            # remember best auc and save checkpoint
            self.state['best_score'] = max(auc, self.state['best_score'])

            if is_best:
                print("updating model!")
                starting_epoch = epoch
            
            self.save_checkpoint({
                'epoch': epoch + 1,
                'arch': self._state('arch'),
                'state_dict': model.module.state_dict() if self.state['use_gpu'] else model.state_dict(),
                'best_score': self.state['best_score'],
            }, is_best)

            print(' *** best_auc={best:.3f}'.format(best=self.state['best_score']))
            if epoch - starting_epoch >= 5:
                print("Breaking")
                print('Stopped at '+str(epoch)+' epoch')
                break


    def train(self, data_loader, model, criterion, optimizer, epoch, val_loader):

        # switch to train mode
        model.train()

        self.on_start_epoch(True, model, criterion, data_loader, optimizer)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Training')

        end = time.time()
        start = 0
        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            
            if self._state('losstype') == 0:
                self.state['target'] = target.type(torch.FloatTensor).unsqueeze(dim=1)
            else:
                self.state['target'] = target.unsqueeze(dim=1)
        

            self.on_start_batch(True, model, criterion, data_loader, optimizer)

            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda(async=True)

            self.on_forward(True, model, criterion, data_loader, optimizer)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(True, model, criterion, data_loader, optimizer)
            
        self.on_end_epoch(True, model, criterion, data_loader, optimizer)

    def validate(self, data_loader, model, criterion, is_training=False):
        loss = self.state['meter_loss'].value()[0]
        # switch to evaluate mode
        model.eval()

        self.on_start_epoch(False, model, criterion, data_loader)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Test')

        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            
            if self._state('losstype') == 0:
                self.state['target'] = target.type(torch.FloatTensor).unsqueeze(dim=1)
            else:
                self.state['target'] = target.unsqueeze(dim=1)
            
            self.on_start_batch(False, model, criterion, data_loader)

            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda(async=True)

            self.on_forward(False, model, criterion, data_loader)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(False, model, criterion, data_loader)
        
        return self.on_end_epoch(False, model, criterion, data_loader)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if self._state('save_model_path') is not None:
            filename_ = filename
            filename = os.path.join(self.state['save_model_path'], filename_)
            if not os.path.exists(self.state['save_model_path']):
                os.makedirs(self.state['save_model_path'])
        print('save model {filename}'.format(filename=filename))
        torch.save(state, filename)
        if is_best:
            print("saving best model!")
            filename_best = 'model_best.pth.tar'
            if self._state('save_model_path') is not None:
                filename_best = os.path.join(self.state['save_model_path'], filename_best)
            shutil.copyfile(filename, filename_best)
            if self._state('save_model_path') is not None:
                if self._state('filename_previous_best') is not None:
                    os.remove(self._state('filename_previous_best'))
                filename_best = os.path.join(self.state['save_model_path'], 'model_best_{score:.4f}.pth.tar'.format(score=state['best_score']))
                shutil.copyfile(filename, filename_best)
                self.state['filename_previous_best'] = filename_best

    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        # lr = args.lr * (0.1 ** (epoch // 30))
        if self.state['epoch'] is not 0 and self.state['epoch'] in self.state['epoch_step']:
            print('update learning rate')
            for param_group in optimizer.state_dict()['param_groups']:
                param_group['lr'] = param_group['lr'] * 0.1
                print(param_group['lr'])


class MulticlassEngine(Engine):
    def __init__(self, state):
        
        Engine.__init__(self, state)
        self.state['classacc'] = tnt.meter.ClassErrorMeter(accuracy=True)
        self.state['auc'] = tnt.meter.AUCMeter()

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        
        self.state['classacc'].reset()
        self.state['auc'].reset()
        Engine.on_start_epoch(self, training, model, criterion, data_loader, optimizer)
        

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        
        top1 = self.state['classacc'].value()[0]
        auc  = self.state['auc'].value()[0]
        loss = self.state['meter_loss'].value()[0]
        
        if display:
            if training:
                print('Epoch: [{0}] \t Loss: {loss:.4f} \t Prec: {top1:.3f} \t AUC: {auc:.3f}'.format(self.state['epoch'], loss=loss, top1=top1, auc=auc))
            else:
                print('Test: \t Loss {loss:.4f} \t Prec: {top1:.3f} \t AUC: {auc:.3f}'.format(loss=loss, top1=top1, auc=auc))

        return auc

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        Engine.on_end_batch(self, training, model, criterion, data_loader, optimizer, display=False)

        # measure accuracy
        probs = torch.cat((1-self.state['output'].data, self.state['output'].data),1)
        self.state['classacc'].add(probs, self.state['target'])
        # measure auc
        self.state['auc'].add(self.state['output'].data, self.state['target'])
        
        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            top1 = self.state['classacc'].value()[0]
            auc  = self.state['auc'].value()[0]
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{0}][{1}/{2}] \t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f}) \t'
                      'Data {data_time_current:.3f} ({data_time:.3f}) \t'
                      'Loss {loss_current:.4f} ({loss:.4f}) \t'
                      'Prec {top1:.3f} \t'
                      'AUC {auc:.3f}'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'], batch_time=batch_time, data_time_current=self.state['data_time_batch'], data_time=data_time, loss_current=self.state['loss_batch'], loss=loss,top1=top1, auc=auc))
            else:
                print('Test: [{0}/{1}] \t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f}) \t'
                      'Data {data_time_current:.3f} ({data_time:.3f}) \t'
                      'Loss {loss_current:.4f} ({loss:.4f}) \t'
                      'Prec {top1:.3f} \t'
                      'AUC {auc:.3f}'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss, top1=top1, auc=auc))