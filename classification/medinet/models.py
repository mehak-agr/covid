import torch.nn as nn
import torchvision.models as models

from medinet.pooling import WildcatPool2d, ClassWisePool, WeldonPool2d



class ResNetWSL(nn.Module):

    def __init__(self, model, num_classes, pooling=WildcatPool2d(), dense=False):
        super(ResNetWSL, self).__init__()

        self.dense = dense

        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4)

        # classification layer
        num_features = model.layer4[1].conv1.in_channels
        self.classifier = nn.Sequential(
            nn.Conv2d(num_features, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

        self.spatial_pooling = pooling

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        if not self.dense:
            x = self.spatial_pooling(x)
        return x

    def get_config_optim(self, lr, lrp):
        return [{'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.classifier.parameters()},
                {'params': self.spatial_pooling.parameters()}]


class DenseNetWSL(nn.Module):

    def __init__(self, model, num_classes, pooling=WildcatPool2d(), dense=False):
        super(DenseNetWSL, self).__init__()

        self.dense = dense

        self.features = model.features

        # classification layer
        num_features = model.classifier.in_features
        self.classifier = nn.Conv2d(num_features, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self.spatial_pooling = pooling

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        if not self.dense:
            x = self.spatial_pooling(x)
        return x

    def get_config_optim(self, lr, lrp):
        return [{'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.classifier.parameters()},
                {'params': self.spatial_pooling.parameters()}]


class VGGWSL(nn.Module):

    def __init__(self, model, num_classes, pooling=WildcatPool2d(), dense=False):
        super(VGGWSL, self).__init__()

        self.dense = dense

        self.features = model.features

        # classification layer
        num_features = model.features[-3]
        # print('model info', num_features)
        num_features = num_features.in_channels
        self.classifier = nn.Conv2d(num_features, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self.spatial_pooling = pooling

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        if not self.dense:
            x = self.spatial_pooling(x)
        return x

    def get_config_optim(self, lr, lrp):
        return [{'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.classifier.parameters()},
                {'params': self.spatial_pooling.parameters()}]

    
    
def resnet50_wildcat(num_classes, pretrained=True, kmax=1, kmin=None, alpha=1, num_maps=1):
    model = models.resnet50(pretrained)
    pooling = nn.Sequential()
    pooling.add_module('class_wise', ClassWisePool(num_maps))
    pooling.add_module('spatial', WildcatPool2d(kmax, kmin, alpha))
    return ResNetWSL(model, num_classes * num_maps, pooling=pooling)


def resnet101_wildcat(num_classes, pretrained=True, kmax=1, kmin=None, alpha=1, num_maps=1):
    model = models.resnet101(pretrained)
    pooling = nn.Sequential()
    pooling.add_module('class_wise', ClassWisePool(num_maps))
    pooling.add_module('spatial', WildcatPool2d(kmax, kmin, alpha))
    return ResNetWSL(model, num_classes * num_maps, pooling=pooling)

def densenet121_wildcat(num_classes, pretrained=True, kmax=1, kmin=None, alpha=1, num_maps=1):
    model = models.densenet121(pretrained)
    pooling = nn.Sequential()
    pooling.add_module('class_wise', ClassWisePool(num_maps))
    pooling.add_module('spatial', WildcatPool2d(kmax, kmin, alpha))
    return DenseNetWSL(model, num_classes * num_maps, pooling=pooling)

def densenet121_weldon(num_classes, pretrained=True, kmax=1, kmin=None):
    model = models.densenet121(pretrained)
    pooling = WeldonPool2d(kmax, kmin)
    return DenseNetWSL(model, num_classes, pooling=pooling)

def vgg_wildcat(num_classes, pretrained=True, kmax=1, kmin=None, alpha=1, num_maps=1):
    model = models.vgg19(pretrained)
    pooling = nn.Sequential()
    pooling.add_module('class_wise', ClassWisePool(num_maps))
    pooling.add_module('spatial', WildcatPool2d(kmax, kmin, alpha))
    return VGGWSL(model, num_classes * num_maps, pooling=pooling)

def vgg_weldon(num_classes, pretrained=True, kmax=1, kmin=None):
    model = models.vgg19(pretrained)
    pooling = WeldonPool2d(kmax, kmin)
    return VGGWSL(model, num_classes, pooling=pooling)


def resnet18_weldon(num_classes, pretrained=True, kmax=1, kmin=None):
    model = models.resnet18(pretrained)
    pooling = WeldonPool2d(kmax, kmin)
    return ResNetWSL(model, num_classes, pooling=pooling)


def resnet34_weldon(num_classes, pretrained=True, kmax=1, kmin=None):
    model = models.resnet34(pretrained)
    pooling = WeldonPool2d(kmax, kmin)
    return ResNetWSL(model, num_classes, pooling=pooling)


def resnet50_weldon(num_classes, pretrained=True, kmax=1, kmin=None):
    model = models.resnet50(pretrained)
    pooling = WeldonPool2d(kmax, kmin)
    return ResNetWSL(model, num_classes, pooling=pooling)


def resnet101_weldon(num_classes, pretrained=True, kmax=1, kmin=None):
    model = models.resnet101(pretrained)
    pooling = WeldonPool2d(kmax, kmin)
    return ResNetWSL(model, num_classes, pooling=pooling)


def resnet152_weldon(num_classes, pretrained=True, kmax=1, kmin=None):
    model = models.resnet152(pretrained)
    pooling = WeldonPool2d(kmax, kmin)
    return ResNetWSL(model, num_classes, pooling=pooling)