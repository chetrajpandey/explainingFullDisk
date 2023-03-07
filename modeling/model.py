import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms 
from torchvision.transforms import ToTensor
# from torch.utils.data import Dataset, DataLoader, ConcatDataset
# from torch.utils.data.sampler import Sampler, WeightedRandomSampler


NET_TYPES = {
             'alexnet': torchvision.models.alexnet             
}

class Custom_AlexNet(nn.Module):

    def __init__(self,
                 ipt_size=(512, 512), 
                 pretrained=True, 
                 net_type='alexnet', 
                 num_classes=2, train=True):
        super(Custom_AlexNet, self).__init__()
        
        #Mode Initialization
        self.tr = train
        #add one convolution layer at the beginning
        self.first_conv_layer = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.ReLU(inplace=True)
            )

        # load convolutional part of AlexNet
        assert net_type in NET_TYPES, "Unknown vgg_type '{}'".format(net_type)
        net_loader = NET_TYPES[net_type]
        net = net_loader(pretrained=pretrained)
        self.features = net.features

        # init fully connected part of AlexNet
        test_ipt = Variable(torch.zeros(1,3,ipt_size[0],ipt_size[1]))
        test_out = net.features(test_ipt)
        self.n_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self._init_classifier_weights()

    def forward(self, x):
        x = self.first_conv_layer(x)
        x = self.features(x)
        x = self.avgpool(x) 
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.tr:
            return F.log_softmax(x)
        else:
            return x

    def _init_classifier_weights(self):
        for m in self.first_conv_layer:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()