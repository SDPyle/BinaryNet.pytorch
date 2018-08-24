import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Function
from .binarized_modules import  BinarizeLinear,BinarizeConv2d, StochasticBinaryActivation



class Test_Cifar10(nn.Module):

    def __init__(self, num_classes=10):
        super(Test_Cifar10, self).__init__()
        self.infl_ratio=1;
        self.features = nn.Sequential(
            BinarizeConv2d(3, 128*self.infl_ratio, kernel_size=3, stride=1, padding=1, bias=True),
            StochasticBinaryActivation(),

            BinarizeConv2d(128*self.infl_ratio, 128*self.infl_ratio, kernel_size=3, padding=1, bias=True),
            StochasticBinaryActivation(),


            BinarizeConv2d(256*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=True),
            StochasticBinaryActivation(),


            BinarizeConv2d(256*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=True),
            StochasticBinaryActivation(),


            BinarizeConv2d(256*self.infl_ratio, 512*self.infl_ratio, kernel_size=3, padding=1, bias=True),
            StochasticBinaryActivation(),


            BinarizeConv2d(512*self.infl_ratio, 512, kernel_size=3, padding=1, bias=True),
            StochasticBinaryActivation()

        )
        self.classifier = nn.Sequential(
            BinarizeLinear(512 * 4 * 4, 1024, bias=True),
            StochasticBinaryActivation(),

            BinarizeLinear(1024, 1024, bias=True),
            StochasticBinaryActivation(),

            BinarizeLinear(1024, num_classes, bias=True),
            StochasticBinaryActivation()
        )

        self.regime = {
            0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 5e-3},
            40: {'lr': 1e-3},
            80: {'lr': 5e-4},
            100: {'lr': 1e-4},
            120: {'lr': 5e-5},
            140: {'lr': 1e-5}
        }

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.classifier(x)
        return x


def test_cifar10_binary(**kwargs):
    num_classes = getattr(kwargs,'num_classes', 10)
    return Test_Cifar10(num_classes)
