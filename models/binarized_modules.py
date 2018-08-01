import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function

import numpy as np


def Binarize(tensor, quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()

    ## SDPyle modified to enable [0,1] quantizing, which is easier for array indexing
    elif quant_mode == 'ge':
        return tensor.ge(0)
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)




class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()
        self.margin=1.0

    def hinge_loss(self,input,target):
            #import pdb; pdb.set_trace()
            output=self.margin-input.mul(target)
            output[output.le(0)]=0
            return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input,target)

class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction,self).__init__()
        self.margin=1.0

    def forward(self, input, target):
        output=self.margin-input.mul(target)
        output[output.le(0)]=0
        self.save_for_backward(input, target)
        loss=output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self,grad_output):
       input, target = self.saved_tensors
       output=self.margin-input.mul(target)
       output[output.le(0)]=0
       import pdb; pdb.set_trace()
       grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
       grad_output.mul_(output.ne(0).float())
       grad_output.div_(input.numel())
       return grad_output,grad_output


def Quantize(tensor,quant_mode='det',  params=None, numBits=8):
    tensor.clamp_(-2**(numBits-1),2**(numBits-1))
    if quant_mode=='det':
        tensor=tensor.mul(2**(numBits-1)).round().div(2**(numBits-1))
    else:
        tensor=tensor.mul(2**(numBits-1)).round().add(torch.rand(tensor.size()).add(-0.5)).div(2**(numBits-1))
        quant_fixed(tensor, params)
    return tensor

import torch.nn._functions as tnnf


class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

        ## SDPyle modified to maintain consistent positive and negative weights
        ## according to a normal distribution
        self.real_pos_weights = np.random.normal(1, 0.25, size=self.weight.data.shape)
        self.real_neg_weights = -1*np.random.normal(1, 0.25, size=self.weight.data.shape)

    ## SDPyle modified to include binarization_type for either ideal or with variations
    def forward(self, input):

        if input.size(1) != 784:
            input.data=Binarize(input.data)
            
        # keep a record of original weight for gradient calculation
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()

        ## SDPyle modified to binarize either ideal [-1,+1] or with variations according
        ## to normal distribution [mu_n+sigma_n, mu_p+sigma+p]

        # binarize the weight from original weight
        # if binarize_type == 'ideal':
        #     self.weight.data = Binarize(self.weight.org)
        # else:
        self.weight.data = torch.tensor(
            np.where(Binarize(self.weight.org, quant_mode='ge'), self.real_pos_weights, self.real_neg_weights),
            dtype=torch.float)


        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)

        ## SDPyle modified to maintain consistent positive and negative weights
        ## according to a normal distribution
        self.real_pos_weights = np.random.normal(1, 0.25, size=self.weight.data.shape)
        self.real_neg_weights = -1 * np.random.normal(1, 0.25, size=self.weight.data.shape)


    def forward(self, input):
        if input.size(1) != 3:
            input.data = Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()

        ## SDPyle modified to binarize either ideal [-1,+1] or with variations according
        ## to normal distribution [mu_n+sigma_n, mu_p+sigma+p]

        # binarize the weight from original weight
        # if binarize_type == 'ideal':
        #     self.weight.data = Binarize(self.weight.org)
        # else:
        self.weight.data = torch.tensor(
            np.where(Binarize(self.weight.org, quant_mode='ge'), self.real_pos_weights, self.real_neg_weights),
            dtype=torch.float)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
