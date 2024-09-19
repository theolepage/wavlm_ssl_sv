#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/wujiyang/Face_Pytorch (Apache License)

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy, math
from utils import accuracy
import numpy as np

class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.3, scale=15, easy_margin=False, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        
        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses, nOut), requires_grad=True)
        # self.ce = nn.CrossEntropyLoss()
        self.ce = nn.CrossEntropyLoss(reduction='none') # return loss per sample
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        
        self.lgl_threshold = 1e6
        self.lc_threshold = 0.5

        print('Initialised AAMSoftmax margin %.3f scale %.3f'%(self.m,self.s))
        
    def _forward(self, x, label):
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        
        return output
    
    def _forward_softmax_sharpened(self, x, e=0.1):
        # regular softmax
        output = F.linear(x, self.weight)
        probas = F.softmax(output / e, dim=1)
        return probas
    
    def forward(self, x, x_clean, label=None, epoch=-1):
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        
        output  = self._forward(x, label)
        output_clean = self._forward_softmax_sharpened(x_clean)

        ce    = self.ce(output, label)
        
        # No LGL
        # prec1   = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        # return ce, prec1, None
        
        mask = (torch.log(ce) <= self.lgl_threshold).detach()
        
        if epoch <= 8:
            # LGL only
            nselect = torch.clamp(sum(mask), min=1).item()
            loss = torch.sum(ce * mask, dim=-1) / nselect
            prec1 = accuracy(output.detach(), label * mask.detach(), topk=(1,))[0]
            return loss, prec1, ce

        # LGL + LC
        
        label_LC = output_clean.argmax(dim=1)
        
        max_vals = torch.gather(output_clean, 1, label_LC.unsqueeze(1)).squeeze(1)
        mask_LC = (max_vals > self.lc_threshold).detach()
        
        ce_LC = self.ce(output, label_LC)
    
        mask_LGL_LC = ~mask & mask_LC
        loss = torch.mean(ce * mask + ce_LC * mask_LGL_LC, dim=-1)
        prec1 = accuracy(output.detach(), label * mask.detach() + label_LC * mask_LGL_LC.detach(), topk=(1,))[0]
        
        return loss, prec1, ce
    
    def get_pseudo_labels(self, x, label):
        output = self._forward_softmax_sharpened(x)
        return output.argmax(dim=1)

"""
    def forward(self, x, x_clean, label=None):

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        
        P_aam = self._forward(x, label)
        
        P_softmax = self._forward_softmax_sharpened(x)
        P_clean_softmax = self._forward_softmax_sharpened(x_clean)

        ce    = self.ce(P_aam, label)
        
        # No LGL
        # prec1   = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        # return ce, prec1, None
        
        mask = (torch.log(ce) <= self.lgl_threshold).detach()
        
        # LGL only
        # nselect = torch.clamp(sum(mask), min=1).item()
        # loss = torch.sum(ce * mask, dim=-1) / nselect
        # prec1 = accuracy(output.detach(), label * mask.detach(), topk=(1,))[0]
        # return loss, prec1, ce
        
        # LGL + LC
        label_LC = P_clean_softmax.argmax(dim=1)
        ce_LC = self.ce(P_softmax, label_LC)
    
        inverted_mask = ~mask
        loss = torch.mean(ce * mask + ce_LC * inverted_mask, dim=-1)
        prec1 = accuracy(P_softmax.detach(), label * mask.detach() + label_LC * inverted_mask.detach(), topk=(1,))[0]
        
        return loss, prec1, ce
"""