import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.autograd import Variable
def sample_gumbel(shape,eps=1e-20):
    U = torch.rand(shape,requires_grad=True)
    dist = - Variable(torch.log(- torch.log(U+eps) + eps))
    if torch.cuda.is_available():
        dist = dist.cuda()
    return dist
def gumbel_softmax_sample(logits,temp):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y/temp,dim=-1)
def gumbel_softmax(logits,temp,hard=False):
    y = gumbel_softmax_sample(logits,temp)
    if hard:
        ind = torch.argmax(y,dim=-1).unsqueeze(1)
        one_hot = torch.zeros(y.size()).float()
        if torch.cuda.is_available():
            one_hot = one_hot.cuda()
        one_hot.scatter_(1,ind,1)
        y = (one_hot - y).detach() + y
    return y

