import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import math
from .module import ConvLayer,LinearLayer,ResizeLayerUp,ResizeLayerDown,FlatenLayer,SingleEncoder,SingleDecoder

from .module import Encoder,Decoder
from .module import Encoder2,Decoder2
from .module import Encoder4,Decoder4
def sample_gumbel(shape,eps=1e-20):
    U = torch.rand(shape,requires_grad=True)
    dist = - Variable(torch.log(- torch.log(U+eps) + eps))
    return dist.cuda()
def gumbel_softmax_sample(logits,temp):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y/temp,dim=-1)
def gumbel_softmax(logits,temp,hard=False):
    y = gumbel_softmax_sample(logits,temp)
    if hard:
        ind = torch.argmax(y,dim=-1).unsqueeze(1)
        one_hot = torch.zeros(y.size()).type(torch.cuda.FloatTensor)
        one_hot.scatter_(1,ind,1)
        y = (one_hot - y).detach() + y
    return y
class Model(nn.Module):
    def __init__(self,c_in=80,hid=64,k=128,tau=1.0,seg_len=20):
        super().__init__()
        self.enc = Encoder(c_in,hid=hid)
        self.dec = Decoder(c_in = hid, c_out = c_in )
        self.k = k
        self.lin1 = nn.Linear(hid,k)
        self.lin2 = nn.Linear(k,hid)
        self.softmax = nn.Softmax(dim=-1)
        self.tau = tau
        self.hid = hid
    def forward(self,x):
        lat = self.enc(x).contiguous()
        B,C,T = lat.size()
        flat_lat = lat.view(B*T,C)
        logits_y = self.lin1(flat_lat)
        q_y = self.softmax(logits_y)
        log_q_y = torch.log(q_y + 1e-20)

        y = gumbel_softmax(logits_y, self.tau,hard=True)
        flat_new_lat = self.lin2(y)
        new_lat = flat_new_lat.view(B,self.hid,T)
        out = self.dec(new_lat)
        kl = torch.mean( torch.sum(q_y * (log_q_y - torch.log( torch.tensor(1.0/self.k).cuda() ) ) , dim=1 ) )
        return out,kl

    def encode(self,x):
        lat = self.enc(x).contiguous()
        B,C,T = lat.size()
        flat_lat = lat.view(B*T,C)
        logits_y = self.lin1(flat_lat)
        q_y = self.softmax(logits_y)
        log_q_y = torch.log(q_y + 1e-20)

        y = gumbel_softmax(logits_y, self.tau,hard=True)
        flat_new_lat = self.lin2(y)
        new_lat = flat_new_lat.view(B,self.hid,T)
        return new_lat

class SingleModel(nn.Module):
    def __init__(self,c_in=80,factor=3,hid=64,seg_len=30,k = 128,tau = 1.0):
        super().__init__()
        self.enc = SingleEncoder(c_in,hid=hid,seg_len=seg_len)
        self.k = k
        self.lin1 = nn.Linear(hid,k)
        self.lin2 = nn.Linear(k,hid)
        self.softmax = nn.Softmax(dim=-1)
        self.dec = SingleDecoder(c_in = hid, c_out = c_in, seg_len=seg_len)
        self.tau = tau
        
    def forward(self,x):
        lat = self.enc(x)
        lat = lat.squeeze(2)
        logits_y = self.lin1(lat)
        q_y = self.softmax(logits_y)
        log_q_y = torch.log(q_y + 1e-20)
        
        #gumbel softmax
        y = gumbel_softmax(logits_y,self.tau,hard=True)
        y = self.lin2(y)
        y = y.unsqueeze(2)
        out = self.dec(y)
        
        kl = torch.mean( torch.sum(q_y * (log_q_y - torch.log( torch.tensor(1.0/self.k).cuda() ) ) , dim=1 ) )
        return out, kl

    def encode(self,x):
        lat = self.enc(x)
        return lat
