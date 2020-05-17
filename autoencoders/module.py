import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import math


class ConvLayer(nn.Module):
    def __init__(self,c_in,c_out,ks=3,pad=1,stride=1,dp=0.1,res = False):
        super().__init__()
        self.res = res
        self.layer = nn.Conv1d(c_in,c_out,kernel_size=ks,padding=pad,stride=stride)
        self.act = nn.ReLU()
        #self.norm = nn.BatchNorm1d(c_out)
        #self.drop = nn.Dropout(p=dp)
    def forward(self,x):
        out = self.layer(x)
        out = self.act(out)
        if self.res:
            out += x
        #out = self.norm(out)
        #out = self.drop(out)

        return out
class TransConvLayer(nn.Module):
    def __init__(self,c_in,c_out,ks=2,pad=0,stride=1):
        super().__init__()
        self.layer = nn.ConvTranspose1d(c_in,c_out,kernel_size=ks,padding=pad,stride=stride)
        self.act = nn.ReLU()
        self.norm = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(p=0.5)
    def forward(self,x):
        out = self.layer(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.drop(out)
        return out
class LinearLayer(nn.Module):
    def __init__(self,c_in,c_out):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.layer = nn.Linear(c_in,c_out)
        self.norm = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
    def forward(self,x):
        B,F,T = x.size()
        x = x.view(B*T,F)
        out = self.layer(x)
        #out = self.norm(out)
        out = out.view(B,self.c_out,T)
        out = self.act(out)
        #out = self.drop(out)
        return out
class ResizeLayerDown(nn.Module):
    # squeeze T dimension
    def __init__(self,factor=2):
        super().__init__()
        self.factor = factor
    def forward(self,x):
        B,C,T = x.size()
        new_C = C * self.factor
        new_T = T // self.factor
        x = x.view(B,new_C, new_T)
        return x
class ResizeLayerUp(nn.Module):
    # expand T dimension
    def __init__(self,factor=2):
        super().__init__()
        self.factor = factor
    def forward(self,x):
        B,C,T = x.size()
        new_C = C // self.factor
        new_T = T * self.factor
        x = x.view(B,new_C, new_T)
        return x

class FlatenLayer(nn.Module):
    #flaten input into one vector
    def __init__(self):
        super().__init__()
    def forward(self,x):
        if len(x.size()) == 3:
            B,F,T = x.size()
            x = x.reshape(B,F*T,1)
        return x
class SingleEncoder(nn.Module):
    def __init__(self,c_in=80,hid=64,factor=3,seg_len=30):
        super().__init__()
        self.factor = factor
        self.net = nn.Sequential()
        self.net.add_module('Conv1',ConvLayer(c_in=c_in,c_out=32))
        self.net.add_module('Conv2',ConvLayer(c_in=32,c_out=64))
        self.net.add_module('Conv3',ConvLayer(c_in=64,c_out=16))
        self.net.add_module('FlatenLayer',FlatenLayer())
        self.out_layer = nn.Linear(16*seg_len,hid)
    def forward(self,x):
        out = self.net(x)
        out = self.out_layer(out.permute(0,2,1)).permute(0,2,1)
        return out
class SingleDecoder(nn.Module):
    def __init__(self,c_in = 64,c_out=80,factor=3,seg_len=30):
        super().__init__()
        self.net = nn.Sequential()
        if seg_len == 30:
            up1 = 2
            up2 = 3
            up3 = 5
        elif seg_len == 20:
            up1 = 2
            up2 = 2
            up3 = 5
        elif seg_len == 10:
            up1 = 1
            up2 = 2
            up3 = 5
        elif seg_len == 40:
            up1 = 2
            up2 = 4
            up3 = 5
        elif seg_len == 50:
            up1 = 2
            up2 = 5
            up3 = 5
        elif seg_len ==6:
            up1 = 2
            up2 = 3
            up3 = 1
        elif seg_len == 4:
            up1 = 2
            up2 = 2
            up3 = 1
        elif seg_len == 8:
            up1 = 2
            up2 = 2
            up3 = 2
        self.net.add_module('Conv1',ConvLayer(c_in=c_in,c_out=64*up1,ks=1,pad=0)) # B hid 1 -> B seg_len*hid 1
        self.net.add_module('ResizeLayerUp',ResizeLayerUp(factor=up1)) # B  hid seg_len
        self.net.add_module('Conv2',ConvLayer(c_in=64,c_out=64*up2,ks=3,pad=1))
        self.net.add_module('ResizeLayerUp2',ResizeLayerUp(factor=up2))
        self.net.add_module('Conv3',ConvLayer(c_in=64,c_out=64*up3,ks=3,pad=1))
        self.net.add_module('ResizeLayerUp3',ResizeLayerUp(factor=up3))
        self.out_layer = nn.Linear(64,c_out)

    def forward(self,x):
        out = self.net(x)
        return self.out_layer(out.permute(0,2,1)).permute(0,2,1)
class Encoder(nn.Module):
    def __init__(self,c_in=80,hid=64):
        super().__init__()
        self.net = nn.Sequential()
        self.net.add_module('Conv1',ConvLayer(c_in=c_in,c_out=256))
        self.net.add_module('Conv2',ConvLayer(c_in=256,c_out=512))
        self.net.add_module('Conv3',ConvLayer(c_in=512,c_out=512,res=True))
        self.net.add_module('Conv4',ConvLayer(c_in=512,c_out=512,res=True))
        self.net.add_module('Conv5',ConvLayer(c_in=512,c_out=512,res=True))
        self.net.add_module('Linear1',LinearLayer(c_in=512,c_out=256))
        self.net.add_module('Linear2',LinearLayer(c_in=256,c_out=128))
        self.out_layer = nn.Linear(128,hid)
    def forward(self,x):
        out = self.net(x)
        return self.out_layer(out.permute(0,2,1)).permute(0,2,1)
class Decoder(nn.Module):
    def __init__(self,c_in = 64,c_out=80):
        super().__init__()
        self.net = nn.Sequential()
        self.net.add_module('Conv1',ConvLayer(c_in=c_in,c_out=256))
        self.net.add_module('Conv2',ConvLayer(c_in=256,c_out=512))
        self.net.add_module('Conv3',ConvLayer(c_in=512,c_out=512,res=True))
        self.net.add_module('Conv4',ConvLayer(c_in=512,c_out=512,res=True))
        self.net.add_module('Conv5',ConvLayer(c_in=512,c_out=512,res=True))
        self.net.add_module('Linear1',LinearLayer(c_in=512,c_out=256))
        self.net.add_module('Linear2',LinearLayer(c_in=256,c_out=128))
        self.out_layer = nn.Linear(128,c_out)

    def forward(self,x):
        out = self.net(x)
        return self.out_layer(out.permute(0,2,1)).permute(0,2,1)
class Encoder2(nn.Module):
    def __init__(self,c_in=80,hid=64):
        super().__init__()
        self.net = nn.Sequential()
        self.net.add_module('Conv1',ConvLayer(c_in=c_in,c_out=64))
        self.net.add_module('Conv2',ConvLayer(c_in=64,stride=2,c_out=64))
        self.net.add_module('Conv3',ConvLayer(c_in=64,c_out=64))
        self.net.add_module('Linear1',LinearLayer(c_in=64,c_out=128))
        self.net.add_module('Linear2',LinearLayer(c_in=128,c_out=64))
        self.out_layer = nn.Linear(64,hid)
    def forward(self,x):
        out = self.net(x)
        return self.out_layer(out.permute(0,2,1)).permute(0,2,1)
class Decoder2(nn.Module):
    def __init__(self,c_in = 64,c_out=80):
        super().__init__()
        self.net = nn.Sequential()
        self.net.add_module('Conv1',ConvLayer(c_in=c_in,c_out=128))
        self.net.add_module('Conv2',TransConvLayer(c_in=128,c_out=64,stride=2))
        self.net.add_module('Conv3',ConvLayer(c_in=64,c_out=60))
        self.net.add_module('Linear1',LinearLayer(c_in=60,c_out=64))
        self.net.add_module('Linear2',LinearLayer(c_in=64,c_out=64))
        self.out_layer = nn.Linear(64,c_out)

    def forward(self,x):
        out = self.net(x)
        return self.out_layer(out.permute(0,2,1)).permute(0,2,1)
class Encoder4(nn.Module):
    def __init__(self,c_in=80,hid=64):
        super().__init__()
        self.net = nn.Sequential()
        self.net.add_module('Conv1',ConvLayer(c_in=c_in,c_out=64))
        self.net.add_module('Conv2',ConvLayer(c_in=64,stride=2,c_out=64))
        self.net.add_module('Conv3',ConvLayer(c_in=64,stride=2,c_out=64))
        self.net.add_module('Linear1',LinearLayer(c_in=64,c_out=128))
        self.net.add_module('Linear2',LinearLayer(c_in=128,c_out=64))
        self.out_layer = nn.Linear(64,hid)
    def forward(self,x):
        out = self.net(x)
        return self.out_layer(out.permute(0,2,1)).permute(0,2,1)
class Decoder4(nn.Module):
    def __init__(self,c_in = 64,c_out=80):
        super().__init__()
        self.net = nn.Sequential()
        self.net.add_module('Conv1',ConvLayer(c_in=c_in,c_out=128))
        self.net.add_module('Conv2',TransConvLayer(c_in=128,c_out=64,stride=2))
        self.net.add_module('Conv3',TransConvLayer(c_in=64,c_out=60,stride=2))
        self.net.add_module('Linear1',LinearLayer(c_in=60,c_out=64))
        self.net.add_module('Linear2',LinearLayer(c_in=64,c_out=64))
        self.out_layer = nn.Linear(64,c_out)

    def forward(self,x):
        out = self.net(x)
        return self.out_layer(out.permute(0,2,1)).permute(0,2,1)
