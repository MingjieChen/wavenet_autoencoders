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
class Model(nn.Module):
    def __init__(self,c_in=80,hid=64,seg_len=20):
        super().__init__()
        self.enc = Encoder(c_in,hid=hid)
        self.dec = Decoder(c_in = hid, c_out = c_in )
    def forward(self,x):
        lat = self.enc(x)
        out = self.dec(lat)
        return out

    def encode(self,x):
        lat = self.enc(x)
        out = self.dec(lat)
        return lat,out
class Model2(nn.Module):
    def __init__(self,c_in=80,hid=64,seg_len=20):
        super().__init__()
        self.enc = Encoder2(c_in,hid=hid)
        self.dec = Decoder2(c_in = hid, c_out = c_in )
    def forward(self,x):
        lat = self.enc(x)
        out = self.dec(lat)
        return out

    def encode(self,x):
        lat = self.enc(x)
        return lat
class Model4(nn.Module):
    def __init__(self,c_in=80,hid=64,seg_len=20):
        super().__init__()
        self.enc = Encoder4(c_in,hid=hid)
        self.dec = Decoder4(c_in = hid, c_out = c_in )
    def forward(self,x):
        lat = self.enc(x)
        out = self.dec(lat)
        return out

    def encode(self,x):
        lat = self.enc(x)
        out = self.dec(lat)
        return lat,out
