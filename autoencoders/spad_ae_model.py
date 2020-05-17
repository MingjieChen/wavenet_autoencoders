import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import math
from .module import ConvLayer,LinearLayer,ResizeLayerUp,ResizeLayerDown,FlatenLayer,SingleEncoder,SingleDecoder
#from .module import Encoder,Decoder
from .wavenet_ae_model import INEncoder, SpeakerEncoder, AdaIN, Encoder

class SpeakerClassifier(nn.Module):
    def __init__(self,c_in=64, hid=256, n_spks=153):
        super().__init__()
        self.c_in = c_in
        self.hid = hid
        self.n_spks = n_spks
        self.act = nn.ReLU()
        
        self.in_conv_layer = nn.Conv1d(c_in, hid, kernel_size = 1)
        self.first_conv_layer = nn.Conv1d(hid, hid, 3, padding = 1, stride=2)
        self.second_conv_layer = nn.Conv1d(hid, hid, 3, padding = 1, stride=2)

        self.pooling_layer = nn.AdaptiveAvgPool1d(1)

        self.dense1 = nn.Linear(hid, hid)
        self.dense2 = nn.Linear(hid, n_spks)

    def forward(self,x):
        out = x
        out = self.in_conv_layer(out)
        out = self.act(out)

        out = self.first_conv_layer(out)
        out = self.act(out)

        out = self.second_conv_layer(out)
        out = self.act(out)
        
        out = self.pooling_layer(out).squeeze(2)
        out = self.dense1(out)
        out = self.dense2(out)
        return out

class AdaINModule(nn.Module):
    def __init__(self, c_in, hid):
        super().__init__()

        self.ada1 = AdaIN(c_in = c_in, hid = hid)
        self.relu1 = nn.ReLU()
 
        self.ada2 = AdaIN(c_in = hid, hid = hid)
        self.relu2 = nn.ReLU()
        
        self.ada3 = AdaIN(c_in = hid, hid = hid)
        self.relu3 = nn.ReLU()
        
        self.ada4 = AdaIN(c_in = hid, hid = hid)
        self.relu4 = nn.ReLU()
    
    
    def forward(self, x , cond):
        out = x

        out = self.ada1(out, cond)
        out = self.relu1(out)

        out = self.ada2(out, cond)
        out = self.relu2(out)

        out = self.ada3(out, cond)
        out = self.relu3(out)

        out = self.ada4(out, cond)
        out = self.relu4(out)

        return out

class Model(nn.Module):
    def __init__(self,c_in=80,hid=64,n_spks=153, wavenet = None):
        super().__init__()
        self.enc = Encoder(c_in = c_in, hid=hid)
        self.wavenet = wavenet
        self.cls = SpeakerClassifier(c_in = hid, hid = 256, n_spks = n_spks)
        self.sp_enc = SpeakerEncoder(c_in = c_in, hid = 256, c_out = hid)
        self.adain = AdaINModule(c_in = hid, hid = hid)
    
    
    def forward(self, x, c, g, softmax = False):
        
        lat = self.enc(c)
        sp_cond = self.sp_enc(c)
        
        lat = self.adain(lat, sp_cond)

        logits = self.cls(lat)
        y_hat = self.wavenet(x, lat, g, softmax)
        return (y_hat,logits)
    
    
    def incremental_forward(self,initial_input,c,g,T,softmax,quantize,tqdm,log_scale_min):
        with torch.no_grad():

            lat = self.encr(c)
            sp_cond = self.sp_enc(c)
            
            lat = self.adain(lat, sp_cond)
            y_hat = self.wavenet.incremental_forward(initial_input, c=lat,g=g,T=T,softmax=softmax,quantize=quantize,tqdm = tqdm,log_scale_min = log_scale_min)
        
        return y_hat

    def encode(self,x):
        lat = self.enc(x)
        #out = self.dec(lat)
        return lat

    def ae_train(self,x, c, g, softmax = False):
        lat = self.enc(c)
        sp_cond = self.sp_enc(c)
        lat = self.adain(lat, sp_cond)
        y_hat = self.wavenet(x, lat, g, softmax)
        return y_hat

    def sp_train(self,x):
        lat = self.enc(x)
        logits = self.cls(lat)
        return logits
        
