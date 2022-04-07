import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.autograd import Variable
from vector_quantization import VectorQuantize

class ConvReLURes(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride =1):
        super().__init__()
        self.stride = stride
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size, stride, padding=kernel_size//2,bias=True)
        self.relu = nn.ReLU()
    def forward(self,x):
        out = self.conv(x)
        out = self.relu(out)
        if self.stride == 1 and self.dim_in == self.dim_out:
            out += x
            
        return out



class Encoder(nn.Module):
    def __init__(self, hid=768,c_in=39,c_out=64):
        super().__init__()
        
        
        block = []
        block += [ConvReLURes(c_in, hid, 3,1)]
        block += [ConvReLURes(hid, hid, 3,1)]
        block += [ConvReLURes(hid, hid, 5,2)]
        block += [ConvReLURes(hid, hid, 5,2)]
        block += [ConvReLURes(hid, hid, 3,1)]
        block += [ConvReLURes(hid, hid, 3,1)]
        for _ in range(4):
            block += [ConvReLURes(hid, hid, 1,1)]
        
        self.net = nn.Sequential(*block)    
        

        
        self.lin = nn.Linear(hid,c_out)

    def forward(self,x):
        out = self.net(x)
        out = self.lin(out.permute(0,2,1)).permute(0,2,1)
        return out
class VQVAE(nn.Module):
                
    def __init__(self, c_in = 39, hid=64,K = 256, wavenet = None, encoder_hid=768):
        super().__init__()
        self.wavenet = wavenet
        

        self.encoder = Encoder(c_in = c_in, c_out = hid, hid = encoder_hid)
        
        
        self.vq = VectorQuantize(K = K, D = hid)
        
    
    
    def forward(self,x,c,g,softmax=False):
        lat = self.encoder(c)
        
        
        quant,vq_loss,perp = self.vq(lat)
        y_hat = self.wavenet(x,quant,g,softmax)
        return (y_hat , vq_loss,perp)
    def incremental_forward(self,initial_input,c,g,T,softmax,quantize,tqdm,log_scale_min):
        with torch.no_grad():

            lat = self.encoder(c)
            quant,vq_loss,perp = self.vq(lat)
            y_hat = self.wavenet.incremental_forward(initial_input, c=quant,g=g,T=T,softmax=softmax,quantize=quantize,tqdm = tqdm,log_scale_min = log_scale_min)
        return y_hat
    def encode(self,x):
        with torch.no_grad():
            out = self.encoder(x)    
            quant,vq_loss,perp = self.vq(out)
        return quant





    
