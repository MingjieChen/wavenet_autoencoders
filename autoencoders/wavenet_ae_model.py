import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.autograd import Variable
from .speaker_encoder import SpeakerEncoder, SpeakerEncoder1
from .vq_fn import VectorQuantize, SlicedVectorQuantize, SlicedVectorQuantizeEMA, VectorQuantizeEMA, SlicedVectorQuantize4
from .gumbel_softmax_fn import gumbel_softmax
class INConvReLURes(nn.Module):
    def __init__(self, n_in_chan, n_out_chan, filter_sz, stride =1,do_conv=True, do_res=True, do_norm = False, name=None):
        super().__init__()

        self.do_norm = do_norm
        self.do_res = do_res
        self.do_conv = do_conv
        if self.do_res:
            if stride != 1:
                raise Exception("Stride must be 1 if do_res")
        if self.do_norm:
            self.norm_layer = nn.InstanceNorm1d(n_out_chan, affine=False)
        self.name = name
        self.n_in = n_in_chan
        self.n_out = n_out_chan
        if self.do_conv:
            if filter_sz %2 ==0:
                self.pad = (filter_sz//2, filter_sz //2-1 )
            else:
                self.pad = (filter_sz//2, filter_sz//2)

            self.conv = nn.Conv1d(self.n_in, self.n_out, filter_sz, stride, padding=0,bias=True)
            self.relu1 = nn.ReLU()
        if self.do_res:
            self.relu2 = nn.ReLU()
    def forward(self,x):
        
        if self.do_conv:
            x_pad = F.pad(x,pad=self.pad,mode='reflect')
            out = self.conv(x_pad)
            out = self.relu1(out)
        else:
            out = x
        
        
        if self.do_res:
            if not self.do_conv:
                out = self.relu2(out)
                out += x
            else:
                out += x
        
        if self.do_norm:
            out = self.norm_layer(out)
        return out

class ConvReLURes(nn.Module):
    def __init__(self, n_in_chan, n_out_chan, filter_sz, stride =1,do_conv=True, do_res=True, name=None):
        super().__init__()

        self.do_res = do_res
        self.do_conv = do_conv
        if self.do_res:
            if stride != 1:
                raise Exception("Stride must be 1 if do_res")
        self.name = name
        self.n_in = n_in_chan
        self.n_out = n_out_chan
        if self.do_conv:
            if filter_sz %2 ==0:
                self.pad = (filter_sz//2, filter_sz //2-1 )
            else:
                self.pad = (filter_sz//2, filter_sz//2)

            self.conv = nn.Conv1d(self.n_in, self.n_out, filter_sz, stride, padding=0,bias=True)
            self.relu1 = nn.ReLU()
        if self.do_res:
            self.relu2 = nn.ReLU()
    def forward(self,x):
        if self.do_conv:
            x_pad = F.pad(x,pad=self.pad,mode='reflect')
            out = self.conv(x_pad)
            out = self.relu1(out)
        else:
            out = x
        if self.do_res:
            if not self.do_conv:
                out = self.relu2(out)
                out += x
            else:
                out += x
        return out



class INEncoder50(nn.Module):
    def __init__(self, hid=768,c_in=39,c_out=64):
        super().__init__()
        
        stack_in_chan = [c_in,hid,hid,hid,hid,hid,hid,hid]
        stack_filter_sz = [3,3,4,3,1,1,1,1]
        stack_strides = [1,1,2,1,1,1,1,1]
        stack_residual = [False,True,False,True,True,True,True,True]
        stack_conv = [True,True,True,True,False,False,False,False]
        stack_norm = [False, True, False, True, False, True, False, True]
        
        
        #stack_in_chan = [c_in,hid]
        #stack_filter_sz = [3,4]
        #stack_strides = [1,2]
        #stack_residual = [False,False]
        stack_info = zip(stack_in_chan,stack_filter_sz,stack_strides,stack_residual,stack_conv, stack_norm)

        self.net = nn.Sequential()

        for i, (in_chan, filt_sz, stride, do_res, do_conv, do_norm) in enumerate(stack_info):
            name = f'CRR_{i}(filt_sz {filt_sz} stri {stride} res {do_res})'
            mod = INConvReLURes(in_chan, hid, filt_sz, stride,do_conv, do_res, do_norm, name)
            self.net.add_module(str(i),mod)
        self.lin = nn.Linear(hid,c_out)

    def forward(self,x):
        out = self.net(x)
        out = self.lin(out.permute(0,2,1)).permute(0,2,1)
        return out
class Encoder50(nn.Module):
    def __init__(self, hid=768,c_in=39,c_out=64):
        super().__init__()
        
        stack_in_chan = [c_in,hid,hid,hid,hid,hid,hid,hid]
        stack_filter_sz = [3,3,4,3,1,1,1,1]
        stack_strides = [1,1,2,1,1,1,1,1]
        stack_residual = [False,True,False,True,True,True,True,True]
        stack_conv = [True,True,True,True,False,False,False,False]
        
        
        #stack_in_chan = [c_in,hid]
        #stack_filter_sz = [3,4]
        #stack_strides = [1,2]
        #stack_residual = [False,False]
        stack_info = zip(stack_in_chan,stack_filter_sz,stack_strides,stack_residual,stack_conv)

        self.net = nn.Sequential()

        for i, (in_chan, filt_sz, stride, do_res, do_conv) in enumerate(stack_info):
            name = f'CRR_{i}(filt_sz {filt_sz} stri {stride} res {do_res})'
            mod = ConvReLURes(in_chan, hid, filt_sz, stride,do_conv, do_res, name)
            self.net.add_module(str(i),mod)
        self.lin = nn.Linear(hid,c_out)

    def forward(self,x):
        out = self.net(x)
        out = self.lin(out.permute(0,2,1)).permute(0,2,1)
        return out
class Encoder(nn.Module):
    def __init__(self, hid=768,c_in=39,c_out=64):
        super().__init__()
        
        #stack_in_chan = [c_in,hid,hid,hid,hid,hid,hid,hid,hid]
        #stack_filter_sz = [3,3,4,3,3,1,1,1,1]
        #stack_strides = [1,1,2,1,1,1,1,1,1]
        #stack_residual = [False,True,False,True,True,True,True,True,True]
        #stack_conv = [True,True,True,True,True,False,False,False,False]
        
        stack_in_chan = [c_in,hid,hid,hid,hid,hid,hid,hid,hid,hid]
        stack_filter_sz = [3,3,4,4,3,3,1,1,1,1,1]
        stack_strides = [1,1,2,2,1,1,1,1,1,1,1]
        stack_residual = [False,True,False,False,True,True,True,True,True,True,True]
        stack_conv = [True,True,True,True,True,True,False,False,False,False,False]
        
        #stack_in_chan = [c_in,hid]
        #stack_filter_sz = [3,4]
        #stack_strides = [1,2]
        #stack_residual = [False,False]
        stack_info = zip(stack_in_chan,stack_filter_sz,stack_strides,stack_residual,stack_conv)

        self.net = nn.Sequential()

        for i, (in_chan, filt_sz, stride, do_res, do_conv) in enumerate(stack_info):
            name = f'CRR_{i}(filt_sz {filt_sz} stri {stride} res {do_res})'
            mod = ConvReLURes(in_chan, hid, filt_sz, stride,do_conv, do_res, name)
            self.net.add_module(str(i),mod)
        self.lin = nn.Linear(hid,c_out)

    def forward(self,x):
        out = self.net(x)
        out = self.lin(out.permute(0,2,1)).permute(0,2,1)
        return out
class INEncoder(nn.Module):
    def __init__(self, hid=768,c_in=39,c_out=64):
        super().__init__()
        
        #stack_in_chan = [c_in,hid,hid,hid,hid,hid,hid,hid,hid]
        #stack_filter_sz = [3,3,4,3,3,1,1,1,1]
        #stack_strides = [1,1,2,1,1,1,1,1,1]
        #stack_residual = [False,True,False,True,True,True,True,True,True]
        #stack_conv = [True,True,True,True,True,False,False,False,False]
        
        stack_in_chan = [c_in,hid,hid,hid,hid,hid,hid,hid,hid,hid]
        stack_filter_sz = [3,3,4,4,3,3,1,1,1,1,1]
        stack_strides = [1,1,2,2,1,1,1,1,1,1,1]
        stack_residual = [False,True,False,False,True,True,True,True,True,True,True]
        stack_conv = [True,True,True,True,True,True,False,False,False,False,False]
        stack_norm = [False, True, False, True, False, True, False, True, False, True]
        #stack_in_chan = [c_in,hid]
        #stack_filter_sz = [3,4]
        #stack_strides = [1,2]
        #stack_residual = [False,False]
        stack_info = zip(stack_in_chan,stack_filter_sz,stack_strides,stack_residual,stack_conv, stack_norm)

        self.net = nn.Sequential()

        for i, (in_chan, filt_sz, stride, do_res, do_conv, do_norm) in enumerate(stack_info):
            name = f'CRR_{i}(filt_sz {filt_sz} stri {stride} res {do_res} norm {do_norm})'
            mod = INConvReLURes(in_chan, hid, filt_sz, stride,do_conv, do_res, do_norm, name)
            self.net.add_module(str(i),mod)
        self.lin = nn.Linear(hid,c_out)

    def forward(self,x):
        out = self.net(x)
        out = self.lin(out.permute(0,2,1)).permute(0,2,1)
        return out
        
class AdaIN(nn.Module):
    def __init__(self, c_in = 64, hid = 64):
        super().__init__()
        self.hid = hid
        self.conv1 = nn.Conv1d(c_in, hid, kernel_size = 3, padding = 1)
        self.lin = nn.Linear(hid, 2*hid)
        self.norm_layer = nn.InstanceNorm1d(hid, affine = False)
    def forward(self, x, cond):
        cond = self.lin(cond)
        mean, std = cond[:, :self.hid], cond[:, self.hid:]
        out = x * std.unsqueeze(dim = 2) + mean.unsqueeze(dim = 2)
        return out

class INAE(nn.Module):
    def __init__(self, c_in = 39, hid=64, wavenet = None, frame_rate = 25, adain = True):
        super().__init__()
        self.wavenet = wavenet
        if frame_rate == 50:
            self.encoder = INEncoder50(c_in = c_in, c_out = hid)
        else:
            self.encoder = INEncoder(c_in = c_in, c_out = hid)

        self.adain = adain

        if self.adain:
            self.sp_encoder = SpeakerEncoder(c_in = c_in, c_out = hid, hid = 256)
            self.ada1 = AdaIN(c_in = hid, hid = hid)
            self.ada2 = AdaIN(c_in = hid, hid = hid)
    
    def forward(self,x,c,g,softmax=False):
        lat = self.encoder(c)
        
        if self.adain:

            sp_cond = self.sp_encoder(c)
            lat = self.ada1(lat, sp_cond)
            lat = self.ada2(lat, sp_cond)
        y_hat = self.wavenet(x,lat,g,softmax)
        return y_hat
    def incremental_forward(self,initial_input,c,g,T,softmax,quantize,tqdm,log_scale_min, tar_c = None):
        if tar_c is None :
            tar_c = c
        with torch.no_grad():

            lat = self.encoder(c)
            if self.adain:
                sp_cond = self.sp_encoder(tar_c)
                lat = self.ada1(lat, sp_cond)
                lat = self.ada2(lat, sp_cond)
            y_hat = self.wavenet.incremental_forward(initial_input, c=lat,g=g,T=T,softmax=softmax,quantize=quantize,tqdm = tqdm,log_scale_min = log_scale_min)
        return y_hat
    def encode(self,x):
        with torch.no_grad():
            out = self.encoder(x)    
        return out
class INAE1(nn.Module):
    def __init__(self, c_in = 39, hid=64, wavenet = None, frame_rate = 25, adain = True):
        super().__init__()
        self.wavenet = wavenet
        if frame_rate == 50:
            self.encoder = INEncoder50(c_in = c_in, c_out = hid)
        else:
            self.encoder = INEncoder(c_in = c_in, c_out = hid)

        self.adain = adain

        if self.adain:
            self.sp_encoder = SpeakerEncoder1(c_in = c_in, c_out = hid, hid = 256)
            self.ada1 = AdaIN(c_in = hid, hid = hid)
            self.ada2 = AdaIN(c_in = hid, hid = hid)
            self.ada3 = AdaIN(c_in = hid, hid = hid)
            self.ada4 = AdaIN(c_in = hid, hid = hid)
            self.ada5 = AdaIN(c_in = hid, hid = hid)
            self.ada6 = AdaIN(c_in = hid, hid = hid)
    
    def forward(self,x,c,g,softmax=False):
        lat = self.encoder(c)
        
        if self.adain:

            sp_cond = self.sp_encoder(c)
            lat = self.ada1(lat, sp_cond)
            lat = self.ada2(lat, sp_cond)
            lat = self.ada3(lat, sp_cond)
            lat = self.ada4(lat, sp_cond)
            lat = self.ada5(lat, sp_cond)
            lat = self.ada6(lat, sp_cond)
        y_hat = self.wavenet(x,lat,g,softmax)
        return y_hat
    def incremental_forward(self,initial_input,c,g,T,softmax,quantize,tqdm,log_scale_min, tar_c = None):
        if tar_c is None :
            tar_c = c
        with torch.no_grad():

            lat = self.encoder(c)
            if self.adain:
                sp_cond = self.sp_encoder(tar_c)
                lat = self.ada1(lat, sp_cond)
                lat = self.ada2(lat, sp_cond)
                lat = self.ada3(lat, sp_cond)
                lat = self.ada4(lat, sp_cond)
                lat = self.ada5(lat, sp_cond)
                lat = self.ada6(lat, sp_cond)
            y_hat = self.wavenet.incremental_forward(initial_input, c=lat,g=g,T=T,softmax=softmax,quantize=quantize,tqdm = tqdm,log_scale_min = log_scale_min)
        return y_hat
    def encode(self,x):
        with torch.no_grad():
            out = self.encoder(x)    
        return out

class NewINAE(nn.Module):
    def __init__(self, c_in = 39, hid=64, wavenet = None, frame_rate = 25):
        super().__init__()
        self.wavenet = wavenet
        if frame_rate == 50:
            self.encoder = INEncoder50(c_in = c_in, c_out = hid)
        else:
            self.encoder = INEncoder(c_in = c_in, c_out = hid)
        self.sp_encoder = SpeakerEncoder(c_in = c_in, c_out = hid, hid = 256)
        #self.encoder = INEncoder(c_in = c_in, c_out = hid)
        self.ada1 = AdaIN(c_in = hid, hid = hid)
        self.ada2 = AdaIN(c_in = hid, hid = hid)
    
    def forward(self, x, c, g, softmax=False):
        lat = self.encoder(c)
        sp_cond = self.sp_encoder(c)
        lat = self.ada1(lat, sp_cond)
        lat = self.ada2(lat, sp_cond)
        y_hat = self.wavenet(x, lat, sp_cond, softmax)
        return y_hat
    
    def incremental_forward(self,initial_input,c,g,T,softmax,quantize,tqdm,log_scale_min, tar_c = None):
        if tar_c is None :
            tar_c = c
        with torch.no_grad():

            lat = self.encoder(c)
            sp_cond = self.sp_encoder(tar_c)
            lat = self.ada1(lat, sp_cond)
            lat = self.ada2(lat, sp_cond)
            y_hat = self.wavenet.incremental_forward(initial_input, c=lat, g=sp_cond, T=T,softmax=softmax,quantize=quantize,tqdm = tqdm,log_scale_min = log_scale_min)
        return y_hat
    def encode(self,x):
        with torch.no_grad():
            out = self.encoder(x)    
        return out

class AE(nn.Module):
    def __init__(self, c_in = 39, hid=64, wavenet = None, frame_rate = 25):
        super().__init__()
        self.wavenet = wavenet
        if frame_rate == 50:
            self.encoder = Encoder50(c_in = c_in, c_out = hid)
        else:
            self.encoder = Encoder(c_in = c_in, c_out = hid)


    def forward(self,x,c,g,softmax=False):
        lat = self.encoder(c)
        y_hat = self.wavenet(x,lat,g,softmax)
        return y_hat
    def incremental_forward(self,initial_input,c,g,T,softmax,quantize,tqdm,log_scale_min):
        with torch.no_grad():

            lat = self.encoder(c)
            y_hat = self.wavenet.incremental_forward(initial_input, c=lat,g=g,T=T,softmax=softmax,quantize=quantize,tqdm = tqdm,log_scale_min = log_scale_min)
        return y_hat
    def encode(self,x):
        with torch.no_grad():
            out = self.encoder(x)    
        return out




class VQVAE(nn.Module):
                
    def __init__(self, c_in = 39, hid=64,K = 256, wavenet = None, time_jitter_prob = 0.12, frame_rate = 25, use_time_jitter = False, ema = False, sliced = False, ins_norm = False, post_conv = True, adain = False, dropout = False, drop_dim = 't', K1 = None, num_slices = 2):
        super().__init__()
        self.time_jitter_prob = time_jitter_prob
        self.wavenet = wavenet
        self.use_time_jitter = use_time_jitter
        #self.encoder = Encoder(c_in = c_in, c_out = hid)
        self.adain = adain
        
        self.dropout = dropout
        self.dropout_rate = 0.05
        #drop out axis, time or frequency.
        self.drop_dim = drop_dim
        

        self.ins_norm = ins_norm
        '''
        if self.adain:
            self.sp_encoder = SpeakerEncoder1(c_in = c_in, c_out = hid, hid = 256)
            self.ada1 = AdaIN(c_in = hid, hid = hid)
            self.ada2 = AdaIN(c_in = hid, hid = hid)
            self.ada3 = AdaIN(c_in = hid, hid = hid)
            self.ada4 = AdaIN(c_in = hid, hid = hid)
            self.ada5 = AdaIN(c_in = hid, hid = hid)
            self.ada6 = AdaIN(c_in = hid, hid = hid)
        '''
        if frame_rate == 50:
            self.encoder = Encoder50(c_in = c_in, c_out = hid)
        else:
            self.encoder = Encoder(c_in = c_in, c_out = hid)
        
        #if self.ins_norm:
        #    self.norm_layer = nn.InstanceNorm1d(hid, affine = False)
        
        if ema:
            if sliced:
                self.vq = SlicedVectorQuantizeEMA(K = K, D = hid)
            else:
                self.vq = VectorQuantizeEMA(K = K, D = hid)
            
        elif sliced:
            if num_slices == 2:
                self.vq = SlicedVectorQuantize(K = K, D = hid, beta = 0.25, dropout = self.dropout, dropout_rate = self.dropout_rate, K1 = K1)
            elif num_slices == 4:
                self.vq = SlicedVectorQuantize4(K = K, D = hid, beta = 0.25)
            else:
                raise Exception
        else:
            self.vq = VectorQuantize(K = K, D = hid, dropout = self.dropout, dropout_rate = self.dropout_rate)
        self.post_conv = post_conv
        if post_conv:
            self.conv = nn.Conv1d(hid,128,3,padding=1)
        
    
    
    def forward(self,x,c,g,softmax=False):
        lat = self.encoder(c)
        
        #if self.ins_norm:
        #    lat = self.norm_layer(lat)
        
        quant,vq_loss,perp = self.vq(lat)
        



        if self.training:
            if self.dropout:
                
                # if drop out axis is time, permute.
                if self.drop_dim == 't':
                    quant_perm = quant.permute(0,2,1) #B T C
                elif self.drop_dim == 'f':
                    quant_perm = quant    
                quant_perm_4d = quant_perm.unsqueeze(-1)
                
                # drop out along T dimension
                quant_perm_4d = torch.nn.functional.dropout2d(quant_perm_4d, self.dropout_rate, training = self.training)
                
                quant_perm = quant_perm_4d.squeeze(-1)
    
                if self.drop_dim == 't':
                    quant = quant_perm.permute(0,2,1) # B C T
                elif self.drop_dim == 'f':
                    quant = quant_perm


        if self.use_time_jitter and self.training:
            quant = self.time_jitter(quant)
        if self.post_conv:
            quant = self.conv(quant)
        '''
        if self.adain:

            sp_cond = self.sp_encoder(c)
            quant = self.ada1(quant, sp_cond)
            quant = self.ada2(quant, sp_cond)
            quant = self.ada3(quant, sp_cond)
            quant = self.ada4(quant, sp_cond)
            quant = self.ada5(quant, sp_cond)
            quant = self.ada6(quant, sp_cond)
        '''
        y_hat = self.wavenet(x,quant,g,softmax)
        return (y_hat , vq_loss,perp)
    def incremental_forward(self,initial_input,c,g,T,softmax,quantize,tqdm,log_scale_min):
        with torch.no_grad():

            lat = self.encoder(c)
            quant,vq_loss,perp = self.vq(lat)
            if self.post_conv:
                quant = self.conv(quant)
            if self.adain:

                sp_cond = self.sp_encoder(c)
                quant = self.ada1(quant, sp_cond)
                quant = self.ada2(quant, sp_cond)
                quant = self.ada3(quant, sp_cond)
                quant = self.ada4(quant, sp_cond)
                quant = self.ada5(quant, sp_cond)
                quant = self.ada6(quant, sp_cond)
            y_hat = self.wavenet.incremental_forward(initial_input, c=quant,g=g,T=T,softmax=softmax,quantize=quantize,tqdm = tqdm,log_scale_min = log_scale_min)
        return y_hat
    def encode(self,x):
        with torch.no_grad():
            out = self.encoder(x)    
            quant,vq_loss,perp = self.vq(out)
        return quant
    def time_jitter(self, x):
        '''
            apply time jitter on time dimension
        '''
        b, c, t = x.size()
        left_prob = torch.rand( t )
        right_prob = torch.rand( t )

        for i in range(0,t):
            if i == 0:
                if right_prob[i] < self.time_jitter_prob:
                    x[:,:,1] = x[:,:,0]
            elif i == t-1:
                if left_prob[i] < self.time_jitter_prob:
                    x[:,:,-2] = x[:,:,-1]
            else:
                if right_prob[i] < self.time_jitter_prob:
                    x[:,:,i+1] = x[:,:,i]
                if left_prob[i] < self.time_jitter_prob:
                    x[:,:,i-1] = x[:,:,i]
        return x





    
class CatWavAE(nn.Module):
                
    def __init__(self, c_in = 39, hid=64, wavenet = None,k=320,tau=1.0, frame_rate = 25, hard = False, slices = 4):
        super().__init__()
        self.hid = hid
        self.k = k
        self.tau = tau
        #self.lin1 = nn.Linear(hid,self.k)
        #self.lin2 = nn.Linear(self.k,hid)
        #self.softmax = nn.Softmax(dim=-1)
        self.wavenet = wavenet
        #self.encoder = Encoder(c_in = c_in, c_out = hid)
        if frame_rate == 50:
            self.encoder = Encoder50(c_in = c_in, c_out = hid)
        else:
            self.encoder = Encoder(c_in = c_in, c_out = hid)
        
        if slices == 4:
            self.bottleneck = GumbelSoftmaxModule4(K = self.k, D = hid, tau = tau, hard = hard, n_d = slices)
        elif slices == 2:
            self.bottleneck = GumbelSoftmaxModule(K = self.k, D = hid, tau = tau, hard = hard, n_d = slices)
        else:
            raise Exception
    def forward(self,x,c,g,softmax=False):
        lat = self.encoder(c)
        
        lat, perp = self.bottleneck(lat) 
        #B,C,T = lat.size()
        #flat_lat = lat.view(B*T,C)
        #logits_y = self.lin1(flat_lat)
        #q_y = self.softmax(logits_y)
        #log_q_y = torch.log(q_y + 1e-20)
        #y = gumbel_softmax(logits_y , self.tau, hard=False)

        #flat_new_lat = self.lin2(y)
        #new_lat = flat_new_lat.view(B,self.hid,T)
        #kl = torch.mean( torch.sum(q_y * (log_q_y - torch.log( torch.tensor(1.0/self.k).cuda() ) ) , dim=1 ) )
        y_hat = self.wavenet(x, lat, g, softmax)
        return (y_hat ,perp )
    def incremental_forward(self,initial_input,c,g,T,softmax,quantize,tqdm,log_scale_min):
        with torch.no_grad():

            lat = self.encoder(c)

            lat, perp = self.bottleneck(lat) 
            #b, f, t = lat.size()
            #flat_lat = lat.view(b*t,f)
            #logits_y = self.lin1(flat_lat)
            #q_y = self.softmax(logits_y)
            #log_q_y = torch.log(q_y + 1e-20)
            #y = gumbel_softmax(logits_y , self.tau, hard=True)

            #flat_new_lat = self.lin2(y)
            #new_lat = flat_new_lat.view(b,self.hid,t)
            #kl = torch.mean( torch.sum(q_y * (log_q_y - torch.log( torch.tensor(1.0/self.k).cuda() ) ) , dim=1 ) )
            y_hat = self.wavenet.incremental_forward(initial_input, c=lat,g=g,T=T,softmax=softmax,quantize=quantize,tqdm = tqdm,log_scale_min = log_scale_min)
        return y_hat
    def encode(self,x):
        with torch.no_grad():
            lat = self.encoder(x)
            lat,_ = self.bottleneck(lat)
        return lat

class GumbelSoftmaxModule4(nn.Module):

    def __init__(self, K, D, n_d = 4, tau = 2, hard = False):
        super().__init__()
        self.K = K
        self.D = D
        self.hard = hard
        self.tau = tau
        self.sub_D = self.D // n_d
        self.n_d = n_d
        self.embedding1 = nn.Embedding(K, self.sub_D)
        self.embedding1.weight.data.uniform_(-1. / K, 1. / K)
        
        self.embedding2 = nn.Embedding(K, self.sub_D)
        self.embedding2.weight.data.uniform_(-1. / K, 1. / K)

        self.embedding3 = nn.Embedding(K, self.sub_D)
        self.embedding3.weight.data.uniform_(-1. / K, 1. / K)
        
        self.embedding4 = nn.Embedding(K, self.sub_D)
        self.embedding4.weight.data.uniform_(-1. / K, 1. / K)
        

    def forward(self, x):
        
        x = x.permute(0,2,1).contiguous()

        B, T, C = x.size()

        flat_in = x.view(-1, C)
        
        assert flat_in.size(1) == self.D

        flat_in1, flat_in2, flat_in3, flat_in4 = flat_in[:, : self.sub_D], flat_in[:, self.sub_D : 2 * self.sub_D], flat_in[:, 2 * self.sub_D : 3 * self.sub_D], flat_in[:, 3 * self.sub_D : ]

        code_sqr1 = torch.sum(self.embedding1.weight **2, dim = 1)
        code_sqr2 = torch.sum(self.embedding2.weight **2, dim = 1)
        code_sqr3 = torch.sum(self.embedding3.weight **2, dim = 1)
        code_sqr4 = torch.sum(self.embedding4.weight **2, dim = 1)

        in_sqr1 = torch.sum(flat_in1**2, dim = 1, keepdim = True)
        in_sqr2 = torch.sum(flat_in2**2, dim = 1, keepdim = True)
        in_sqr3 = torch.sum(flat_in3**2, dim = 1, keepdim = True)
        in_sqr4 = torch.sum(flat_in4**2, dim = 1, keepdim = True)

        dis1 = torch.addmm(code_sqr1 + in_sqr1, flat_in1, self.embedding1.weight.t(), alpha = -2.0, beta = 1.0)
        dis2 = torch.addmm(code_sqr2 + in_sqr2, flat_in2, self.embedding2.weight.t(), alpha = -2.0, beta = 1.0)
        dis3 = torch.addmm(code_sqr3 + in_sqr3, flat_in3, self.embedding3.weight.t(), alpha = -2.0, beta = 1.0)
        dis4 = torch.addmm(code_sqr4 + in_sqr4, flat_in4, self.embedding4.weight.t(), alpha = -2.0, beta = 1.0)

        
        if self.training:
            hard = self.hard
        else:
            hard = self.hard

        encodings1 = gumbel_softmax( -1. * dis1, self.tau, hard) # B*T, K
        encodings2 = gumbel_softmax( -1. * dis2, self.tau, hard)
        encodings3 = gumbel_softmax( -1. * dis3, self.tau, hard) # B*T, K
        encodings4 = gumbel_softmax( -1. * dis4, self.tau, hard)
        
        quant1 = torch.matmul(encodings1, self.embedding1.weight).view(B, T, self.sub_D)
        quant2 = torch.matmul(encodings2, self.embedding2.weight).view(B, T, self.sub_D)
        quant3 = torch.matmul(encodings3, self.embedding3.weight).view(B, T, self.sub_D)
        quant4 = torch.matmul(encodings4, self.embedding4.weight).view(B, T, self.sub_D)
        
        quant = torch.cat([quant1, quant2, quant3, quant4], dim = 2)

        avg_probs1 = torch.mean(encodings1, dim=0)
        avg_probs2 = torch.mean(encodings2, dim=0)
        avg_probs3 = torch.mean(encodings3, dim=0)
        avg_probs4 = torch.mean(encodings4, dim=0)

        perp1 = torch.exp( -1. * torch.sum(avg_probs1 * torch.log(avg_probs1 + 1e-10)))
        perp2 = torch.exp( -1. * torch.sum(avg_probs2 * torch.log(avg_probs2 + 1e-10)))
        perp3 = torch.exp( -1. * torch.sum(avg_probs3 * torch.log(avg_probs3 + 1e-10)))
        perp4 = torch.exp( -1. * torch.sum(avg_probs4 * torch.log(avg_probs4 + 1e-10)))
        perp = perp1 + perp2 + perp3 + perp4
        
        return quant.permute(0,2,1), perp
class GumbelSoftmaxModule(nn.Module):

    def __init__(self, K, D, n_d = 2, tau = 2, hard = False):
        super().__init__()
        self.K = K
        self.D = D
        self.hard = hard
        self.tau = tau
        self.sub_D = self.D // n_d
        self.n_d = n_d
        self.embedding1 = nn.Embedding(K, self.sub_D)
        self.embedding1.weight.data.uniform_(-1. / K, 1. / K)
        
        self.embedding2 = nn.Embedding(K, self.sub_D)
        self.embedding2.weight.data.uniform_(-1. / K, 1. / K)

        

    def forward(self, x):
        
        x = x.permute(0,2,1).contiguous()

        B, T, C = x.size()

        flat_in = x.view(-1, C)
        
        assert flat_in.size(1) == self.D

        flat_in1, flat_in2 = flat_in[:, : self.sub_D], flat_in[:, self.sub_D :]

        code_sqr1 = torch.sum(self.embedding1.weight **2, dim = 1)
        code_sqr2 = torch.sum(self.embedding2.weight **2, dim = 1)

        in_sqr1 = torch.sum(flat_in1**2, dim = 1, keepdim = True)
        in_sqr2 = torch.sum(flat_in2**2, dim = 1, keepdim = True)

        dis1 = torch.addmm(code_sqr1 + in_sqr1, flat_in1, self.embedding1.weight.t(), alpha = -2.0, beta = 1.0)
        dis2 = torch.addmm(code_sqr2 + in_sqr2, flat_in2, self.embedding2.weight.t(), alpha = -2.0, beta = 1.0)

        
        if self.training:
            hard = self.hard
        else:
            hard = self.hard

        encodings1 = gumbel_softmax( -1. * dis1, self.tau, hard) # B*T, K
        encodings2 = gumbel_softmax( -1. * dis2, self.tau, hard)
        
        quant1 = torch.matmul(encodings1, self.embedding1.weight).view(B, T, self.sub_D)
        quant2 = torch.matmul(encodings2, self.embedding2.weight).view(B, T, self.sub_D)
        
        quant = torch.cat([quant1, quant2], dim = 2)

        avg_probs1 = torch.mean(encodings1, dim=0)
        avg_probs2 = torch.mean(encodings2, dim=0)

        perp1 = torch.exp( -1. * torch.sum(avg_probs1 * torch.log(avg_probs1 + 1e-10)))
        perp2 = torch.exp( -1. * torch.sum(avg_probs2 * torch.log(avg_probs2 + 1e-10)))
        perp = perp1 + perp2
        
        return quant.permute(0,2,1), perp
