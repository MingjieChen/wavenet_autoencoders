import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.autograd import Variable

def pad_layer(inp, layer, pad_type = 'reflect'):
    
    kernel_size = layer.kernel_size[0]
    if kernel_size %2 ==0:
        pad = (kernel_size // 2, kernel_size // 2 -1 )
    else:
        pad = (kernel_size // 2, kernel_size // 2)
    
    inp = F.pad(inp, pad = pad, mode=pad_type)

    out = layer(inp)

    return out
class SpeakerEncoder(nn.Module):
    def __init__(self, c_in, hid, c_out, ):
        super().__init__()

        self.c_in = c_in
        self.hid = hid
        self.c_out = c_out
        self.act = nn.ReLU()
        
        self.in_conv_layer = nn.Conv1d(c_in, hid, kernel_size = 1)
        self.first_conv_layer = nn.Conv1d(hid, hid, 3, padding = 1, stride=2)
        self.second_conv_layer = nn.Conv1d(hid, hid, 3, padding = 1, stride=2)

        self.pooling_layer = nn.AdaptiveAvgPool1d(1)

        self.dense1 = nn.Linear(hid, hid)
        self.dense2 = nn.Linear(hid, c_out)
    def forward(self, x):
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
class SpeakerEncoder1(nn.Module):
    def __init__(self, c_in, hid, c_out, c_bank = 128, bank_size = 8):
        super().__init__()

        self.c_in = c_in
        self.hid = hid
        self.c_out = c_out
        self.act = nn.ReLU()
        
        self.conv_bank_modules = nn.ModuleList(
                                    [nn.Conv1d(c_in, c_bank, kernel_size = k) for k in range(1, bank_size +1)]
                                        )
        
        in_channels = c_bank * bank_size + c_in
        
        self.in_conv_layer = nn.Conv1d(in_channels, hid, kernel_size = 1)
        
        self.conv_layer1 = nn.Conv1d(hid, hid, 5, padding = 2) 
        self.conv_layer2 = nn.Conv1d(hid, hid, 5, padding = 2)
        self.conv_layer3 = nn.Conv1d(hid, hid, 5, padding = 2) 
        self.conv_layer4 = nn.Conv1d(hid, hid, 5, stride=2, padding = 2)
        self.conv_layer5 = nn.Conv1d(hid, hid, 5, padding = 2)
        self.conv_layer6 = nn.Conv1d(hid, hid, 5, stride=2, padding = 2)

        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.drop_layer = nn.Dropout(0.1)
        
        self.dense1 = nn.Linear(hid, hid) 
        self.dense2 = nn.Linear(hid, hid) 
        self.dense3 = nn.Linear(hid, hid) 
        self.dense4 = nn.Linear(hid, hid) 
        self.dense_out = nn.Linear(hid, c_out)
    

    def conv_bank(self, x, module_list, act, pad_type = 'reflect'):
        
        outs = []

        for layer in module_list:
            out = act(pad_layer(x, layer, pad_type))
            outs.append(out)
        out = torch.cat(outs + [x], dim = 1)

        return out
    
    def conv_block(self, x, layers, act, drop, sub = False):
        
        out = x

        y = layers[0](out)
        y = act(y)
        y = drop(y)

        y = layers[1](y)
        y = act(y)
        y = drop(y)

        if sub:
            out = F.avg_pool1d(out, kernel_size = 2, ceil_mode = True)
        out = y + out
        
        return out
    def dense_block(self, x, layers, act, drop):
        
        out = x

        y = layers[0](out)
        y = act(y)
        y = drop(y)

        y = layers[1](out)
        y = act(y)
        y = drop(y)

        out = y + out

        return out
    def forward(self, x):
        out = x

        out = self.conv_bank(out, self.conv_bank_modules, self.act)

        out = pad_layer(out, self.in_conv_layer)
        out = self.act(out)
        out = self.drop_layer(out)

        out = self.conv_block(out, [self.conv_layer1, self.conv_layer2], self.act, self.drop_layer, False)
        out = self.conv_block(out, [self.conv_layer3, self.conv_layer4], self.act, self.drop_layer, True)
        out = self.conv_block(out, [self.conv_layer5, self.conv_layer6], self.act, self.drop_layer, True)
        
        out = self.pooling_layer(out).squeeze(2)
        
        out = self.dense_block(out, [self.dense1, self.dense2], self.act, self.drop_layer)
        out = self.dense_block(out, [self.dense3, self.dense4], self.act, self.drop_layer)
        
        out = self.dense_out(out)
        return out
