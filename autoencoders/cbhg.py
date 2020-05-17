import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import math


class Encoder(nn.Module):

    def __init__(self, c_in = 39, c_h1 = 64, c_h2=128, embed_size=64):
        super().__init__()

        c_h3 = embed_size

        self.conv1s = nn.ModuleList(
            [nn.Conv1d(c_in,c_h1,kernel_size=k) for k in range(1,8)]    
        )

        self.conv2 = nn.Conv1d(len(self.conv1s) * c_h1 + c_in, c_h2, kernel_size=1)
        self.conv3 = nn.Conv1d(c_h2, c_h2, kernel_size=3)
        self.conv4 = nn.Conv1d(c_h2, c_h2, kernel_size=3)
        self.conv5 = nn.Conv1d(c_h2, c_h2, kernel_size=3)
        self.conv6 = nn.Conv1d(c_h2, c_h2, kernel_size=3)
        self.conv7 = nn.Conv1d(c_h2, c_h2, kernel_size=3)
        self.conv8 = nn.Conv1d(c_h2, c_h2, kernel_size=3)
        
        self.dense1 = nn.Linear(c_h2,c_h2)
        self.dense2 = nn.Linear(c_h2,c_h2)
        self.dense3 = nn.Linear(c_h2,c_h2)
        self.dense4 = nn.Linear(c_h2,c_h2)

        self.ins_norm1 = nn.InstanceNorm1d(c_h2)
        self.ins_norm2 = nn.InstanceNorm1d(c_h2)
        self.ins_norm3 = nn.InstanceNorm1d(c_h2)
        self.ins_norm4 = nn.InstanceNorm1d(c_h2)
        self.ins_norm5 = nn.InstanceNorm1d(c_h2)
        self.ins_norm6 = nn.InstanceNorm1d(c_h2)

        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)
        self.drop3 = nn.Dropout(0.5)
        self.drop4 = nn.Dropout(0.5)
        self.drop5 = nn.Dropout(0.5)
        self.drop6 = nn.Dropout(0.5)

        
