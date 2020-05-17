import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.autograd import Variable



class VectorQuantize(nn.Module):

    def __init__(self,K,D,beta=0.25, dropout = False, dropout_rate = 0.25):
        super().__init__()
        self.K = K
        self.D = D
        self.embedding = nn.Embedding(K,D)
        self.embedding.weight.data.uniform_(-1./K,1./K)
        self.beta=beta


        self.dropout = dropout
        self.dropout_rate = dropout_rate
    def forward(self,inputs):
        inputs = inputs.permute(0,2,1).contiguous() # B T F
        B, T ,F = inputs.size()

        flat_inputs = inputs.view(-1,self.D)

        in_sqr = torch.sum(flat_inputs**2,dim=1,keepdim=True)
        embed_sqr = torch.sum(self.embedding.weight**2,dim=1)

        dis = torch.addmm(embed_sqr + in_sqr, flat_inputs, self.embedding.weight.t(), alpha=-2.0,beta=1.0)
        encoding_ind = torch.argmin(dis,dim=1).unsqueeze(1)
        encodings = torch.zeros(B*T,self.K)#.type(torch.cuda.FloatTensor)
        if torch.cuda.is_available():
            encodings = encodings.type(torch.cuda.FloatTensor)
        encodings.scatter_(1,encoding_ind,1)

        
        quant = torch.matmul(encodings,self.embedding.weight).view(B,T,F)

        
        vq_loss = torch.mean((quant.detach() - inputs) **2)
        commit_loss = torch.mean((quant - inputs.detach()) **2)
        vq_loss = self.beta * vq_loss + commit_loss

        quant = inputs + (quant - inputs).detach()

        avg_probs = torch.mean(encodings,dim=0)
        perp = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        '''
        if self.training:
            if self.dropout:
                quant_perm = quant.permute(0,2,1) #B T C
                quant_perm_4d = quant_perm.unsqueeze(-1)
                
                # drop out along T dimension
                quant_perm_4d = torch.nn.functional.dropout2d(quant_perm_4d, self.dropout_rate, training = self.training)
                
                quant_perm = quant_perm_4d.squeeze(-1)

                quant = quant_perm.permute(0,2,1) # B C T
        '''
        return quant.permute(0,2,1).contiguous(), vq_loss, perp 

class SlicedVectorQuantize4(nn.Module):

    def __init__(self, K, D, beta = 0.25, decay = 0.99, n_d = 4):
        super().__init__()
        self.K = K
        self.D = D
        
        self.sub_D = self.D // n_d
        self.embedding1 = nn.Embedding(K, self.sub_D)
        self.embedding1.weight.data.uniform_(-1. / K, 1. / K)
        
        self.embedding2 = nn.Embedding(self.K, self.sub_D)
        self.embedding2.weight.data.uniform_(-1. / self.K, 1. / self.K)

        self.embedding3 = nn.Embedding(K, self.sub_D)
        self.embedding3.weight.data.uniform_(-1. / K, 1. / K)
        
        self.embedding4 = nn.Embedding(self.K, self.sub_D)
        self.embedding4.weight.data.uniform_(-1. / self.K, 1. / self.K)

        self.beta = beta
        
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

        encoding_ind1 = torch.argmax(-1. * dis1, dim = 1).unsqueeze(1)
        encoding_ind2 = torch.argmax(-1. * dis2, dim = 1).unsqueeze(1)
        encoding_ind3 = torch.argmax(-1. * dis3, dim = 1).unsqueeze(1)
        encoding_ind4 = torch.argmax(-1. * dis4, dim = 1).unsqueeze(1)

        encodings1 = torch.zeros(encoding_ind1.size(0), self.K).type(torch.FloatTensor)
        encodings2 = torch.zeros(encoding_ind2.size(0), self.K).type(torch.FloatTensor)
        encodings3 = torch.zeros(encoding_ind3.size(0), self.K).type(torch.FloatTensor)
        encodings4 = torch.zeros(encoding_ind4.size(0), self.K).type(torch.FloatTensor)

        if torch.cuda.is_available():
            encodings1 = encodings1.cuda()
            encodings2 = encodings2.cuda()
            encodings3 = encodings3.cuda()
            encodings4 = encodings4.cuda()
        
        encodings1.scatter_(1, encoding_ind1, 1)
        encodings2.scatter_(1, encoding_ind2, 1)
        encodings3.scatter_(1, encoding_ind3, 1)
        encodings4.scatter_(1, encoding_ind4, 1)
        
        quant1 = torch.matmul(encodings1, self.embedding1.weight).view(B, T, self.sub_D)
        quant2 = torch.matmul(encodings2, self.embedding2.weight).view(B, T, self.sub_D)
        quant3 = torch.matmul(encodings3, self.embedding3.weight).view(B, T, self.sub_D)
        quant4 = torch.matmul(encodings4, self.embedding4.weight).view(B, T, self.sub_D)
        
        quant = torch.cat([quant1, quant2, quant3, quant4], dim = 2)

        vq_loss =  torch.mean( (quant.detach() - x)**2 )
        
        commit_loss = torch.mean((quant - x.detach()) **2)
        
        vq_loss = vq_loss + self.beta * commit_loss
        
        quant = x + (quant - x).detach()

        avg_probs1 = torch.mean(encodings1, dim=0)
        avg_probs2 = torch.mean(encodings2, dim=0)
        avg_probs3 = torch.mean(encodings3, dim=0)
        avg_probs4 = torch.mean(encodings4, dim=0)

        perp1 = torch.exp( -1. * torch.sum(avg_probs1 * torch.log(avg_probs1 + 1e-10)))
        perp2 = torch.exp( -1. * torch.sum(avg_probs2 * torch.log(avg_probs2 + 1e-10)))
        perp3 = torch.exp( -1. * torch.sum(avg_probs3 * torch.log(avg_probs3 + 1e-10)))
        perp4 = torch.exp( -1. * torch.sum(avg_probs4 * torch.log(avg_probs4 + 1e-10)))
        perp = perp1 + perp2 + perp3 + perp4
        
        return quant.permute(0,2,1), vq_loss, perp
class SlicedVectorQuantize(nn.Module):

    def __init__(self, K, D, beta = 0.25, decay = 0.99, n_d = 2, dropout = False, dropout_rate = 0.25, K1 = None ):
        super().__init__()
        self.K = K
        if K1 is not None:
            self.K1 = K1
        else:
            self.K1 = K
        self.D = D
        
        self.sub_D = self.D // n_d
        self.embedding1 = nn.Embedding(K, self.sub_D)
        self.embedding1.weight.data.uniform_(-1. / K, 1. / K)
        
        self.embedding2 = nn.Embedding(self.K1, self.sub_D)
        self.embedding2.weight.data.uniform_(-1. / self.K1, 1. / self.K1)


        self.decay = decay
        self.beta = beta
        
        self.dropout = dropout
        self.dropout_rate = dropout_rate
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

        encoding_ind1 = torch.argmax(-1. * dis1, dim = 1).unsqueeze(1)
        encoding_ind2 = torch.argmax(-1. * dis2, dim = 1).unsqueeze(1)

        encodings1 = torch.zeros(encoding_ind1.size(0), self.K).type(torch.FloatTensor)
        encodings2 = torch.zeros(encoding_ind2.size(0), self.K1).type(torch.FloatTensor)

        if torch.cuda.is_available():
            encodings1 = encodings1.cuda()
            encodings2 = encodings2.cuda()
        
        encodings1.scatter_(1, encoding_ind1, 1)
        encodings2.scatter_(1, encoding_ind2, 1)
        
        quant1 = torch.matmul(encodings1, self.embedding1.weight).view(B, T, self.sub_D)
        quant2 = torch.matmul(encodings2, self.embedding2.weight).view(B, T, self.sub_D)
        '''       
        if self.training:
            if self.dropout:
                quant_perm1 = quant1.permute(0,2,1) #B T C
                quant_perm_4d1 = quant_perm1.unsqueeze(-1)
                
                # drop out along T dimension
                quant_perm_4d1 = torch.nn.functional.dropout2d(quant_perm_4d1, self.dropout_rate, training = self.training)
                
                quant_perm1 = quant_perm_4d1.squeeze(-1)

                quant1 = quant_perm1.permute(0,2,1) # B C T
                
                quant_perm2 = quant2.permute(0,2,1) #B T C
                quant_perm_4d2 = quant_perm2.unsqueeze(-1)
                
                # drop out along T dimension
                quant_perm_4d2 = torch.nn.functional.dropout2d(quant_perm_4d2, self.dropout_rate, training = self.training)
                
                quant_perm2 = quant_perm_4d2.squeeze(-1)

                quant2 = quant_perm2.permute(0,2,1) # B C T
        
        '''
        quant = torch.cat([quant1, quant2], dim = 2)

        vq_loss =  torch.mean( (quant.detach() - x)**2 )
        
        commit_loss = torch.mean((quant - x.detach()) **2)
        
        vq_loss = vq_loss + self.beta * commit_loss
        
        quant = x + (quant - x).detach()

        avg_probs1 = torch.mean(encodings1, dim=0)
        avg_probs2 = torch.mean(encodings2, dim=0)

        perp1 = torch.exp( -1. * torch.sum(avg_probs1 * torch.log(avg_probs1 + 1e-10)))
        perp2 = torch.exp( -1. * torch.sum(avg_probs2 * torch.log(avg_probs2 + 1e-10)))
        perp = perp1 + perp2
        return quant.permute(0,2,1), vq_loss, perp



class SlicedVectorQuantizeEMA(nn.Module):

    def __init__(self, K, D, beta = 0.25, decay = 0.99, n_d = 2 ):
        super().__init__()
        self.K = K
        self.D = D
        
        self.sub_D = self.D // n_d
        self.embedding1 = nn.Embedding(K, self.sub_D)
        self.embedding1.weight.data.uniform_(-1. / K, 1. / K)
        
        self.embedding2 = nn.Embedding(K, self.sub_D)
        self.embedding2.weight.data.uniform_(-1. / K, 1. / K)


        if self.training:
            self.register_buffer('ema_cluster_size1', torch.zeros(K))
            self.register_buffer('ema_w1', torch.zeros(K, self.sub_D))
            self.register_buffer('ema_cluster_size2', torch.zeros(K))
            self.register_buffer('ema_w2', torch.zeros(K, self.sub_D))
        self.decay = decay
        self.beta = beta


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

        encoding_ind1 = torch.argmax(-1. * dis1, dim = 1).unsqueeze(1)
        encoding_ind2 = torch.argmax(-1. * dis2, dim = 1).unsqueeze(1)

        encodings1 = torch.zeros(encoding_ind1.size(0), self.K).type(torch.FloatTensor)
        encodings2 = torch.zeros(encoding_ind2.size(0), self.K).type(torch.FloatTensor)

        if torch.cuda.is_available:
            encodings1 = encodings1.cuda()
            encodings2 = encodings2.cuda()
        
        encodings1.scatter_(1, encoding_ind1, 1)
        encodings2.scatter_(1, encoding_ind2, 1)

        if self.training:
            with torch.no_grad():
                self.ema_cluster_size1 = self.ema_cluster_size1 * self.decay + (1.0 - self.decay) * torch.sum(encodings1, 0)
                self.ema_cluster_size2 = self.ema_cluster_size2 * self.decay + (1.0 - self.decay) * torch.sum(encodings2, 0)
                
                
                n1 = torch.sum(self.ema_cluster_size1.data)
                n2 = torch.sum(self.ema_cluster_size2.data)
                
                
                self.ema_cluster_size1 =( 
                                          (self.ema_cluster_size1 + 1e-5) 
                                        / (n1 + self.K*1e-5) * n1 )

                self.ema_cluster_size2 =( 
                                          (self.ema_cluster_size2 + 1e-5) 
                                        / (n2 + self.K*1e-5) * n2 )
                
                dw1 = torch.matmul(encodings1.t(), flat_in1)
                dw2 = torch.matmul(encodings2.t(), flat_in2)
                
                
                self.ema_w1 = self.ema_w1 * self.decay + (1 - self.decay) * dw1
                self.ema_w2 = self.ema_w2 * self.decay + (1 - self.decay) * dw2

            
            self.embedding1.weight.data.copy_( self.ema_w1 / self.ema_cluster_size1.unsqueeze(1) )
            self.embedding2.weight.data.copy_( self.ema_w2 / self.ema_cluster_size2.unsqueeze(1) )
        
        quant1 = torch.matmul(encodings1, self.embedding1.weight).view(B, T, self.sub_D)
        quant2 = torch.matmul(encodings2, self.embedding2.weight).view(B, T, self.sub_D)
        
        quant = torch.cat([quant1, quant2], dim = 2)

        vq_loss = self.beta * torch.mean( (quant.detach() - x)**2 )
        
        quant = x + (quant - x).detach()

        avg_probs1 = torch.mean(encodings1, dim=0)
        avg_probs2 = torch.mean(encodings2, dim=0)

        perp1 = torch.exp( -1. * torch.sum(avg_probs1 * torch.log(avg_probs1 + 1e-10)))
        perp2 = torch.exp( -1. * torch.sum(avg_probs2 * torch.log(avg_probs2 + 1e-10)))
        perp = perp1 + perp2
        
        return quant.permute(0,2,1), vq_loss, perp



class VectorQuantizeEMA(nn.Module):

    def __init__(self, K, D, beta = 0.25, decay = 0.99 ):
        super().__init__()
        self.K = K
        self.D = D

        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1. / K, 1. / K)


        if self.training:
            self.register_buffer('ema_cluster_size', torch.zeros(K))
            self.register_buffer('ema_w', torch.zeros(K, D))
        self.decay = decay
        self.beta = beta


    def forward(self, x):
        
        x = x.permute(0,2,1).contiguous()

        in_size = x.size()

        flat_in = x.view(-1,in_size[2])
        
        assert flat_in.size(1) == self.D

        code_sqr = torch.sum(self.embedding.weight **2, dim = 1)

        in_sqr = torch.sum(flat_in**2, dim = 1, keepdim = True)

        dis = torch.addmm(code_sqr + in_sqr, flat_in, self.embedding.weight.t(), alpha = -2.0, beta = 1.0)

        encoding_ind = torch.argmax(-1. * dis, dim = 1).unsqueeze(1)

        encodings = torch.zeros(encoding_ind.size(0), self.K).type(torch.FloatTensor)

        if torch.cuda.is_available:
            encodings = encodings.cuda()
        
        encodings.scatter_(1, encoding_ind, 1)

        if self.training:
            with torch.no_grad():
                self.ema_cluster_size = self.ema_cluster_size * self.decay + (1.0 - self.decay) * torch.sum(encodings, 0)
                n = torch.sum(self.ema_cluster_size.data)
                self.ema_cluster_size =( 
                                          (self.ema_cluster_size + 1e-5) 
                                        / (n + self.K*1e-5) * n )

                dw = torch.matmul(encodings.t(), flat_in)
                
                self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw

            self.embedding.weight.data.copy_( self.ema_w / self.ema_cluster_size.unsqueeze(1) )
        
        quant = torch.matmul(encodings, self.embedding.weight).view(in_size)

        vq_loss = self.beta * torch.mean( (quant.detach() - x)**2 )
        
        quant = x + (quant - x).detach()

        avg_probs = torch.mean(encodings, dim=0)

        perp = torch.exp( -1. * torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quant.permute(0,2,1), vq_loss, perp





class GourpVectorQuantize(nn.Module):
    '''copy from https://github.com/shaojinding/GroupLatentEmbedding/blob/master/layers/vector_quant.py '''
    def __init__(self, n_class, vec_len, num_group ):
        super().__init__()
        
        #norm scale
        self.target_scale = 0.06
        
        self.n_classes = n_classes

        self._num_group = num_group

        if self.n_class % num_group !=0 :
            raise Exception

        self._num_classes_per_group = self.n_classes // self._num_group

        self.embedding0 = nn.Parameter(torch.randn(n_classes, vec_len, requires_grad = True) * self.target_scale)
        
        self.after_update()
        
    def forward(self, x0):
        
        '''
            x0: [B, F, T]
        ''' 
        
        BATCH, CH, T  = x0.size()

        #normalize 

        target_norm = self.target_scale * math.sqrt(x0.size(1))
        
        x = target_norm * x0 / x0.norm(dim = 1, keepdim = True)

        embedding = target_norm * self.embedding0 / self.embedding0.norm(dim = 1, keepdim = True)

        x = x.view(BATCH * T, 1, CH)

        d = x - embedding # B * T , n_classes, vec_len

        d = d.norm(dim = 2) # B * T, n_classes
        
        # group wise distance

        d_group = torch.zeros(BATCH * T, self._num_group)

        if torch.cuda.is_available():
            d_group = d_group.cuda()

        for i in range(self._num_group):
            d_group[:, i] = torch.mean(d[:, i * self._num_group : (i + 1) * self._num_group], 1)

        index_group = d_group.argmin(dim = 1, keepdim = True) # B*T, 1
        
        # generate mask for chosen group

        index_group = index_group.repeat(1, self._num_classes_per_group) # B * T, classes_per_group

        index_group = torch.mul(self._num_classes_per_group, index_group) # B * T, classes_per_group

        idx_mtx = torch.LongTensor( [x for x in range(self._num_classes_per_group)] ).unsqueeze(0) # 1, classes_per_group

        if torch.cuda.is_available():
            idx_mtx = idx_mtx.cuda()


        index_group += id_mtx

        encoding_mask = torch.zeros(BATCH * T, self.n_classes)

        if torch.cuda.is_available():
            encoding_mask = encoding_mask.cuda()

        encoding_mask.scatter(1, index_group, 1)

        encoding_prob = torch.div(1, d)

        masked_encoding_prob = torch.mul(encoding_mask, encoding_prob)
        
        perp = torch.exp( -1* torch.sum(masked_encoding_prob * torch.log(masked_encoding_prob + 1e-10) ) )
        
        quant = torch.matmul(encoding_mask * embedding)
        

