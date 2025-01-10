import torchvision.models as models
from torch.nn import Parameter
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer. Reference https://arxiv.org/abs/1710.10903
    the basic GCN can be modified to this layer in recognition code

    """
    def __init__(self,in_features, out_features, dropout,alpha,batch_size,concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout=dropout
        self.in_features = in_features
        self.out_features = out_features

        self.concat = concat
        self.batch_size=batch_size


        self.weight = Parameter(torch.Tensor(in_features, out_features))

        self.leakyrelu=nn.LeakyReLU(alpha)

        self.hidden = 64

        self.ht = Parameter(torch.Tensor(out_features, self.hidden))

        self.a = Parameter(torch.Tensor(self.hidden * 2, 1))
        self.reset_parameters()




    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data,gain=1.414)
        nn.init.xavier_uniform_(self.a.data,gain=1.414)
        nn.init.xavier_uniform_(self.ht.data, gain=1.414)
    def forward(self,input,adj):
        h=torch.matmul(input,self.weight)

        b=h.size()[0]
        N=h.size()[1] # nodes number
        #print(h.size())
        trans_h = torch.einsum('inc,ck->ink',(h,self.ht))
        #print(trans_h.size())
        a_input=torch.cat([trans_h.repeat(1,1,N).view(b,N*N,-1),trans_h.repeat(1,N,1)],dim=2).view(b,N,-1,2*self.hidden)
        #print(h.repeat(1,1,N).view(b,N*N,-1)[0,:,:])
        #print(h.repeat(1,N,1)[0,:,:])


        #print(a_input.size())
        # x=out*2
        #e = self.leakyrelu(torch.einsum('iwhx,ixc->iwhc',(a_input, self.a)).squeeze(3))
        e= self.leakyrelu(torch.matmul(a_input,self.a).squeeze(3))
        zero_vec=-9e15 * torch.ones_like(e)
        attention=torch.where(adj>0,e,zero_vec)

        attention = F.softmax(attention,dim=2)
        attention = F.dropout(attention,self.dropout,training=self.training)
        h_prime =torch.matmul(attention,h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907

    in_features: input feature dimension
    out_features: output feature dimension

    dropout: default=0 not usedcall nn.dropout function

    bias: the learnable bias in GCN layer, default set as True
    """

    def __init__(self, in_features, out_features,dropout=0., bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)

        if dropout>0.01:
            self.drop_flag = True
            self.dropout_layer = nn.Dropout(p=dropout)
        else:
            self.drop_flag = False

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            #self.bias.data.uniform_(-stdv, stdv)
            nn.init.kaiming_normal_(self.weight, a=0)
            nn.init.constant_(self.bias, 0.0)

    def forward(self, input, adj):
        if self.drop_flag:
            input = self.dropout_layer(input)
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNResnet(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, t=0, adj_file=None):
        super(GCNResnet, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)

        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp):
        feature = self.features(feature)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)


        inp = inp[0]
        adj = gen_adj(self.A).detach()
        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]



def gen_A(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj

def norm_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

def norm_adj_batch(A):
    eye_vec = torch.eye(A.size(1)).cuda()
    #print(A[0])
    #A_x = A.sum(2).float()
    A = A+eye_vec
    A[A<0] = 0
    A[A>2] = 2
    D = torch.pow(A.sum(2).float(), -0.5)
    D = torch.diag_embed(D)
    adj = torch.matmul(A, D).permute(0,2,1)
    adj = torch.matmul(adj, D)
    return adj

