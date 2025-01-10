import logging
import os
import sys
import numpy as np
import torchvision.transforms as T
import torch
import torch.nn.functional as F
import random


def build_idx(num):
    """


    Args:
        num: number of graph nodes

    Returns:
        indices of each nodes

    """
    # num denotes the batch size
    idx=torch.linspace(1,num,steps=num)

    for i in range(0,num,2):
        idx[i]=idx[i]+1
    for i in range(1,num+1,2):
        # for odd
        idx[i]=idx[i]-1

    return idx-1

def gen_adj(block_width,batch):
    """

    Args:
        block_width: feature size width/height used for spatial GCN
        batch: batchsize

    Returns:
        adj matrices
    """
    #edge [2,num_edge]
    #edge_W [num_edge,edge_featuredim]
    num_block=block_width*block_width+1
    edge = np.zeros(shape=[num_block, num_block])

    for i in range(block_width):
        for j in range(block_width):
            edge=buildnear(i,j,block_width,edge)

    for k in range(num_block):
        edge[k, num_block-1] = 1
        edge[num_block-1, k] = 1

    # edge=np.expand_dims(edge,axis=0)
    #print(edge)
    edge=torch.from_numpy(edge).cuda().float()


    return edge


def sim_dis(feature1,feature2):
    """cosine simlarity function"""

    dist=torch.cosine_similarity(feature1,feature2,dim=1)
    return dist
def mul_dis(feature1,feature2):
    """matmul similarity function """
    feature1=feature1.view(feature1.shape[0],1,-1)
    feature2=feature2.view(feature2.shape[0],-1,1)
    dist=feature1.matmul(feature2)
    return dist

def gen_adj_sim(part_features1,part_features2,enable_mask=False):
    """
    default similarity function, calculate similarity between two input features

    enable_mask: encourage sparse connections when similarities higher than the threshold

    Args:
        num_block: abandoned feature
        part_features1: vector1, size = b,c,n1
        part_features2: vector2, size = b,c,n2
        enable_mask: True/ False

    Returns:
        similarity matrices: size: n1 * n2

    """


    b,p,c = part_features1.size()
    features = part_features1
    #edge = torch.zeros(b,num_block, num_block)  # 33, 33
    feature2 = part_features2.permute(0,2,1)

    edge =torch.matmul(features,feature2)#.detach()
    #edge = torch.div(edge, float(p*p))
    #edge = torch.mul(torch.sign(edge), torch.sqrt(torch.abs(edge) + 1e-12))

    threshold = random.uniform(0.5, 0.6)

    zeros_vec = torch.zeros_like(edge)
    ones_vec = torch.ones_like(edge)
    eye_vec = torch.eye(p)


    value,idx = edge.max(dim=1)
    value_thre = value.unsqueeze(1)*threshold

    if enable_mask:
        mask = edge-value_thre.repeat(1,p,1)
        mask = torch.where(mask >= 0, ones_vec, zeros_vec)+eye_vec.cuda()
        return edge *mask
    #edge=F.softmax(edge,dim=2)
    #print(mask[0].sum(1))

    else:
        return edge


def Norm_F (feature,norm_flag):
    """

    Args:
        feature: features to be normalized, (b,c,1,n)
        norm_flag: normalization flag: constant, l1norm, l2norm

    Returns:
        normalized feature in the third dimension  size: (b,c,1,n)

    """
    # can be modified here
    W=10
    H=10

    if norm_flag =='constant':
        N_feature = feature*W*H
        return N_feature

    if norm_flag == 'l1norm':
        s1 = torch.norm(feature,p=1,dim=3)
        return s1

    if norm_flag == 'l2norm':
        s2 = torch.norm(feature,p=2,dim=3)
        return s2

    assert ("unimplemeted error found")

def get_sim_local(edge1,edge2):
    """

    get local connections
    Args:
        edge1:
        edge2:

    Returns:

    """
    b, c1, c1 = edge1.size()
    b, c2, c2 = edge2.size()
    edge_concat = torch.zeros(b,c1+c2,c1+c2)
    edge_concat[:,:c1,:c1] = edge1
    edge_concat[:,c1:, c1:] = edge2
    return edge_concat


def get_sim_cross2(edge):
    """handcrafted similarity matrices """
    b, num_block, num_block = edge.size()
    mask = torch.zeros(b,num_block,num_block)
    for i in range(64):
        mask[:,i, i] = 1
        mask[:,i + 64, i + 64] = 1

        mask[:,i,i+64] = 1
        mask[:,i+64,i] = 1

    new_edge=mask.cuda()#*edge
    return new_edge

def get_sim_cross(edge):
    """
    handcrafted similarity matrices

    Args:
        edge:

    Returns:

    """
    b, num_block, num_block = edge.size()
    mask = torch.zeros(b,num_block,num_block)
    for i in range(64):
        mask[:,i, i] = 1
        mask[:,i + 64, i + 64] = 1
        mask[:,i + 64*2, i + 64*2] = 1
        mask[:,i,i+64] = 1
        mask[:,i+64,i] = 1
        mask[:,i, i + 64*2] = 1
        mask[:,i + 64*2, i] = 1
        mask[:,i+64, i + 64 * 2] = 1
        mask[:,i + 64 * 2, i+64] = 1
    new_edge=mask.cuda()*edge
    return new_edge


def gen_adj_topk(num_block, part_features):

    num_k=5

    b, p, c = part_features.size()
    features = part_features.clone()
    # edge = torch.zeros(b,num_block, num_block)  # 33, 33
    feature2 = features.permute(0, 2, 1)

    edge = torch.matmul(features, feature2)
    edge = torch.div(edge, float(p * p))
    edge = torch.mul(torch.sign(edge), torch.sqrt(torch.abs(edge) + 1e-12))
    # print(edge)

    # edge=F.softmax(edge,dim=2)
    values, idx = edge.topk(num_k, dim=2)
    thre = values[:,:,4].view(b,num_block,1)
    sim =torch.sub(edge ,thre.repeat(1,1,num_block))

    e= edge.clone()
    e_zero=torch.zeros_like(sim)
    edge_new = torch.where(sim >= 0, e, e_zero)
    #print(edge)
    #print(edge_new)
    return edge_new



def gen_adj_sim_old(num_block,part_features):

    b,p,c =part_features.size()
    part_features = part_features.permute(0,2,1)
    #print(part_features.size())
    edge = torch.zeros(b,num_block, num_block)  # 33, 33
    #edge = torch.from_numpy(edge).cuda().float()
    thre = torch.tensor([1]).cuda()
    for i in range(num_block):
        feature1 = part_features[:, :, i]

        for j in range(i+1,num_block):
            feature2 = part_features[:, :, j]
            dist=mul_dis(feature1,feature2)
            #print(dist)
            edge[:, i, j] = dist.view(b) #torch.mul(torch.sign(dist),dist)
            edge[:, j, i] =  edge[:, i, j]
    # print(edge[1, :, :])
    edge = F.softmax(edge,dim=2)
    #print(edge[1, :, :])
    for k in range(num_block):
        edge[:, k, k] = 0.5
        #edge[:,k, num_block-1] = 1+edge[:, k, num_block-1]
        #edge[:, num_block-1, k] = 1+edge[:, num_block-1, k]
    #edge[:, num_block-1, num_block-1] = 1
    #print(edge[1,:,:])
    return edge


def gen_adj_nearst(num_block,part_features):
    """
    enable nearst connections for the graph embedding
    Args:
        num_block:
        part_features:

    Returns:

    """
    b,c,p =part_features.size()

    num_k=5
    edge = np.zeros(shape=[b,num_block, num_block])  # 33, 33
    edge = torch.from_numpy(edge).cuda().float()

    sim = np.zeros(shape=[b,num_block, num_block])  # 33, 33
    sim = torch.from_numpy(sim).cuda().float()

    for i in range(num_block):
        feature1 = part_features[:, :, i]

        for j in range(i+1,num_block):
            feature2 = part_features[:, :, j]
            dist=sim_dis(feature1,feature2)
            sim[:, i, j] = torch.mul(torch.sign(dist),dist)
            sim[:, j, i] =  sim[:, i, j]

    values, idx = sim.topk(num_k, dim=2)


    thre = values[:,:,4].view(b,num_block,1)

    sim =torch.sub(sim ,thre.repeat(1,1,num_block))
    #sim=torch.sign(sim)
    e= torch.ones_like(sim)
    e_zero=torch.zeros_like(sim)
    edge = torch.where(sim >= 0, e, e_zero)
    #print(edge.sum(dim=2))

    for k in range(num_block-1):
        edge[:, k, k] = 1
        edge[:,k, num_block-1] = 1+edge[:, k, num_block-1]
        edge[:, num_block-1, k] = 1+edge[:, num_block-1, k]
    edge[:, num_block-1, num_block-1] = 1
    #print(edge[1,:,:])
    return edge




def gen_adj2(num_block):
    #edge [2,num_edge]
    #edge_W [num_edge,edge_featuredim]
    rep_field=2
    edge = np.zeros(shape=[num_block, num_block]) # 33, 33



    for k in range(num_block):
        edge[k, k] = 1
        for r in range(-rep_field,rep_field+1):
            if k+r>0 and k+r<num_block and r!=0:
                edge[k, k+r] = 0.5
                edge[k+r,k] = 0.5
        #edge[num_block-1, k] = 1

    # edge=np.expand_dims(edge,axis=0)
    #print(edge)
    edge=torch.from_numpy(edge).cuda().float()


    return edge


def gen_adj_coo(block_width,batch):
    """generate adjacent matrices by COO form"""


    num_block=block_width*block_width
    edge=np.zeros(shape=[block_width,block_width])

    edge_w=np.empty(shape=[0,1],dtype=np.long)

    for i in range(block_width):
        for j in range(block_width):
            edge,edge_w=buildnear(i,j,block_width,edge)

    # edge=np.expand_dims(edge,axis=0)
    # edge_w = np.expand_dims(edge_w, axis=0)
    #print(edge)
    #print(edge_w.shape)
    edge=torch.from_numpy(edge).long()
    edge_w=torch.from_numpy(edge_w).long()
    edge_w=edge_w.squeeze()
    #print(edge.type())
    # edge.repeat(batch,1,1)
    # edge_w.repeat(batch, 1, 1)

    return edge,edge_w

def buildnear(i,j,len,edge):

    for x0 in range(-1,2):
        for y0 in range(-1,2):
            x=i+x0
            y=j+y0
            if x>=0 and y>=0 and x<len and y<len :
                edge[i*len+j,x*len+y]=1


    return edge