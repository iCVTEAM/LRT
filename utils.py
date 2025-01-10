import random
import torch
import os
import time

import numpy as np
import pprint as pprint

_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def num_add(self, x):
        self.n = self.n+x

    def acc_add(self, x):
        self.v = self.v+x

    def calc(self):
        return self.v/(self.n+0.0001)

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def count_acc_class(logits, label, class_list):
    pred = torch.argmax(logits, dim=1)
    num_acc = 0.0
    num_tot = 0.0
    for i in range(pred.size(0)):
        if (label[i].cpu().numpy() in class_list) and (pred[i]==label[i]):
            num_acc = num_acc + 1
        if (label[i].cpu().numpy() in class_list):
            num_tot = num_tot + 1

    return num_acc, num_tot
    
    #if torch.cuda.is_available():
    #    return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    #else:
    #    return (pred == label).type(torch.FloatTensor).mean().item()



def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)

    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def save_list_to_txt(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(str(item) + '\n')
    f.close()
