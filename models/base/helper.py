# import new Network name here and add in model_class args
from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
from models.clip import clip
import numpy as np
from kmeans_pytorch import kmeans
import torch.nn as nn
from .Inc_loss import *
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

DEBUG_FLAG = False

def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()



    # import your loss function here

    criterion_ce = nn.CrossEntropyLoss()
    # baseline CE loss

    criterion_arc = nn.CrossEntropyLoss()
    # arc loss for space reservation in the main manuscript.

    
    criterion_clip = nn.CrossEntropyLoss()
    # contrastive learning CLip loss


    ori_mode = model.module.mode
    # standard classification for pretrain

    tqdm_gen = tqdm(trainloader)

    if DEBUG_FLAG:
        torch.autograd.set_detect_anomaly(True)
        pass
        # set gradient operation here


    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]
        ground_truth = torch.arange(data.size(0),dtype=torch.long).cuda()
        optimizer.zero_grad()
        model.module.mode = ori_mode
        ## standard training process
        model.module.mix_flag = False
        #print(train_label)
        logits,feature_img, text_vision,arc_output = model(data,train_label)
        logits_new = logits[:, :args.base_class]
        arc_output = arc_output[:, :args.base_class]
        
        logits_per_image = model.module.measure_imgtxt(feature_img,text_vision)
        
        # using the img to text measurements 
        
        loss_ce = criterion_ce(logits_new, train_label)
        loss_arc = criterion_arc(arc_output, train_label)
        loss_clip = criterion_clip(logits_per_image, ground_truth)
        # load loss function here

        acc = count_acc(logits, train_label)

        total_loss = loss_ce+loss_clip*0+0.1*loss_arc

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f}, loss_clip={:.3f}, acc={:.4f}'.format(epoch, lrc, total_loss.item(), loss_clip.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)
        
        
        
        
        total_loss.backward()
        optimizer.step()
        
        ## new imagined training
        if i%2==1:
            # set the imagined mix contrastive training here 

            model.module.mode = 'generate'
            optimizer.zero_grad()
            model.module.mix_flag = True
            x_gen,label_gen,arcloss = model(data,train_label)
            model.module.mix_flag = False
            #print(label_gen)
            #print(x[0])
            #gen_size = data.size(0)
            #label_gen = torch.arange(gen_size).cuda()
            #print(gen_size)
            #logits = logits[:, :args.base_class]


            # set hyper parameters  use 0.1 as default
            loss =0.1* criterion_ce(x_gen, label_gen)#*2#*epoch/100.0
            pred  = torch.argmax(x_gen,dim=1)
            #print(pred)
            total_loss = loss
    
            lrc = scheduler.get_last_lr()[0]
            tqdm_gen.set_description(
                'Session 0, epo {}, lrc={:.4f},total loss={:.4f}'.format(epoch, lrc, total_loss.item()))
    
            
            loss.backward()
            optimizer.step()
        
    tl = tl.item()
    ta = ta.item()
    model.module.mode = ori_mode
    return tl, ta

def update_textproto(trainset, model, args):
    # replace fc.weight with the embedding average of train data
    model.module.text_dict = trainset.text_dict
    model = model.eval()
    class_num = args.base_class + (args.sessions-1) * args.way
    text_dict = trainset.text_dict
    pre_length = 4
    after_length = 4
    # length parameters are passed by other functions. Options here abandoned.

    if DEBUG_FLAG:
        print(f"a photo of a {text_dict[str(k)]}")
    
    #text_inputs = torch.cat([clip.tokenize(f"a photo of a {text_dict[str(c)]}",pre_length=pre_length,after_length=after_length) for c in range(class_num)]).cuda()
    



    # length parameters are passed by other functions. Options here abandoned.
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {text_dict[str(c)]}") for c in range(class_num)]).cuda()


    proto_list = []
    with torch.no_grad():
        #for class_index in range(class_num):
        text_features = model.module.encode_text(text_inputs)
        #print(text_features.size())

    model.module.proto_txt.data[:class_num] = text_features
    print("update proto using text information!")
    return model

def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    angles_mean = []
    angles_var = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        proto_this = embedding_this.mean(0)
        proto_list.append(proto_this)

        angles_mean_this,angles_var_this = model.module.get_angle_list(embedding_this, proto_this, class_index, verbose=False)
        angles_mean.append(angles_mean_this)
        angles_var.append(angles_var_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.weight.data[:args.base_class] = proto_list

    angles_mean = torch.stack(angles_mean, dim=0)
    angles_var = torch.stack(angles_var, dim=0)

    model.module.angles_mean.data[:args.base_class] = angles_mean
    model.module.angles_var.data[:args.base_class] = angles_var
    ## add inc
    #cluster_proto = replace_base_cluster(embedding_list,label_list,args)
    #model.module.cluster_proto.data[:,:args.base_class,:] = cluster_proto
    return model


def test(model, testloader, epoch, args, session):
    test_class = args.base_class + session * args.way
    VIS_FLAG=False

    model = model.eval()
    vl = Averager()
    va_sum = Averager()
    va_new = Averager()
    va_base = Averager()
    
    #if epoch ==0:
    #    return vl, va_sum, va_base, va_new
    #if epoch %10 >0:
    #    return vl.item(), va_sum.item(), va_base.item(), va_new.item()

    base_classes = np.arange(0, args.base_class, 1)
    new_classes = np.arange(args.base_class,args.base_class + session * args.way, 1)
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)

        cm = np.zeros((100,100))
        # default not used, only for debug mode

        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits,x1,x2,x_loss = model(data)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)

            base_correct_num, base_sum_num = count_acc_class(logits, test_label, base_classes)
            va_base.num_add(base_sum_num)
            va_base.acc_add(base_correct_num)

            new_correct_num, new_sum_num = count_acc_class(logits, test_label, new_classes)
            va_new.num_add(new_sum_num)
            va_new.acc_add(new_correct_num)
            
            if VIS_FLAG and session==2:
                  y_pred = torch.argmax(logits,dim=-1).cpu().numpy()
                  cm = confusion_matrix(test_label.cpu().numpy(), y_pred, labels= np.arange(0,100))+cm
                  # default not used, only for debug mode

            vl.add(loss.item())
            va_sum.add(acc)

        vl = vl.item()
        va_sum = va_sum.item()
        va_base = va_base.calc()
        va_new = va_new.calc()
        if VIS_FLAG and session==2:
            # default not used, only for debug mode
            df_cm = pd.DataFrame(cm[51:71,51:71], index = [i for i in range(51,71)],
            columns = [i for i in range(51,71)])
            plt.figure(figsize = (20,20))
            sn.heatmap(df_cm, annot=False)
            plt.savefig("withoutmixconfusion_matrix"+".png")  

        
        
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va_sum))
    print('epo {}, base_class, new_class={:.4f} acc={:.4f}'.format(epoch, va_base, va_new))
    return vl, va_sum, va_base, va_new


def replace_base_cluster(embedding_list,label_list, args):
    """REPLANCE BASE USING CLUSTER PROTOTYPES, NOT USED."""
    # replace fc.weight with the embedding average of train data
    num_clusters = 5
    # this must be preset! parameter passing failed.
    cluster_list = []
    for class_index in range(args.base_class):
        proto_list = []
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        cluster_ids_x, cluster_centers = kmeans(
        X=embedding_this, num_clusters=num_clusters, distance='cosine', device=torch.device('cuda'))
        # num_clusters * num_features
        
        for i in range(cluster_centers.size(0)):
            sim = F.cosine_similarity(embedding_this, cluster_centers[i])
            sim_sum = torch.sum(sim)
            embedding_sim_i = embedding_this*(sim/sim_sum).unsqueeze(1)
            embedding_out = torch.sum(embedding_sim_i,dim=0)*0.5 + cluster_centers[i]*0.5
            proto_list.append(embedding_out) 
            # cluster_num * num_features
        proto_list = torch.stack(proto_list, dim=0)
        cluster_list.append(proto_list)
    cluster_list = torch.stack(cluster_list, dim=0)
    
    # size = self.cluster_num,self.args.num_classes, self.num_features
    return cluster_list.permute(1,0,2)

