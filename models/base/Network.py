import argparse
from utils import *
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from models.clip import clip
from pkg_resources import packaging
from .decoder import ProtoFusion
import numpy as np
from kmeans_pytorch import kmeans
from ..utils.adj_matrix import *
from .GCN import norm_adj_batch as norm_adj, GraphConvolution as GCNConv, GraphAttentionLayer as  GATConv
from .Inc_loss import ArcMarginProduct

class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()
        self.scale_mm = nn.Parameter(torch.ones(1),
                                  requires_grad=True)
        self.mode = mode
        self.args = args
        # self.num_features = 512
        if self.args.dataset in ['cifar100']:
            self.encoder = resnet20()
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet','imagenet100']:
            #self.encoder = resnet18(False, args)  # pretrained=False
            self.num_features = 1024
            self.clip_model,_ = clip.load("RN50", device ='cpu') 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cluster_num = 5

        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        self.base_proto = nn.Parameter(torch.randn(self.args.num_classes, self.num_features),
                                  requires_grad=True)
        self.cluster_proto = nn.Parameter(torch.randn(self.cluster_num,self.args.num_classes, self.num_features),
                                  requires_grad=True) # params not used 

        # new prompt parameters  
        # set your own length here
        # set your own length here
        # Prompt parameters setup
        self.pre_length = 4  # Length before text prompt
        self.after_length = 4  # Length after text prompt
        self.vision_length = 4  # Vision length options not used 
        self.vision_cluster = 10  # Vision cluster size. options not used 
        
        self.mix_flag = False  # Flag for mixing images only flags not need preset


        self.gamma = nn.Parameter(torch.ones(1)*0.1, requires_grad=True)  # Learnable gamma parameter

        
        self.arc_loss = ArcMarginProduct()
        # start gradient should be tried
        self.angles_mean = nn.Parameter(torch.zeros(self.args.num_classes), requires_grad=False)
        self.angles_var = nn.Parameter(torch.zeros(self.args.num_classes), requires_grad=False)

        self.dtype = self.clip_model.dtype
        self.transformer_dim = self.clip_model.transformer_width#token_embedding.weight.size(0)
        
        #self.reproj = nn.ModuleList(ProtoFusion(self.num_features, self.transformer_dim) for i in range(self.pre_length+self.after_length) )
        if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
            self.prompt_text = nn.Parameter(torch.zeros(self.args.num_classes, self.pre_length+self.after_length,self.transformer_dim),
                                      requires_grad=True) 
            self.cluster_prompt_text = nn.Parameter(torch.zeros(self.vision_cluster*self.args.base_class,self.transformer_dim),
                                      requires_grad=True) 
        else:
            self.prompt_text = nn.Parameter(torch.zeros(self.args.num_classes, self.pre_length+self.after_length,self.transformer_dim),
                                      requires_grad=True) 
            self.cluster_prompt_text = nn.Parameter(torch.zeros(self.vision_cluster*self.args.base_class,self.transformer_dim),
                                      requires_grad=True) 
        #My initialization                               
        #nn.init.uniform_(self.prompt_text)
        self.proj_prompt = nn.Linear(self.num_features, self.vision_cluster*self.args.base_class, bias=False)

        
        self.softmax = nn.Softmax(dim=1)
        self.text_dict=[]   
        self.dropout =0.2
        self.graphconv = GCNConv(self.num_features,
                                  self.num_features,dropout=self.dropout)

        self.relu = nn.LeakyReLU(0.2)
      
        self.elu = nn.ELU()
        self.bn = nn.BatchNorm1d(self.num_features)
        
        
        self.fusion = 'norm'


        # mlp only usded for proto fusions
        if self.fusion== 'proto':
            self.mlp = nn.Sequential(
            nn.Linear(self.num_features*2, self.num_features),
            nn.GELU(),
            nn.Linear(self.num_features, self.num_features)
        )


    def selective_fusion(self, img_proto, text_proto):

        # fusion strategy defined here 

        if self.fusion == 'proto':
            concat_proto = torch.cat([img_proto[0,:,:],text_proto[0,:,:]],dim=1)
            proto_fuse = self.mlp(concat_proto)
        if self.fusion == 'add':
            proto_fuse = img_proto[0,:,:]+text_proto[0,:,:]
        return proto_fuse
    
    def norm_std(self, x):
        # mean x not instance-depedent  x: batchsize * num_classes    
        mean_x = x.mean(dim=-1).unsqueeze(1)
        std_x = x.std(dim=-1).unsqueeze(1)
        norm_x = (x-mean_x)/std_x
        return norm_x
    
    def reproj_func(self, prototype):
        
        """ Reproject prototype into prompt text space """

        prompt_text_list = []
        for i in range(self.pre_length+self.after_length):
            prompt_text_this = self.reproj[i](prototype).unsqueeze(1)
            prompt_text_list.append(prompt_text_this)
        prompt_text_list = torch.stack(prompt_text_list, dim=1)
        return prompt_text_list
    
    def forward_metric(self, x, label=None ):

        MAX_FLAG = True
        if  not self.mix_flag:
            x = self.encode_img(x)
            x = x.squeeze(-1).squeeze(-1)
            text_features = self.encode_text_prompt()
            #1) option text_vision vs visual_features
            #2) option text_vision +x vs text_features/fc weights
            clone_x = x.detach().clone()
            text_vision = x
            #x = self.bn(x)
            text_features = F.normalize(text_features, p=2, dim=-1)
            imgproto_features = F.normalize(self.fc.weight, p=2, dim=-1)
            

            
            text_features = text_features.repeat(x.size(0),1,1)
            imgproto_features = imgproto_features.repeat(x.size(0),1,1)
            
            # construct the adj matrixes and construct the graph edges using learnt text features
            edge_text = gen_adj_sim(text_features, text_features)
            adj = edge_text.cuda()
            adj = norm_adj(adj)
            img_gfeature = self.graphconv(imgproto_features, adj)
            img_gfeature = self.relu(img_gfeature)
            # batch*batch

            # fusion types, users can modified this for the final visual and text prototypes
            # fusion features

            if self.fusion == 'proto' or self.fusion == 'add':
                fused_proto = self.selective_fusion(img_gfeature,text_features)
                
                x_fuse = torch.einsum('bc,nc->bn',F.normalize(x, p=2, dim=-1),fused_proto)
                arc_output = self.arc_loss(x_fuse, label)
                return x_fuse,clone_x,text_vision,arc_output
            x_img = torch.einsum('bc,bnc->bn',F.normalize(x, p=2, dim=-1),img_gfeature)
            #x_img = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            # generating txt proto
            x_txt = torch.einsum('bc,bnc->bn',F.normalize(x, p=2, dim=-1),text_features)
            #x_txt = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(text_features, p=2, dim=-1))
            x_txt =  self.scale_mm * x_txt   
            #x_txt = self.norm_std(x_txt)
            x_fuse = x_img+x_txt
            arc_output = self.arc_loss(x_fuse, label)
            #print(x_fuse[0])
            return x_fuse,clone_x,text_vision,arc_output
        elif not label is None and self.mix_flag :
            lamda = torch.rand(1)*0.2+0.4
            rand_index = torch.randperm(x.size()[0]).cuda()
            label_a = label
            label_b = label[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lamda)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))


            x = self.encode_img(x)
            x = x.squeeze(-1).squeeze(-1)
            text_vision = self.encode_text_prompt_vision(x)
            x = self.gamma*text_vision+x
            #x = self.bn(x)
            
            text_features_gen = self.encode_text_prompt_label(label_a,label_b)
            text_features_class = self.encode_text_prompt()
            gen_size = x.size(0)
            label_gen = torch.arange(gen_size).cuda()
            #print(label_gen)
            class_size = text_features_class.size(0)

            text_features = torch.cat([text_features_gen,text_features_class], dim=0)

            # generating txt proto
            x_txt = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(text_features, p=2, dim=-1))
            x_txt =  self.scale_mm * x_txt   
            #x_txt = self.norm_std(x_txt)
            x = x_txt
            #print(x)
            arc_loss= x #self.arc_loss(x, label)
            return x,label_gen,arc_loss
    def rand_bbox(self, size, lam):

        """ Generate random bounding box for Imagined Contrastive Learning of the visual branches """

        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
    
        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
    
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
    
        return bbx1, bby1, bbx2, bby2

    def encode_text_prompt_label(self,label_a,label_b):

        """ Encode text prompts for given labels """

        MAXN = 66666666
        # replace fc.weight with the embedding average of train data
        if len(self.text_dict)!=self.args.num_classes:
            assert False 
        else:
            text_dict = self.text_dict

        class_num = self.args.num_classes
        text_inputs_a = torch.cat([clip.tokenize(f"{text_dict[str(c.cpu().numpy())]}",pre_length=self.pre_length,after_length=self.after_length)[0] for c in label_a]).cuda()
        token_len_a = torch.cat([clip.tokenize(f"{text_dict[str(c.cpu().numpy())]}",pre_length=self.pre_length,after_length=self.after_length)[1] for c in label_a]).cuda()
        text_inputs_b = torch.cat([clip.tokenize(f"{text_dict[str(c.cpu().numpy())]}",pre_length=self.pre_length,after_length=self.after_length)[0] for c in label_b]).cuda()
        token_len_b = torch.cat([clip.tokenize(f"{text_dict[str(c.cpu().numpy())]}",pre_length=self.pre_length,after_length=self.after_length)[1] for c in label_b]).cuda()

        prompt_list = []
        text_inputs_a = text_inputs_a.type(torch.long)  # num_classes * vocab size (100 * 77)
        text_inputs_b = text_inputs_b.type(torch.long)  # num_classes * vocab size (100 * 77)
        text_inputs = text_inputs_a.clone() 
        text_embedding_a = self.clip_model.token_embedding(text_inputs_a).type(self.dtype)  # num_classes * vocab size* transformer_dim (100 * 77 *512)
        text_embedding_b = self.clip_model.token_embedding(text_inputs_b).type(self.dtype)  # num_classes * vocab size* transformer_dim (100 * 77 *512)

        list_size = label_a.size(0)
        vocab_size = 77
        for i in range(list_size):
            l_a = label_a[i]
            l_b = label_b[i]
            left_length = vocab_size - token_len_a[i]-token_len_b[i]+2
            text_inputs[i,token_len_a[i]+token_len_b[i]-2] = MAXN
            prompt_this = torch.cat([text_embedding_a[i,0:1,:],  # shared
                        self.prompt_text[l_a,:self.pre_length,:],
                        text_embedding_a[i,1+self.pre_length:token_len_a[i]-1-self.after_length,:],
                        self.prompt_text[l_a,self.pre_length:,:],
                        self.prompt_text[l_b,:self.pre_length,:],
                        text_embedding_b[i,1+self.pre_length:token_len_b[i]-1-self.after_length,:],
                        self.prompt_text[l_b,self.pre_length:,:],
                        text_embedding_a[i,token_len_a[i]-1:token_len_a[i],:], # shared
                        text_embedding_a[i,vocab_size-left_length:vocab_size,:], # zeros
                        ],dim=0)
        # wrong length size    
            prompt_list.append(prompt_this)
            
        
        prompt_text = torch.stack(prompt_list, dim=0).type(self.dtype) #num_classes * vocab size* transformer_dim
        #print(prompt_text[0,1:10,1:10])
        text_features = self.encode_text(prompt_text,text_inputs) #num_class,1024


        return text_features

    def encode_text_prompt_vision(self,x, train_label=None):
        # replace fc.weight with the embedding average of train data
        if len(self.text_dict)!=self.args.num_classes:
            assert False 
        else:
            text_dict = self.text_dict
        class_num = self.args.base_class

        batchsize = x.size(0)

        selection_weight = self.proj_prompt(x.clone().detach())
        selection_weight = self.softmax(selection_weight) # batchsize, pool_size
        k_values,k_idxs = torch.topk(selection_weight,self.vision_length,dim=1) # batchsize,k

        

        text_inputs = torch.cat([clip.tokenize(f"",pre_length=self.vision_length,after_length=0)[0] for c in range(batchsize)]).cuda()
        token_len = torch.cat([clip.tokenize(f"",pre_length=self.vision_length,after_length=0)[1] for c in range(batchsize)]).cuda()
        prompt_list = []
        text_inputs = text_inputs.type(torch.long)  # num_classes * vocab size (100 * 77)
        
        text_embedding = self.clip_model.token_embedding(text_inputs).type(self.dtype)  # num_classes * vocab size* transformer_dim (100 * 77 *512)

        for i in range(batchsize):
            if train_label ==None:
                select_prompt_this = self.cluster_prompt_text[k_idxs[i],:]
            else:
                select_prompt_all = self.cluster_prompt_text[train_label[i]*self.vision_cluster:(train_label[i]+1)*self.vision_cluster,:]
                rand_idx = torch.randperm(self.vision_cluster)
                idx_pre = rand_idx[:self.vision_length]
                select_prompt_this = select_prompt_all[idx_pre,:]

            prompt_this = torch.cat([text_embedding[i,0:1,:],
                        select_prompt_this[:,:],
                        text_embedding[i,1+self.vision_length:token_len[i]-1,:], # this should be empty
                        text_embedding[i,token_len[i]-1:,:],
                        ],dim=0)
            
            prompt_list.append(prompt_this)
            
        
        prompt_text = torch.stack(prompt_list, dim=0).type(self.dtype) #num_classes * vocab size* transformer_dim
        #print(prompt_text[0,1:10,1:10])
        text_features_vision = self.encode_text(prompt_text,text_inputs) #self.vision_cluster*class_num,1024




        return text_features_vision

    def encode_text_prompt(self):
        # replace fc.weight with the embedding average of train data
        if len(self.text_dict)!=self.args.num_classes:
            assert False 
        else:
            text_dict = self.text_dict
        class_num = self.args.num_classes

        text_inputs = torch.cat([clip.tokenize(f"{text_dict[str(c)]}",pre_length=self.pre_length,after_length=self.after_length)[0] for c in range(class_num)]).cuda()
        token_len = torch.cat([clip.tokenize(f"{text_dict[str(c)]}",pre_length=self.pre_length,after_length=self.after_length)[1] for c in range(class_num)]).cuda()
        prompt_list = []
        text_inputs = text_inputs.type(torch.long)  # num_classes * vocab size (100 * 77)
        
        text_embedding = self.clip_model.token_embedding(text_inputs).type(self.dtype)  # num_classes * vocab size* transformer_dim (100 * 77 *512)

        for i in range(self.args.num_classes):
            prompt_this = torch.cat([text_embedding[i,0:1,:],
                        self.prompt_text[i,:self.pre_length,:],
                        text_embedding[i,1+self.pre_length:token_len[i]-1-self.after_length,:],
                        self.prompt_text[i,self.pre_length:,:],
                        text_embedding[i,token_len[i]-1:,:],
                        ],dim=0)
            
            prompt_list.append(prompt_this)
            
        
        prompt_text = torch.stack(prompt_list, dim=0).type(self.dtype) #num_classes * vocab size* transformer_dim
        #print(prompt_text[0,1:10,1:10])
        text_features = self.encode_text(prompt_text,text_inputs) #num_class,1024
        return text_features
        

    

    def encode_img(self, x):
        x = nn.functional.interpolate(x, size=[224,224], mode='bilinear', align_corners=False) 
        x = self.clip_model.encode_image(x)
        #x = F.adaptive_avg_pool2d(x, 1)
        #x = x.squeeze(-1).squeeze(-1)
        return x
    def encode_text(self, prompt_text,text_inputs):
        x = self.clip_model.encode_text_prompt(prompt_text,text_inputs)
        return x    
    def measure_imgtxt(self, img_features,txt_features):
        #txt_features: size of classes * num_of_features 
        img_features_norm =img_features/ img_features.norm(dim=-1, keepdim=True)
        txt_features_norm =txt_features/ txt_features.norm(dim=-1, keepdim=True)
        scale_mm = 1.0
        similarity = (scale_mm* img_features_norm @ txt_features_norm.T)#.softmax(dim=-1)
        return similarity

    def forward(self, x, label=None):
        if self.mode != 'encoder' and self.mode != 'generate':
            x = self.forward_metric(x,label)
            return x
        elif self.mode == 'generate':
            x,label_gen,arc = self.forward_metric(x,label)
            return x,label_gen,arc
        elif self.mode == 'encoder':
            x = self.encode_img(x)
            return x
        else:
            raise ValueError('Unknown mode')

    def update_fc_abandon(self,dataloader,class_list,session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode_img(data).detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        if 'ft' in self.args.new_mode:  # further finetune
            self.update_fc_ft(new_fc,data,label,session)

    def update_fc_avg(self,data,label,class_list):
        new_fc=[]
        angles_mean = []
        angles_var = []
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto

            # calc angles
            angles_mean_this, angles_var_this = self.get_angle_list(embedding, proto, class_index, verbose=False)
            angles_mean.append(angles_mean_this)
            angles_var.append(angles_var_this)

        angles_var=torch.stack(angles_var,dim=0)
        angles_mean=torch.stack(angles_mean,dim=0)
        self.angles_mean.data[class_list] = angles_mean
        self.angles_var.data[class_list] = angles_var
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    def update_cluster_avg(self,data,label,class_list):
        new_cluster=[]
        for class_index in class_list:
            proto_list = []
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            for i in range(embedding.size(0)):
                proto=embedding[i,:]
                sim = F.cosine_similarity(embedding, proto)
                sim_sum = torch.sum(sim)
                embedding_sim_i = embedding*(sim/sim_sum).unsqueeze(1)
                embedding_out = torch.sum(embedding_sim_i,dim=0)*0.9 + proto*0.1
                proto_list.append(embedding_out) 
            proto_list=torch.stack(proto_list,dim=0)
            new_cluster.append(proto_list)
        new_cluster=torch.stack(new_cluster,dim=0)
        return new_cluster.permute(1,0,2)

    def get_logits(self,x,fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x,fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))




#### re-implement multiple
    def update_fc(self,dataloader, transform2,class_list,session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode_img(data).detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        
        # should do sth before the final session
        if session!=0:
            new_cluster = self.update_cluster_avg(data, label, class_list)

        if 'ft' in self.args.new_mode:  # further finetune
            dataloader.dataset.transform = transform2
            self.update_fc_ft(new_fc,new_cluster,dataloader,label,session)
            
      

        
    def encode_text_prompt_finetune(self, prompt_new,class_num):
        # replace fc.weight with the embedding average of train data

        if len(self.text_dict)!=self.args.num_classes:
            assert False 
        else:
            text_dict = self.text_dict
        class_num = class_num #current clas_num
        text_inputs = torch.cat([clip.tokenize(f"{text_dict[str(c)]}",pre_length=self.pre_length,after_length=self.after_length)[0] for c in range(class_num)]).cuda()
        token_len = torch.cat([clip.tokenize(f"{text_dict[str(c)]}",pre_length=self.pre_length,after_length=self.after_length)[1] for c in range(class_num)]).cuda()
        prompt_list = []
        text_inputs = text_inputs.type(torch.long)  # num_classes * vocab size (100 * 77)
        
        text_embedding = self.clip_model.token_embedding(text_inputs).type(self.dtype)  # num_classes * vocab size* transformer_dim (100 * 77 *512)

        for i in range(class_num):
            prompt_this = torch.cat([text_embedding[i,0:1,:],
                        prompt_new[i,:self.pre_length,:],
                        text_embedding[i,1+self.pre_length:token_len[i]-1-self.after_length,:],
                        prompt_new[i,self.pre_length:,:],
                        text_embedding[i,token_len[i]-1:,:],
                        ],dim=0)
            
            prompt_list.append(prompt_this)

        
        prompt_text = torch.stack(prompt_list, dim=0).type(self.dtype) #num_classes * vocab size* transformer_dim
        #print(prompt_text[0,1:10,1:10])
        text_features = self.encode_text(prompt_text,text_inputs) #num_class,1024

        return text_features    

    def encode_text_prompt_cluster_finetune(self, cluster_prompt_new,class_num):
        # replace fc.weight with the embedding average of train data
        if len(self.text_dict)!=self.args.num_classes:
            assert False 
        else:
            text_dict = self.text_dict
        

        text_inputs = torch.cat([clip.tokenize(f"{text_dict[str(c)]}",pre_length=self.pre_length,after_length=self.after_length)[0] for c in range(class_num)]).cuda()
        token_len = torch.cat([clip.tokenize(f"{text_dict[str(c)]}",pre_length=self.pre_length,after_length=self.after_length)[1] for c in range(class_num)]).cuda()
        
        text_inputs = text_inputs.type(torch.long)  # num_classes * vocab size (100 * 77)
        
        text_embedding = self.clip_model.token_embedding(text_inputs).type(self.dtype)  # num_classes * vocab size* transformer_dim (100 * 77 *512)

        cluster_text = []
        for cl in range(self.cluster_num):
            prompt_list = []
            for i in range(class_num):
                prompt_this = torch.cat([text_embedding[i,0:1,:],
                            cluster_prompt_new[cl,i,:self.pre_length,:],
                            text_embedding[i,1+self.pre_length:token_len[i]-1-self.after_length,:],
                            cluster_prompt_new[cl,i,self.pre_length:,:],
                            text_embedding[i,token_len[i]-1:,:],
                            ],dim=0)
                
                prompt_list.append(prompt_this)
            
        
            prompt_text = torch.stack(prompt_list, dim=0).type(self.dtype) #num_classes * vocab size* transformer_dim
            #print(prompt_text[0,1:10,1:10])
            
            cluster_text.append(prompt_text)
        cluster_text = torch.stack(cluster_text, dim=0).type(self.dtype)
        cluster_text = cluster_text.view(self.cluster_num,class_num,77,-1)
        with torch.no_grad():
            tmp = cluster_text[:,:class_num-self.args.way,:,:]
            #print(tmp.size())
            old_text  = tmp.contiguous().view(self.cluster_num*(class_num-self.args.way),cluster_text.size(2),cluster_text.size(3))
            old_text_features = self.encode_text(old_text,text_inputs[:class_num-self.args.way,:].repeat(self.cluster_num,1)) #num_class,1024
            old_text_features = old_text_features.view(self.cluster_num,(class_num-self.args.way),-1)
        with torch.enable_grad():
            tmp = cluster_text[:,class_num-self.args.way:class_num,:,:]
            new_text  = tmp.contiguous().view(self.cluster_num*self.args.way,cluster_text.size(2),cluster_text.size(3))
            new_text_features = self.encode_text(new_text,text_inputs[class_num-self.args.way:,:].repeat(self.cluster_num,1)) 
            new_text_features = new_text_features.view(self.cluster_num,self.args.way,-1)
        #cat_text = torch.cat([ori_text,cluster_text],dim=0)
        text_features = torch.cat([old_text_features,new_text_features],dim=1)
        text_features = text_features.view(self.cluster_num,class_num,-1)
        return text_features# self.cluster_num, num_class, 1024

    def get_angle_list(self, features, protos, class_id, verbose=False):
        #features batch_list * C
        # proto N*C
        #angles = []
        features = F.normalize(features, p=2, dim=-1)
        protos = F.normalize(self.fc.weight[class_id,:], p=2, dim=-1)
        
        angles_this = torch.einsum('bc,c->b',features.cuda(),protos)
        cos_this = self.arccosine(angles_this)
        #angles.append(cos_this)

        angles = cos_this#torch.stack(angles,dim=0) # class * b
        angles_mean = torch.mean(angles,dim=0)
        angles_var = torch.var(angles,dim=0)
        # calc mean and var


        return angles_mean, angles_var

    def arccosine(self, x):
        maxn = 1-1e-6
        minn = -1+1e-6  
        x[x>maxn] = maxn
        x[x<minn] = minn
        return torch.arccos(x)



    def joint_finetune(self,x,fc,cluster_proto,text_prompt,cluster_prompt_text,label):

        x = self.encode_img(x)
        x = x.squeeze(-1).squeeze(-1)
        # x is encoded data
        MAX_FLAG = True
        class_num = fc.size(0)
        text_features = self.encode_text_prompt_finetune(text_prompt,class_num)
        text_vision = self.encode_text_prompt_vision(x)
        #x = self.gamma*text_vision+x
        #x = self.bn(x)
        var_list = []
        for idx in range(label.size(0)):
            var_mean = self.angles_var[:self.args.base_class].mean()
            var = self.angles_var[label[idx]] - var_mean
            std = torch.sqrt(torch.abs(var))
            var_list.append(std)
        var_list = torch.stack(var_list,dim=0)
        sample_angles = torch.normal(0.0,var_list)
        if 'cos' in self.mode:
            output_x = []
            #sim_cons = []
            text_features = F.normalize(text_features, p=2, dim=-1)
            
            
            imgproto_features = F.normalize(self.fc.weight[:class_num,:], p=2, dim=-1)
            text_features = text_features.repeat(x.size(0),1,1)
            imgproto_features = imgproto_features.repeat(x.size(0),1,1)
            edge_text = gen_adj_sim(text_features, text_features)
            adj = edge_text.cuda()
            adj = norm_adj(adj)
            #print(adj[0])
            img_gfeature = self.graphconv(imgproto_features, adj)
            img_gfeature = self.relu(img_gfeature)
            # batch*batch

            x_img = torch.einsum('bc,bnc->bn',F.normalize(x, p=2, dim=-1),img_gfeature)
            #x_img = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            # generating txt proto
            x_txt = torch.einsum('bc,bnc->bn',F.normalize(x, p=2, dim=-1),text_features)
            #x_txt = self.norm_std(x_txt)
            x_fuse = self.scale_mm * x_txt+x_img

            arc_output = self.arc_loss(x_fuse, label, sample_angles=sample_angles.clone().detach())
            
            #loss_con = (x_img-x_txt)**2

            
            #1) option text_vision vs text_features
            #2) option text_vision +x vs text_features/fc weights
            
            #loss_con = loss_con/(loss_con.size(1))
            return x_fuse,arc_output
            
        elif 'dot' in self.mode:
            x = self.fc(x)

        return x



    def update_fc_ft(self,new_fc,new_cluster,dataloader,label,session):
        torch.autograd.set_detect_anomaly(True)
        learn_prompt = self.prompt_text[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * (session),:,:].clone().detach()
        learn_prompt.requires_grad=True

        #learn_prompt_cluster = self.cluster_prompt_text[:,self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * (session),:,:].clone().detach()
        #learn_prompt.requires_grad=True

        new_cluster = new_cluster.clone().detach()
        new_cluster.requires_grad=True
        new_fc=new_fc.clone().detach()
        #new_fc.requires_grad=True


        weight = nn.Parameter(torch.rand(self.args.base_class, self.args.way,device='cuda'),requires_grad=True)        
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        #weight = weight.clone().detach()
        weight.requires_grad=True
        

        optimized_parameters = [{'params': weight},{'params': learn_prompt}] #,{'params': learn_prompt_cluster}
        #optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=0.01)
        optimizer = torch.optim.Adam(optimized_parameters,lr=self.args.lr_new, weight_decay=0.0001)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma = 0.8)


        with torch.enable_grad():
            for epoch in tqdm(range(self.args.epochs_new)):
                #scheduler.step()
                for batch in dataloader:
                    data, label = [_.cuda() for _ in batch]
                    optimizer.zero_grad()
                    old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].clone().detach()
                    old_cluster = self.cluster_proto[:,:self.args.base_class + self.args.way * (session - 1), :].clone().detach()
                    fc = torch.cat([old_fc, new_fc], dim=0)
                    cluster_proto = torch.cat([old_cluster, new_cluster], dim=1)
                        
                    base_prompt = self.prompt_text[:self.args.base_class, :,:].view(self.args.base_class, -1).clone()
                    constructed_prompt = torch.matmul(weight.t(),base_prompt)
                    constructed_prompt = constructed_prompt.view(self.args.way, self.pre_length+self.after_length,-1)
                                
                    new_prompt = constructed_prompt+learn_prompt
                    #new_prompt_cluster = constructed_prompt.unsqueeze(0)+learn_prompt_cluster
                                
                    old_prompt = self.prompt_text[:self.args.base_class + self.args.way * (session - 1), :].clone().detach()
                    #old_prompt_cluster = self.cluster_prompt_text[:,:self.args.base_class+ self.args.way * (session - 1), :,:].clone().detach()
                    prompt = torch.cat([old_prompt,new_prompt], dim=0)
                    #prompt_cluster = torch.cat([old_prompt_cluster,new_prompt_cluster], dim=1)     
                    prompt_cluster = self.cluster_prompt_text  

                    logits,arc_output = self.joint_finetune(data,fc,cluster_proto,prompt,prompt_cluster,label)
                    
                    acc = count_acc(logits, label)
                    loss_ce = F.cross_entropy(arc_output, label)
                    
                    #loss_con = loss_con.mean(1).mean(0)
                    
                    loss = loss_ce #+ loss_con*10
                    #loss_ce.backward()
                    #sim_cons = self.measure_imgtxt(cluster_proto.view(self.cluster_num,-1),cluster_text.view(self.cluster_num,-1))#torch.stack(sim_cons, dim=0)
                    #cons_label =  torch.arange(self.cluster_num).cuda()
                    #print(loss_con.size())
                    #loss_con =  #F.cross_entropy(sim_cons, cons_label)
                    loss.backward()
                    optimizer.step()

                    print('finetune session cross entropy={:.3f},{:.3f}, acc={:.3f}'.format(loss_ce.item(),loss.item(),acc*100))
        self.prompt_text.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(new_prompt.data)
        #self.cluster_prompt_text.data[:,self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(new_prompt_cluster.data)
        #self.fc.weight.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(new_fc.data)

