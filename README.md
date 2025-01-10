### Introduction


![License](https://img.shields.io/badge/License-MIT-brightgreen)
![Copyright](https://img.shields.io/badge/Copyright-CVTEAM-red)

<div align="center">

<img src="https://github.com/iCVTEAM/LRT/blob/master/teaser.jpg" width = "600" height = "400" align=center />
</div>


Depicting novel classes with language descriptions by observing few-shot samples is inherent in human learning systems. This lifelong learning capability helps to distinguish new knowledge from old ones through the increase of open-world learning, namely Few-Shot Class-Incremental Learning (FSCIL). Existing works to solve this problem mainly rely on the careful tuning of visual encoders, which shows an evident trade-off between the base knowledge and incremental ones. Motivated by human learning systems, we propose a new Language-inspired Relation Transfer (LRT) paradigm to understand objects by joint visual clues and text depictions, composed of two major steps. We first transfer the pretrained text knowledge to the visual domains by proposing a graph relation transformation module and then fuse the visual and language embedding by a text-vision prototypical fusion module. Second, to mitigate the domain gap caused by visual finetuning, we propose context prompt learning for fast domain alignment and imagined contrastive learning to alleviate the insufficient text data during alignment. With collaborative learning of domain alignments and text-image transfer, our proposed LRT outperforms the state-of-the-art models by over 13% and 7% on the final session of mini ImageNet and CIFAR-100 FSCIL benchmarks.

 [Paper Link](https://ieeexplore.ieee.org/abstract/document/10746343)


### Requirements

```
PyTorch>=1.1, tqdm, torchvsion.
```

### Data Preparation


1. Download the benchmark datasets and unzip them in your customized path.
    For miniImageNet dataset from other sharing links in [CEC](https://github.com/icoz69/CEC-CVPR2021), Click [links](https://drive.google.com/drive/folders/11LxZCQj2FRCs0JTsf_dafvTHqFn2yGSN?usp=sharing)  to download.
    For ImageNet100 dataset from other sharing links in [links](https://www.kaggle.com/datasets/ambityga/imagenet100/data) to download.

2. Modify the lines in train.py from 3~5 [links]()
3. unzip these datasets 



### How to run

#### For Incremental Stage:


Step 1  For  mini_imagenet (lower lr_new(0.001) to achieve better learning model.)
```
$python train.py -project base -dataset mini_imagenet -base_mode "ft_cos" -new_mode "ft_cos" -gamma 0.1  -lr_base 0.01  -lr_new 0.01 -decay 0.0005 -epochs_base 0 -epochs_new 20 -schedule Milestone -milestones 40 70 -gpu 1,2 -temperature 16 -model_dir "dir to your path.pth"
```

Step 2  For  ImageNet100 dataset or other datasets

modified the following hyper-parameters in the default project.
```
epochs_new=30, lr_base=0.01, lr_new=0.02, schedule='Milestone', milestones=[20, 35], step=40, decay=0.0005, momentum=0.9, gamma=0.1, temperature=16, not_data_init=False, batch_size_base=128, batch_size_new=0, test_batch_size=100, base_mode='ft_cos', new_mode='ft_cos',
```

#### For Base Pretraining Stage:

Step 3 For  mini_imagenet
```
$python train.py -project base -dataset mini_imagenet -base_mode 'ft_cos' -new_mode 'ft_cos' -gamma 0.1 -lr_base 0.01 -lr_new 0.01 -decay 0.0005 -epochs_base 0 -epochs_new 20 -schedule Milestone -milestones 40 70  -temperature 16 -gpu 0,1
```

Step 4 For  ImageNet100 dataset or other datasets
```
modified the hyper-parameters following step 2, dataset name shoud be "imagenet100"
```

#### Pretraining models
| Type/Datasets | ImageNet100                                                | mini-ImageNet                                                |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Pretrained    | [Links](https://drive.google.com/file/d/1KwXPV7FjYMaI28XGr4OHMy3VjtU4OtHN/view?usp=sharing) | [Links](https://drive.google.com/file/d/1hRecni9x5lhevsnKAcCEcflJSgYrIDPx/view?usp=drive_link) |


#### Training logs
| Type/Datasets | ImageNet100                                                 | mini-ImageNet                                                |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Base/incremental    | [Links](https://drive.google.com/file/d/1eKuVMtuxXr62SOqhpkDWG6_9nPcqpgcK/view?usp=sharing) | [Links](https://drive.google.com/file/d/1P77ecM55lAwiIasZ4CWXTsxA1pQ_UvrO/view?usp=drive_link) |





#### Running Tips

1. The performance may be fluctuated in different GPUs and PyTorch platforms. Pytorch versions higher than 1.7.1 are tested. 
2.  Two 3090/4090 GPUs are used in our experiments. 


### To do

1. The project is still ongoing, finding suitable platforms and GPU devices for complete stable results.

2. The project is re-constructed for better understanding, we release this version for a quick preview of our paper.

   
### License

The code of the paper is freely available for non-commercial purposes. Permission is granted to use the code given that you agree:

1. That the code comes "AS IS", without express or implied warranty. The authors of the code do not accept any responsibility for errors or omissions.

2. That you include necessary references to the paper in any work that makes use of the code. 

3. That you may not use the code or any derivative work for commercial purposes as, for example, licensing or selling the code, or using the code with a purpose to procure a commercial gain.

4. That you do not distribute this code or modified versions. 

5. That all rights not expressly granted to you are reserved by the authors of the code.

### Citations:

Please remember to cite us if u find this useful :)
```
@article{zhao2024language,
  title={Language-Inspired Relation Transfer for Few-Shot Class-Incremental Learning},
  author={Zhao, Yifan and Li, Jia and Song, Zeyin and Tian, Yonghong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```


### Acknowledgment
Our project references the codes in the following repos.
Please refer to these codes for details.
- [CLIP](https://github.com/openai/CLIP)
- [CEC](https://github.com/icoz69/CEC-CVPR2021)



