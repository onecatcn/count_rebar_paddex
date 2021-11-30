#!/usr/bin/env python
# coding: utf-8

# # 1. Install PaddleX

# In[2]:


get_ipython().system('pip install paddlex==2.0.0 -i https://mirror.baidu.com/pypi/simple')


# # 2. Check the instalation and version of PaddleX

# In[3]:


import paddlex
print('paddlex version : ', paddlex.__version__)


# # 3. Data prep

# In[4]:


## unzip the dataset in the current directory
get_ipython().system('unzip -o /home/aistudio/data/data107745/dataset_reinforcing_steel_bar_counting.zip')


# In[5]:


# split the dataset into two parts: train and val
get_ipython().system('paddlex --split_dataset --format VOC --dataset_dir dataset_reinforcing_steel_bar_counting --val_value 0.15')


# # 5. Model train and inference

# In[11]:


# train the model, with log saving at the root directory.
get_ipython().system('python code/train.py > log')


# In[ ]:


# model inference
get_ipython().system('python code/infer.py')


# In[ ]:


# model export
paddlex --export_inference --model_dir=output/yolov3_mobilnetv1/best_model --save_dir=output/inference_model --fixed_input_shape=[480,480]


# # 6. Model optimization strategies

# 模型优化
# 改主干网络、改变网络输入、anchor聚类、数据增强

# In[ ]:


##### several stategies of model optimization are provided in this session. These codes will need to replace the correspoinding ones in the train.py.
##edit the network: backbone network is changed from MobileNetV1 to yolov3_resnet34

model = pdx.det.YOLOv3(num_classes=num_classes, backbone='yolov3_resnet34', label_smooth=True, ignore_threshold=0.7)

##edit the network input: the file train_list.txt shall be editted
train_dataset = pdx.datasets.VOCDetection(
    data_dir='dataset_reinforcing_steel_bar_counting',
    file_list='dataset_reinforcing_steel_bar_counting/train_list.txt',
    label_list='dataset_reinforcing_steel_bar_counting/labels.txt',
    transforms=train_transforms,
    shuffle=True)

##data augumentation: randomrotate is added to the preprocessing in additional to the existing ones.
train_transforms = transforms.Compose([
    transforms.MixupImage(mixup_epoch=-1),
    transforms.RandomDistort(),
    ## 增加randomrotate
    transforms.RandomRotate(),
    transforms.RandomExpand(),
    transforms.RandomCrop(),
    transforms.Resize(
        target_size=480, interp='RANDOM'),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(),
])



# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
