## PDO Image Classification
---


## 组合配置
```
data_list_32 = ["CIFAR10", "CIFAR100", "SVHN", "STL10","EuroSAT"]
model_list_32 = ["resnet20","resnet32", "resnet56","wrn16_32","wrn28_10","vgg11_32","vgg16_32"]
data_list_224 = [ "FGVCAircraft","OxfordIIITPet","Caltech101", "Flowers102", "StanfordCars", "Clothing1M", "Country211"]
data_lt_list_224 = ["Places-LT", "ImageNet-LT"]
model_list_224 = ["resnet18", "resnet50","mobilenetv2","vgg11","vgg16"]
model_vit_list_224 = ["vit_b_16","swin_transformer_tiny"]

lr_decay_list = ["StepLR", "ExponentialLR", "MultiStepLR", "CosineAnnealingLR"]
optimize_list = ["Adam", "SGD", "RMSprop", "AdamW", "Adadelta"] 
criterion_list = ["CrossEntropy", "LabelSmooth", "Bootstrap", "FocalLoss", "LogitAdj"]

corruption_type_list =["clean","uniform", "flip", "imbalance"]
noise_radio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
imbalance_radio_list = [10, 20, 30, 40, 50, 60, 70, 80, 90,100, 120, 150, 200]
```


## 机器扩展

+ 1、扩展机器新建root文件夹
+ 2、在root文件加下，复制14个images数据集文件夹：images
+ 3、在root文件加下，复制2个预训练模型文件夹：images_model
+ 4、在root文件加下，新建存储td文件夹：images_td
+ 5、修改：train.py中以上三个文件的路径


## 测试

CUDA_VISIBLE_DEVICES=1 python main.py --flag 0 --root_dir "path"

## 循环训练

CUDA_VISIBLE_DEVICES=1 python main.py --flag 1 --root_dir "path"


