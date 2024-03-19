
import numpy as np
import json
import uuid
import os
import uuid

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


def generate_config(number,reslution,data_list,model_list,output_dir):
    # 根据配置随机生成m个json文件,保存在configs文件夹下
    for n in range(number):
        # 每份数据和模型对应10组超参
        uuid_root = str(uuid.uuid1())
        seed = int(np.random.randint(1,10000))
        #选定数据集
        dataset_name = np.random.choice(data_list)
        corruption_type = np.random.choice(corruption_type_list)
        if "Places-LT" in data_list:
            corruption_type = "clean"
        noise_radio = np.random.choice(noise_radio_list)
        imbalance_radio = int(np.random.choice(imbalance_radio_list))
        #选定模型
        model_name = np.random.choice(model_list)

        for t in range(10):
            uuid_str = f"{uuid_root}_config{t}"
            lr_decay =  np.random.choice(lr_decay_list)
            optimize = np.random.choice(optimize_list)
            criterion = np.random.choice(criterion_list)

            batch_size = int(np.random.randint(32,1024))
            # lr取值范围(0.001,0.2)
            

            pretrained = int(np.random.choice([0, 1]))
            # 小图全部不预训练
            if data_list == data_list_32:
                pretrained = 0

            if model_list==model_vit_list_224:
                batch_size = min(256, batch_size)

            if pretrained==0:
                epoch = int(np.random.randint(20,120))
                lr = round(np.random.uniform(0.01,0.1),4)
            else:
                epoch = int(np.random.randint(10,30))
                lr = round(np.random.uniform(0.0001,0.01),4)

            config = {
                "uuid":uuid_str,
                "dataset":{
                    "seed":seed,
                    "reslution":int(reslution),
                    "dataset_name":dataset_name, 
                    "regenerate":{
                        "corruption_type":corruption_type,
                        "noise_radio":noise_radio,
                        "imbalance_radio":imbalance_radio
                    }
                },
                "model":{
                    "backbone":model_name,
                    "pretrained":pretrained,
                    "loss_fuction":criterion
                },
                "hyperparameter":{
                    "epoch":epoch,
                    "batch_size":batch_size,
                    "lr":lr,
                    "lr_decay":lr_decay,
                    "optimize":optimize
                }
            }

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_path= os.path.join(output_dir, f"{uuid_str}.json")
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=4)


if __name__ == "__main__":
    # 生成小图1000个配置
    generate_config(1000,32,data_list_32,model_list_32,"./config/configs_32/")
    # 生成大图1000个配置
    generate_config(900,224,data_list_224,model_list_224,"./config/configs_224_1/")
    generate_config(50,224,data_lt_list_224,model_list_224,"./config/configs_224_2/")
    generate_config(40,224,data_list_224,model_vit_list_224,"./config/configs_224_3/")
    generate_config(10,224,data_lt_list_224,model_vit_list_224,"./config/configs_224_4/")