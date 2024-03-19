
import numpy as np
import json
import uuid
import os


#根据配置随机生成m个json文件,保存在configs文件夹下
def generate_config(m=2000,path="./configs/"):

    for _ in range(m):
        uuid_str = str(uuid.uuid1())
        dataset_32 = np.random.choice(["CIFAR10", "MNIST", "Qmnist", "KMNIST", "Fashion MNIST", "SVHN", "STL10", "CIFAR100"])
        dataset_224 = np.random.choice(["FER2013","Clothing1M","EuroSAT", "FGVCAircraft", "OxfordIIIT Pet", "Caltech101", "Flowers102", "Stanford Cars", "Caltech256", "SUN397", "Places-LT", "ImageNet-LT"])
        class_sample = {
            "type":"class_sample",
            "rate":round(np.random.uniform(0.1,0.9),1),
            "class_min":2,
            "sample_min":2000
        }
        sample_sample = {
            "type":"sample_sample",
            "rate": round(np.random.uniform(0.1,0.9),1),
            "sample_min":2000
        }
        noise_generate = {
            "type":"noise_generate",
            "sample":int(np.random.choice([0, 1], p=[0.1, 0.9])),
            "sample_type":np.random.choice(["class", "sample"]),
            "types":np.random.choice(["Asym", "Sym"]),
            "rate":round(np.random.uniform(0.1,0.9),1),
            "sample_min":2000
        }
        long_tail_generate = {
            "type":"long_tail_generate",
            "sample":int(np.random.choice([0, 1], p=[0.1, 0.9])),
            "sample_type":np.random.choice(["class", "sample"]),
            "rate":int(np.random.choice([10,20,50,100,200])),
            "sample_min":2000
        }

        reslution = np.random.choice(["dataset_32", "dataset_224"])
        if reslution == "dataset_32": 
            reslution=32
            dataset_name=dataset_32
        else: 
            reslution=224
            dataset_name=dataset_224
        regenerate = np.random.choice([class_sample, sample_sample, noise_generate, long_tail_generate])

        for t in range(10):
            epoch = np.random.choice([int(np.random.randint(5,20)), int(np.random.choice([50,80,100,120,150,180,200]))])
            config = {
                "uuid":uuid_str,
                "dataset":{
                    "reslution":reslution,
                    "dataset_name":dataset_name, 
                    "regenerate":regenerate
                },
                "loss fuction":np.random.choice(["CE", "LS", "BS", "Focal", "LA"]),
                "model":np.random.choice(["VGG", "ResNet", "Swin", "Vit", "Mobilenet"]),
                "hyperparameter":{
                    "Epoch":int(epoch),
                    "BS":int(np.random.choice([32,64,128,256,512,1024])),
                    "LR":np.random.choice([0.0001,0.001,0.01,0.1]),
                    "LR decay":np.random.choice(["MultiStep", "Cosine", "Warm-up"]),
                    "optimize":np.random.choice(["SGD", "Adam", "AdamW"])
                }
            }
            with open(os.path.join(path, f"task_{uuid_str}_{t}.json"), 'w') as f:
                json.dump(config, f, indent=4)



if __name__ == "__main__":
    generate_config(10,"./dataset/tasks/")

