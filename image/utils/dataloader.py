

import os
import glob
import numpy as np
from PIL import Image

import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms as transforms

# 固定numpy随机数种子
np.random.seed(0)


# 图像预处理步骤
transform32_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),
            ])
transform32_val = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),
            ])


transform224_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

transform224_val = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])



class TrainSampleImageFolder(ImageFolder):
    def __init__(self,root: str, class_rate: float, data_rate: float, sample_td: bool,transform=None):
        super().__init__(root=root,transform=transform)
        # 对类别进行采样
        self.targets = [s[1] for s in self.samples]
        self.targets_list = list(set(self.targets))
        # 对targets_list 进行抽样
        self.targets_list = [i for i in self.targets_list if np.random.rand() < class_rate]
        self.samples = [s for s in self.samples if s[1] in self.targets_list]
        # 对self.samples 按照 train_rate 进行抽样
        self.samples = [s for s in self.samples if np.random.rand() < data_rate]
        self.path = [s[0] for s in self.samples]
        self.targets = [s[1] for s in self.samples]
        self.targets_list = list(set(self.targets))
        self.class_number = len(self.targets_list)
        # 新旧target的映射
        self.target_map = {self.targets_list[i]:i for i in range(self.class_number)}
        self.targets = [self.target_map[i] for i in self.targets]
        self.class_count = [self.targets.count(i) for i in range(self.class_number)]

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index: int):
        path, target = self.path[index], self.targets[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = self.targets[index]
        return index,sample, target 


class TdSampleImageFolder(ImageFolder):
    def __init__(self,root: str, paths: list, targets: list,transform=None):
        super().__init__(root=root,transform=transform)
        self.path,self.targets = paths,targets
        self.class_number = len(list(set(self.targets)))
        if self.class_number <100: max_number = 50
        elif self.class_number < 500: max_number = 30
        else: max_number = 20
        # 二次采样，每个类别都要采样，并且每个类别最多保留抽样30张图片
        self.samples = []
        for i in range(self.class_number):
            index = [j for j in range(len(self.targets)) if self.targets[j] == i]
            if len(index) > max_number:
                index = np.random.choice(index,max_number,replace=False)
            for j in index:
                self.samples.append((self.path[j],self.targets[j]))
        self.path = [s[0] for s in self.samples]
        self.targets = [s[1] for s in self.samples]
        self.class_count = [self.targets.count(i) for i in range(self.class_number)]
        print(f"train dataset class number: {self.class_number} class count: {self.class_count}")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index: int):
        path, target = self.path[index], self.targets[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = self.targets[index]
        return index,sample, target 
    
    
class TrainNoiseImageFolder(ImageFolder):
    def __init__(self,root: str, train_rate: float, sample_td: bool,transform=None):
        super().__init__(root=root,transform=transform)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.path[index], self.targets[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = self.targets[index]
        return index,sample, target 
    

class TdNoiseImageFolder(ImageFolder):
    def __init__(self,root: str, train_rate: float, sample_td: bool,transform=None):
        super().__init__(root=root,transform=transform)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.path[index], self.targets[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = self.targets[index]
        return index,sample, target 



class TrainTailImageFolder(ImageFolder):
    def __init__(self,root: str, train_rate: float, sample_td: bool,transform=None):
        super().__init__(root=root,transform=transform)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.path[index], self.targets[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = self.targets[index]
        return index,sample, target 
    

class TdTailImageFolder(ImageFolder):
    def __init__(self,root: str, train_rate: float, sample_td: bool,transform=None):
        super().__init__(root=root,transform=transform)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.path[index], self.targets[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = self.targets[index]
        return index,sample, target 

    

if __name__ == "__main__":

       
    train_dataset = TrainSampleImageFolder('/mnt/st_data/dataset/places-lt/train',train_rate=0.13,transform=transform224_train,sample_td=True)
    target_map = train_dataset.target_map
    td_dataset = TdSampleImageFolder('/mnt/st_data/dataset/places-lt/train',target_map,transform=transform224_train,sample_td=True)
    # td_dataset = TrainImageFolder('/mnt/st_data/dataset/places-lt/train',train_rate=0.13,transform=transform224_train,sample_td=True)
    # val_dataset = ValImageFolder('/mnt/st_data/dataset/places-lt/test',transform=transform224_val)

    # train_loader = DataLoader(dataset=train_dataset,batch_size=5000,shuffle=True,num_workers=8,pin_memory=True)
    # td_loader = DataLoader(dataset=td_dataset,batch_size=5000,shuffle=True,num_workers=8,pin_memory=True)
    # val_loader = DataLoader(dataset=val_dataset,batch_size=5000,shuffle=True,num_workers=8,pin_memory=True)

    # for i, (index, data, target) in enumerate(train_loader):
    #     print(f"index: {index} data: {data.shape} target: {target.shape}")

    # for i, (index, data, target) in enumerate(td_loader):
    #     print(f"index: {index} data: {data.shape} target: {target.shape}")

    # for i, (index, data, target) in enumerate(val_loader):
    #     print(f"index: {index} data: {data.shape} target: {target.shape}")