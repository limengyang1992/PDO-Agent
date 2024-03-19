

import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


# 图像预处理步骤
transform32 = transforms.Compose([
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

transform224 = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

transform224_val = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])


def uniform_corruption(corruption_ratio, num_classes):
    eye = np.eye(num_classes)
    noise = np.full((num_classes, num_classes), 1/num_classes)
    corruption_matrix = eye * (1 - corruption_ratio) + noise * corruption_ratio
    return corruption_matrix


def flip1_corruption(corruption_ratio, num_classes, seed):
    np.random.seed(seed)
    corruption_matrix = np.eye(num_classes) * (1 - corruption_ratio)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        corruption_matrix[i][np.random.choice(row_indices[row_indices != i])] = corruption_ratio
    return corruption_matrix  


def reconstruct_dataset(seed,samples,corruption_type,noise_radio,imbalance_radio):
    np.random.seed(seed)

    if len(samples)>50000:
        class_rate  = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0])
        data_rate = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0])
    else:
        class_rate  = np.random.choice([0.5, 0.6, 0.7, 0.8, 0.9,1.0])
        data_rate = np.random.choice([0.5, 0.6, 0.7, 0.8, 0.9,1.0])

    # assert corruption_type in ["clean","uniform", "flip", "imbalance"]
    # assert noise_radio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # assert imbalance_radio in [10, 20, 30, 40, 50, 60, 70, 80, 90,100, 120, 150, 200]

    # 对类别进行采样
    targets = [s[1] for s in samples]
    targets_list = list(set(targets))
    targets_list = [i for i in targets_list if np.random.rand() < class_rate]
    if len(targets_list) < 2:
        targets_list = np.random.choice(list(set(targets)), 2)

    samples = [s for s in samples if s[1] in targets_list]
    print(f"step1: class_rate:{class_rate}, len targets_list1: {len(targets_list)}, samples1: {len(samples)}")
    # 对samples 按照 train_rate 进行抽样,不能低于5000
    s0 = [s for s in samples if np.random.rand() < max(data_rate,5000/len(samples))]

    samples = s0
    path = [s[0] for s in samples]
    targets = [s[1] for s in samples]
    targets_list = list(set(targets))
    class_number = len(targets_list)
    print(f"step2: data_rate:{data_rate}, len targets_list2: {len(targets_list)}, samples2: {len(samples)}")
    # 新旧target的映射
    target_map = {targets_list[i]:i for i in range(class_number)}
    targets = [target_map[i] for i in targets]
    
    if corruption_type=="imbalance": 
        imbalanced_num_list = []
        index_to_train = []
        sample_num = int((len(samples)) / class_number)

        for class_index in range(class_number):
            imbalanced_num = sample_num / (imbalance_radio ** (class_index / (class_number - 1)))
            imbalanced_num_list.append(int(imbalanced_num))
        np.random.shuffle(imbalanced_num_list)

        for class_index in range(class_number):
            index_to_class = [index for index, t in enumerate(targets) if t == class_index]
            np.random.shuffle(index_to_class)
            index_to_class_for_train = index_to_class[:imbalanced_num_list[class_index]]
            index_to_train.extend(index_to_class_for_train)

        targets = [targets[i] for i in index_to_train]
        path = [path[i] for i in index_to_train]


    elif corruption_type in ["uniform", "flip"]: 
        if corruption_type == "uniform":
            corruption_matrix = uniform_corruption(noise_radio, class_number)
        else:
            corruption_matrix = flip1_corruption(noise_radio, class_number, seed)
        for index in range(len(targets)):
            p = corruption_matrix[targets[index]]
            targets[index] = np.random.choice(len(target_map), p=p)

    class_count = [targets.count(i) for i in range(class_number)]
    print(f"train dataset class number: {class_number}")

    return path, targets, target_map,class_count

def reconstruct_td(samples, target_map, if_td):
    targets_list = [k for k,v in target_map.items()]
    class_number = len(list(set(targets_list)))
    if class_number <100: max_number = 100
    elif class_number < 300: max_number = 50
    elif class_number < 500: max_number = 20
    else: max_number = 10
    # 测试情况，最大取500张图片
    if not if_td: max_number=100 if len(samples) > 10000 else 500
    s_samples = []
    for i in targets_list:
        i_samples = [s for s in samples if s[1] == i]
        if len(i_samples) > max_number:
            i_samples = i_samples[:max_number]
        s_samples += i_samples
    samples = s_samples
    path = [x[0] for x in s_samples]
    targets = [target_map[x[1]] for x in s_samples]
    print(f"td or val dataset class number: {class_number} samples count: {len(samples)}")

    return path, targets


def reconstruct_val(samples, target_map):
    targets_list = [k for k,v in target_map.items()]
    class_number = len(list(set(targets_list)))
    max_number = 500
    # 二次采样，每个类别500张图片
    s_samples = []
    for i in targets_list:
        s_samples = [s for s in samples if s[1] == i]
        if len(s_samples) > max_number:
            s_samples = s_samples[:max_number]
        s_samples += s_samples
    samples = s_samples
    path = [x[0] for x in s_samples]
    targets = [target_map[x[1]] for x in s_samples]
    print(f"train dataset class number: {class_number} samples count: {len(samples)}")

    return path, targets


class RECIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self,seed: int,root: str, corruption_type: str,noise_radio: float,imbalance_radio: float,transform=None, train: bool =True):
        super().__init__(root=root,train=train,transform=transform,download=True)
        samples = list(zip(self.data, self.targets))
        self.data, self.targets, self.target_map,self.class_count = reconstruct_dataset(seed,samples, corruption_type, noise_radio, imbalance_radio)

    
class TDCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self,root: str, target_map: dict,transform=None, train=None,if_td=None):
        super().__init__(root=root,train=train,transform=transform,download=True)
        samples = list(zip(self.data, self.targets))
        self.data, self.targets = reconstruct_td(samples, target_map, if_td)


class RECIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self,seed: int,root: str,corruption_type: str,noise_radio: float,imbalance_radio: float,transform=None, train: bool =True):
        super().__init__(root=root,train=train,transform=transform,download=True)

        samples = list(zip(self.data, self.targets))
        self.data, self.targets, self.target_map,self.class_count = reconstruct_dataset(seed,samples, corruption_type, noise_radio, imbalance_radio)


class TDCIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self,root: str, target_map: dict,transform=None, train=None,if_td=None):
        super().__init__(root=root,train=train,transform=transform,download=True)
        samples = list(zip(self.data, self.targets))
        self.data, self.targets = reconstruct_td(samples, target_map, if_td)


class RESVHN(torchvision.datasets.SVHN):
    def __init__(self,seed: int,root: str,corruption_type: str,noise_radio: float,imbalance_radio: float,transform=None, split: str = "train"):
        super().__init__(root=root,split=split,transform=transform,download=True)

        samples = list(zip(self.data, self.labels))
        self.data, self.labels, self.target_map,self.class_count = reconstruct_dataset(seed,samples, corruption_type, noise_radio, imbalance_radio)

class TDSVHN(torchvision.datasets.SVHN):
    def __init__(self,root: str, target_map: dict,transform=None, split: str = "train",if_td=None):
        super().__init__(root=root,split=split,transform=transform,download=True)
        samples = list(zip(self.data, self.labels))
        self.data, self.labels = reconstruct_td(samples, target_map, if_td)


class RESTL10(torchvision.datasets.STL10):
    def __init__(self,seed: int,root: str,corruption_type: str,noise_radio: float,imbalance_radio: float,transform=None, split: str = "train"):
        super().__init__(root=root,split=split,transform=transform,download=True)

        samples = list(zip(self.data, self.labels))
        self.data, self.labels, self.target_map,self.class_count = reconstruct_dataset(seed,samples, corruption_type, noise_radio, imbalance_radio)
        self.data = np.array(self.data)
        

class TDSTL10(torchvision.datasets.STL10):
    def __init__(self,root: str, target_map: dict,transform=None, split: str = "train",if_td=None):
        super().__init__(root=root,split=split,transform=transform,download=True)
        samples = list(zip(self.data, self.labels))
        self.data, self.labels = reconstruct_td(samples, target_map, if_td)
        self.data = np.array(self.data)


class REEuroSAT(torchvision.datasets.EuroSAT):
    def __init__(self,seed: int,root: str,corruption_type: str,noise_radio: float,imbalance_radio: float,transform=None, split: str = "train"):
        super().__init__(root=root,transform=transform,download=True)

        samples = self.samples
        self.data, self.targets, self.target_map,self.class_count = reconstruct_dataset(seed,samples, corruption_type, noise_radio, imbalance_radio)
        self.samples = list(zip(self.data, self.targets))

class TDEuroSAT(torchvision.datasets.EuroSAT):
    def __init__(self,root: str, target_map: dict,transform=None, split: str = "train",if_td=None):
        super().__init__(root=root,transform=transform,download=True)
        samples = self.samples
        self.data, self.targets = reconstruct_td(samples, target_map, if_td)
        self.samples = list(zip(self.data, self.targets))

class REFGVCAircraft(torchvision.datasets.FGVCAircraft):
    def __init__(self,seed: int,root: str,corruption_type: str,noise_radio: float,imbalance_radio: float,transform=None, split: str = "train"):
        super().__init__(root=root,split=split,transform=transform,download=True)

        samples = list(zip(self._image_files, self._labels)) 
        self._image_files, self._labels, self.target_map,self.class_count = reconstruct_dataset(seed,samples, corruption_type, noise_radio, imbalance_radio)

class TDFGVCAircraft(torchvision.datasets.FGVCAircraft):
    def __init__(self,root: str, target_map: dict,transform=None, split: str = "trainval",if_td=None):
        super().__init__(root=root,split=split,transform=transform,download=True)
        samples = list(zip(self._image_files, self._labels)) 
        self._image_files, self._labels, = reconstruct_td(samples, target_map, if_td)

class REOxfordIIITPet(torchvision.datasets.OxfordIIITPet):
    def __init__(self,seed: int,root: str,corruption_type: str,noise_radio: float,imbalance_radio: float,transform=None, split: str = "trainval"):
        super().__init__(root=root,split=split,transform=transform,download=True)

        samples = list(zip(self._images, self._labels)) 
        self._images, self._labels, self.target_map,self.class_count = reconstruct_dataset(seed,samples, corruption_type, noise_radio, imbalance_radio)

class TDOxfordIIITPet(torchvision.datasets.OxfordIIITPet):
    def __init__(self,root: str, target_map: dict,transform=None, split: str = "test",if_td=None):
        super().__init__(root=root,split=split,transform=transform,download=True)
        samples = list(zip(self._images, self._labels)) 
        self._images, self._labels, = reconstruct_td(samples, target_map, if_td)


class REFlowers102(torchvision.datasets.Flowers102):
    def __init__(self,seed: int,root: str,corruption_type: str,noise_radio: float,imbalance_radio: float,transform=None, split: str = "train"):
        super().__init__(root=root,split=split,transform=transform,download=True)
        samples = list(zip(self._image_files, self._labels))
        self._image_files, self._labels, self.target_map,self.class_count = reconstruct_dataset(seed,samples, corruption_type, noise_radio, imbalance_radio)


class TDFlowers102(torchvision.datasets.Flowers102):
    def __init__(self,root: str,target_map: dict, transform=None, split=None,if_td=None):
        super().__init__(root=root,split=split,transform=transform,download=True)
        samples = list(zip(self._image_files, self._labels))
        self._image_files, self._labels, = reconstruct_td(samples, target_map,if_td)


class TrainImageFolder(ImageFolder):
    def __init__(self,seed: int,root: str,corruption_type: str,noise_radio: float,imbalance_radio: float,transform=None):
        super().__init__(root=root,transform=transform)
        self._image_files, self._labels, self.target_map,self.class_count = reconstruct_dataset(seed, self.samples, corruption_type, noise_radio, imbalance_radio)
        self.samples = list(zip(self._image_files, self._labels))


class TdImageFolder(ImageFolder):
    def __init__(self,root: str,target_map: dict, transform=None,if_td=None):
        super().__init__(root=root,transform=transform)
        self._image_files, self._labels = reconstruct_td(self.samples, target_map, if_td)
        self.samples = list(zip(self._image_files, self._labels))
     

def load_config(config,root):
    seed = config["dataset"]["seed"]
    dataset = config["dataset"]["dataset_name"] 
    batch_size = config["hyperparameter"]["batch_size"]
    corruption_type = config["dataset"]["regenerate"]["corruption_type"]
    noise_radio = config["dataset"]["regenerate"]["noise_radio"]
    imbalance_radio = config["dataset"]["regenerate"]["imbalance_radio"]


    if dataset == "CIFAR10":
        train_dataset = RECIFAR10(seed, root=root, corruption_type=corruption_type, noise_radio=noise_radio, imbalance_radio=imbalance_radio, transform=transform32,train=True)
        target_map = train_dataset.target_map
        td_dataset = TDCIFAR10(root=root, target_map=target_map, transform=transform32_val, train=True, if_td=True)
        test_dataset = TDCIFAR10(root=root, target_map=target_map, transform=transform32_val, train=False, if_td=False)

    elif dataset == "CIFAR100":
        train_dataset = RECIFAR100(seed, root=root, corruption_type=corruption_type, noise_radio=noise_radio, imbalance_radio=imbalance_radio, transform=transform32, train=True)
        target_map = train_dataset.target_map
        td_dataset = TDCIFAR100(root=root, target_map=target_map, transform=transform32_val, train=True,if_td=True)
        test_dataset = TDCIFAR100(root=root, target_map=target_map, transform=transform32_val, train=False,if_td=False)


    elif dataset == "SVHN":
        train_dataset = RESVHN(seed, root=root, corruption_type=corruption_type, noise_radio=noise_radio, imbalance_radio=imbalance_radio, transform=transform32, split="train")
        target_map = train_dataset.target_map
        td_dataset = TDSVHN(root=root, target_map=target_map, transform=transform32_val, split="train",if_td=True)
        test_dataset = TDSVHN(root=root, target_map=target_map, transform=transform32_val, split="test",if_td=False)

    elif dataset == "STL10":
        train_dataset = RESTL10(seed, root=root, corruption_type=corruption_type, noise_radio=noise_radio, imbalance_radio=imbalance_radio, transform=transform32, split="train")
        target_map = train_dataset.target_map
        td_dataset = TDSTL10(root=root, target_map=target_map, transform=transform32_val, split="train",if_td=True)
        test_dataset = TDSTL10(root=root, target_map=target_map, transform=transform32_val, split="test",if_td=False)


    elif dataset == "EuroSAT":
        train_dataset = REEuroSAT(seed, root=root, corruption_type=corruption_type, noise_radio=noise_radio, imbalance_radio=imbalance_radio, transform=transform32, split="train")
        target_map = train_dataset.target_map
        td_dataset = TDEuroSAT(root=root, target_map=target_map, transform=transform32_val, split="train",if_td=True)
        test_dataset = TDEuroSAT(root=root, target_map=target_map, transform=transform32_val, split="test",if_td=False)

    elif dataset == "FGVCAircraft":
        train_dataset = REFGVCAircraft(seed, root=root, corruption_type=corruption_type, noise_radio=noise_radio, imbalance_radio=imbalance_radio, transform=transform224, split="train")
        target_map = train_dataset.target_map
        td_dataset = TDFGVCAircraft(root=root, target_map=target_map, transform=transform224_val, split="train",if_td=True)
        test_dataset = TDFGVCAircraft(root=root, target_map=target_map, transform=transform224_val, split="test",if_td=False)

    elif dataset == "OxfordIIITPet":
        train_dataset = REOxfordIIITPet(seed, root=root, corruption_type=corruption_type, noise_radio=noise_radio, imbalance_radio=imbalance_radio, transform=transform224, split="trainval")
        target_map = train_dataset.target_map
        td_dataset = TDOxfordIIITPet(root=root, target_map=target_map, transform=transform224_val, split="trainval",if_td=True)
        test_dataset = TDOxfordIIITPet(root=root, target_map=target_map, transform=transform224_val, split="test",if_td=False)


    elif dataset == "Flowers102":
        train_dataset = REFlowers102(seed, root=root, corruption_type=corruption_type, noise_radio=noise_radio, imbalance_radio=imbalance_radio, transform=transform224, split="train")
        target_map = train_dataset.target_map
        td_dataset = TDFlowers102(root=root, target_map=target_map, transform=transform224_val, split="train",if_td=True)
        test_dataset = TDFlowers102(root=root, target_map=target_map, transform=transform224_val, split="test",if_td=False)


    elif dataset == "Caltech101":
        train_dataset = TrainImageFolder(seed, root+'/caltech101/train', corruption_type, noise_radio, imbalance_radio, transform=transform224)
        target_map = train_dataset.target_map
        td_dataset = TdImageFolder(root+'/caltech101/train', target_map, transform=transform224_val,if_td=True)
        test_dataset = TdImageFolder(root+'/caltech101/val', target_map, transform=transform224_val,if_td=False)


    elif dataset == "StanfordCars":
        train_dataset = TrainImageFolder(seed, root+'/stanford_cars/train', corruption_type, noise_radio, imbalance_radio, transform=transform224)
        target_map = train_dataset.target_map
        td_dataset = TdImageFolder(root+'/stanford_cars/train', target_map, transform=transform224_val,if_td=True)
        test_dataset = TdImageFolder(root+'/stanford_cars/val', target_map, transform=transform224_val,if_td=False)

    elif dataset == "Clothing1M":
        train_dataset = TrainImageFolder(seed, root+'/clothing1m/train', corruption_type, noise_radio, imbalance_radio, transform=transform224)
        target_map = train_dataset.target_map
        td_dataset = TdImageFolder(root+'/clothing1m/train', target_map, transform=transform224_val,if_td=True)
        test_dataset = TdImageFolder(root+'/clothing1m/val', target_map, transform=transform224_val,if_td=False)

    elif dataset == "Country211":
        train_dataset = TrainImageFolder(seed, root+'/country211/train', corruption_type, noise_radio, imbalance_radio, transform=transform224)
        target_map = train_dataset.target_map
        td_dataset = TdImageFolder(root+'/country211/train', target_map, transform=transform224_val,if_td=True)
        test_dataset = TdImageFolder(root+'/country211/val', target_map, transform=transform224_val,if_td=False)


    elif dataset == "Places-LT":
        train_dataset = TrainImageFolder(seed, root+'/places-lt/train', corruption_type, noise_radio, imbalance_radio, transform=transform224)
        target_map = train_dataset.target_map
        td_dataset = TdImageFolder(root+'/places-lt/train', target_map, transform=transform224_val,if_td=True)
        test_dataset = TdImageFolder(root+'/places-lt/val', target_map, transform=transform224_val,if_td=False)

    elif dataset == "ImageNet-LT":
        train_dataset = TrainImageFolder(seed, root+'/imagenet_lt/train', corruption_type, noise_radio, imbalance_radio, transform=transform224)
        target_map = train_dataset.target_map
        td_dataset = TdImageFolder(root+'/imagenet_lt/train', target_map, transform=transform224_val,if_td=True)
        test_dataset = TdImageFolder(root+'/imagenet_lt/val', target_map, transform=transform224_val,if_td=False)

    else:
        raise ValueError("dataset name not found")

    cls_num_list = train_dataset.class_count
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16,pin_memory=True)
    td_loader = DataLoader(td_dataset, batch_size=64, shuffle=True, num_workers=8,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=8,pin_memory=True)

    return train_loader, td_loader, test_loader, target_map, cls_num_list
    


if __name__ == "__main__":
    data_list_32 = ["CIFAR10", "CIFAR100", "SVHN", "STL10","EuroSAT"]
    data_list_224 = [ "FGVCAircraft","OxfordIIITPet","Caltech101", "Flowers102", "StanfordCars","Places-LT", "ImageNet-LT", "Clothing1M", "Country211"]

    config = {
        "uuid": "b6cb9788-d868-11ee-bb66-552f1d10f6a7",
        "dataset": {
            "reslution": 32,
            "dataset_name": "CIFAR10",
            "regenerate": {
                "corruption_type": "uniform",
                "noise_radio": 0.2,
                "imbalance_radio": 20
            }
        },
        "model": {
            "backbone": "resnet20",
            "pretrained": "False",
            "loss_fuction": "CrossEntropy"
        },
        "hyperparameter": {
            "epoch": 120,
            "batch_size": 128,
            "lr": 0.1,
            "lr_decay": "MultiStepLR",
            "optimize": "SGD"
        }
    }
    
    config["dataset"]["regenerate"]["corruption_type"] = np.random.choice(["clean","uniform", "flip", "imbalance"])
    config["dataset"]["regenerate"]["noise_radio"] = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    config["dataset"]["regenerate"]["imbalance_radio"] = np.random.choice([10, 20, 30, 40, 50, 60, 70, 80, 90,100, 120, 150, 200])

    for data in data_list_32:
        print(f"dataset: {data}")
        config["dataset"]["dataset_name"] = data
        config["dataset"]["reslution"] = 32 
        output = load_config(config)

        train_loader, td_loader, test_loader, target_map, cls_num_list = output

        for i, (x, y) in enumerate(train_loader):
            print(i,x.shape, y.shape)
            if i > 100: break

        for i, (x, y) in enumerate(td_loader):
            print(i,x.shape, y.shape)
            if i > 100: break

        for i, (x, y) in enumerate(test_loader):
            print(i,x.shape, y.shape)
            if i > 100: break     


    for data in data_list_224:
        print(f"dataset: {data}")
        config["dataset"]["dataset_name"] = data
        config["dataset"]["reslution"] = 224 
        output = load_config(config)
    
        train_loader, td_loader, test_loader, target_map, cls_num_list = output

        for i, (x, y) in enumerate(train_loader):
            print(i,x.shape, y.shape)
            if i > 100: break

        for i, (x, y) in enumerate(td_loader):
            print(i,x.shape, y.shape)
            if i > 100: break

        for i, (x, y) in enumerate(test_loader):
            print(i,x.shape, y.shape)
            if i > 100: break