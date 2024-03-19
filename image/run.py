import os
import time
import json
from thop import profile
import torch.optim as optim
from torch.cuda.amp import GradScaler

from nets import get_model_from_name
from utils.dataloader import load_config
from utils.losses import *
from utils.utils_fit import *


def task_app(config, base_dataset_dir, bath_model_dir, output_dir):
    # 1. 加载配置文件
    uuids = config["uuid"]
    # 2. 加载数据集
    train_loader, td_loader, test_loader, target_map, cls_num_list = load_config(config, base_dataset_dir)

    # 3. 加载模型
    backbone = config["model"]["backbone"]
    num_classes = len(target_map)
    if backbone == "vit_b_16": bath_model_dir = os.path.join(bath_model_dir, "vit.pth")
    if backbone == "swin_transformer_tiny": bath_model_dir = os.path.join(bath_model_dir, "swin.pth")
    pretrained = bath_model_dir if config["model"]["pretrained"]==1 else 0
    model = get_model_from_name[backbone](num_classes = num_classes, pretrained = pretrained)
    model = model.cuda()
    # 4. 加载损失函数
    loss_fuction = config["model"]["loss_fuction"]
    criterion = {
            'CrossEntropy'  : CrossEntropyLoss(),
            'LabelSmooth'  : LabelSmoothingLoss(),
            'Bootstrap'  : SoftBootstrappingLoss(),
            'FocalLoss' : FocalLoss(),
            'LogitAdj' : LogitAdjustmentLoss(cls_num_list),
        }[loss_fuction]
    
    # 5. 优化器
    optimizer_type = config["hyperparameter"]["optimize"]
    Init_lr_fit = config["hyperparameter"]["lr"]
    
    momentum = 0.9
    weight_decay = 5e-4
    optimizer = {
        'Adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay=weight_decay),
        'SGD'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay=weight_decay),
        'RMSprop' : optim.RMSprop(model.parameters(), Init_lr_fit, momentum = momentum, weight_decay=weight_decay),
        'AdamW' : optim.AdamW(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay=weight_decay),
        'Adadelta' : optim.Adadelta(model.parameters(), Init_lr_fit, weight_decay=weight_decay),
    }[optimizer_type]

    print(optimizer)
    # 6 学习率调整
    lr_decay_type = config["hyperparameter"]["lr_decay"]
    epoch = config["hyperparameter"]["epoch"]

    if lr_decay_type == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=0.3*epoch, gamma=0.1)
    elif lr_decay_type == "ExponentialLR":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif lr_decay_type == "MultiStepLR":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.6*epoch),int(0.8*epoch)], gamma=0.1)
    elif lr_decay_type == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(0.5*epoch), eta_min=0.0001)
    else:
        raise ValueError("Invalid lr_decay_type")
    

    # 6. 训练模型
    best_acc = 0
    train_acc = 0
    td_task_dict = {} 
    td_calss_dict = {}
    scalar = GradScaler()

    # 合并训练状态
    config.update({"train_status":"success"})
    config.update({"train_info":"train success"})
    try:
        for ep in range(epoch):
            start = time.time()
            train_loss,train_acc,lr = fit_one_epoch(model, optimizer, criterion, scalar, train_loader)
            # nan or inf
            if str(train_loss) == "nan":
                config.update({"train_status":"fail"})
                config.update({"train_info":"train loss is nan"})
                break
            print("Epoch: %d, Train Loss: %.4f, Train Acc: %.4f, Learning Rate: %.4f,Cost time: %.2f" % (ep, train_loss, train_acc, lr, time.time()-start))
            result_task_json,result_class_json = td_one_epoch(model,criterion, optimizer, td_loader)
            val_loss,val_acc = val_one_epoch(model, test_loader)
            print("Epoch: %d, Test Loss: %.4f, Test Acc: %.4f" % (ep, val_loss, val_acc))
            result_task_json.update({"train_acc":train_acc, "val_acc":val_acc})
            result_class_json.update({"train_acc":train_acc, "val_acc":val_acc})
            td_task_dict[f"e_{ep}"] = result_task_json
            td_calss_dict[f"e_{ep}"] = result_class_json
            scheduler.step()
            if val_acc > best_acc:
                best_acc = val_acc

    except Exception as e:
        # 合并训练状态
        config.update({"train_status":"error"})
        config.update({"train_info":str(e)})

    # 7. 计算模型的flops和参数量
    if config["dataset"]["reslution"] == 32:
        input_shape = [32, 32]
    else:
        input_shape = [224, 224]
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    print("FLOPs: %.2fM, Params: %.2fM" % (flops/1e6, params/1e6))

    # 8. 保存TD和测试集的结果
    # 合并训练集、td和测试集个数
    config.update({"train_num":len(train_loader.dataset),"td_num":len(td_loader.dataset), "val_num":len(test_loader.dataset)})
    # 合并num_list
    config.update({"cls_num_list":cls_num_list})
    # 合并config和flops和params
    config.update({"flops":flops/1e6, "params":params/1e6})
    # 合并best_acc
    config.update({"best_acc":best_acc})


    # 保存config,名字在config_path加上后缀TD
    config_class = config.copy()
    config_class.update({"training_dynamic_class":td_calss_dict})
    config_task = config.copy()
    config_task.update({"training_dynamic_task":td_task_dict})


    output_class_path = os.path.join(output_dir, f"{uuids}_TD_class.json")
    with open(output_class_path, 'w') as f:
        json.dump(config_class, f, indent=4)

    output_task_path = os.path.join(output_dir, f"{uuids}_TD_task.json")
    with open(output_task_path, 'w') as f:
        json.dump(config_task, f, indent=4)



if __name__ == "__main__":
    import glob
    import numpy as np
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--config_path', default="config/configs_32/*.json", type=str, help='config file path')
    parser.add_argument('--base_dataset_dir', default="/home/mengyang/dataset/images/", type=str, help='base dataset dir')
    parser.add_argument('--bath_model_dir', default="/home/mengyang/dataset/images_model/", type=str, help='bath model dir')
    parser.add_argument('--output_dir', default="/home/mengyang/dataset/images_td/", type=str, help='output dir')
    args = parser.parse_args()
    config_path = args.config_path
    base_dataset_dir = args.base_dataset_dir
    bath_model_dir = args.bath_model_dir
    output_dir = args.output_dir

    configs = glob.glob(config_path)

    # 随机抽取50个配置文件
    for config in np.random.choice(configs, 50):
        print(config)
        with open(config, 'r') as f:
            config = json.load(f)
        print(config)
        task_app(config, base_dataset_dir, bath_model_dir, output_dir)    

    # # debug
    # path = "task_28083cf5-e593-11ee-ab5e-b4055d1d7a2d_config9.json"
    # with open(path, 'r') as f:
    #     config = json.load(f)
    # # config["hyperparameter"]["optimize"] = "Adam"
    # config["model"]["backbone"] = "resnet50"
    # # config["hyperparameter"]["lr"] = 0.001
    # # config["hyperparameter"]["lr_decay"] = "CosineAnnealingLR"
    # main(config, base_dataset_dir, bath_model_dir, output_dir="./")
