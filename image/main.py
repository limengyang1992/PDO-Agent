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
from mysql_tool import MySQLJSONStorage

import argparse
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--root_dir', default="/home/mengyang/dataset/", type=str, help='config file path')
parser.add_argument('--sub_dataset_dir', default="images_dataset", type=str, help='sub_dataset_dir')
parser.add_argument('--sub_model_dir', default="images_model", type=str, help='sub_model_dir')
parser.add_argument('--sub_output_dir', default="images_td", type=str, help='sub_output_dir')
parser.add_argument('--flag', default=0, type=int, help='config file path')
args = parser.parse_args()

database = MySQLJSONStorage()


def app(config, base_dataset_dir, bath_model_dir, output_dir):
    
    # 1. 加载配置文件
    config = json.loads(config)
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

    root_dir = args.root_dir
    dataset_dir = os.path.join(root_dir, args.sub_dataset_dir)
    model_dir = os.path.join(root_dir, args.sub_model_dir)
    output_dir = os.path.join(root_dir, args.sub_output_dir)

    if args.flag==0:
        result = database.get_pending_json_file()
        config = result["json_data"]
        print(config)
        app(config, dataset_dir, model_dir, output_dir)

    while args.flag:
        time.sleep(2)
        print("===============starting next task==========")
        # 1. 获取status=0的json文件
        result = database.get_pending_json_file()
        name = result["name"]
        database.update_status(name, status=1)
        
        if result is None:
            print("No pending json file")
            break
        print(result)
        # 2. 获取config
        config = result["json_data"]
        # 3. 训练模型并更新状态
        try:
            app(config, dataset_dir, model_dir, output_dir)
            database.update_status(name, status=2)
        except Exception as e:
            print(e)
            database.update_status(name, status=4)



