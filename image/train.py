import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets import get_model_from_name
from utils.callbacks import LossHistory
from utils.dataloader import DataGenerator, detection_collate
from utils.utils import (download_weights, get_classes, get_lr_scheduler,
                         set_optimizer_lr, show_config, weights_init)
from utils.utils_fit import fit_one_epoch

# if __name__ == "__main__":

    # if pretrained:
    #     download_weights(backbone)
    # else:
    #     weights_init(model)
            

    # if backbone not in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base']:
    #     model = get_model_from_name[backbone](num_classes = num_classes, pretrained = pretrained)
    # else:
    #     model = get_model_from_name[backbone](input_shape = input_shape, num_classes = num_classes, pretrained = pretrained)

        
    
    # if fp16:
    #     from torch.cuda.amp import GradScaler as GradScaler
    #     scaler = GradScaler()
    # else:
    #     scaler = None

    # model_train     = model.train()


        
    # optimizer = {
    #     'adam'  : optim.Adam(model_train.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay=weight_decay),
    #     'sgd'   : optim.SGD(model_train.parameters(), Init_lr_fit, momentum = momentum, nesterov=True)
    # }[optimizer_type]
    
    # #---------------------------------------#
    # #   获得学习率下降的公式
    # #---------------------------------------#
    # lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
    
    
    
    
    # train_dataset   = DataGenerator(train_lines, input_shape, True)
    # val_dataset     = DataGenerator(val_lines, input_shape, False)
    

        
    # gen_train       = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True, 
    #                         drop_last=True, collate_fn=detection_collate, sampler=train_sampler)
    # gen_val         = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
    #                         drop_last=True, collate_fn=detection_collate, sampler=val_sampler)
    # #---------------------------------------#
    # #   开始模型训练
    # #---------------------------------------#
    # for epoch in range(Epoch):
    #     #---------------------------------------#
    #     #   如果模型有冻结学习部分
    #     #   则解冻，并设置参数
    #     #---------------------------------------#
    #     if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
    #         batch_size = Unfreeze_batch_size

    #         #-------------------------------------------------------------------#
    #         #   判断当前batch_size，自适应调整学习率
    #         #-------------------------------------------------------------------#
    #         nbs             = 64
    #         lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
    #         lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
    #         if backbone in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base']:
    #             nbs             = 256
    #             lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
    #             lr_limit_min    = 1e-5 if optimizer_type == 'adam' else 5e-4
    #         Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    #         Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    #         #---------------------------------------#
    #         #   获得学习率下降的公式
    #         #---------------------------------------#
    #         lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
            
    #         model.Unfreeze_backbone()

    #         epoch_step      = num_train // batch_size
    #         epoch_step_val  = num_val // batch_size

    #         if epoch_step == 0 or epoch_step_val == 0:
    #             raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

    #         if distributed:
    #             batch_size = batch_size // ngpus_per_node

    #         gen             = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
    #                                 drop_last=True, collate_fn=detection_collate, sampler=train_sampler)
    #         gen_val         = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
    #                                 drop_last=True, collate_fn=detection_collate, sampler=val_sampler)

    #         UnFreeze_flag = True

    #     if distributed:
    #         train_sampler.set_epoch(epoch)
            
    #     set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        
    #     fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)

    # if local_rank == 0:
    #     loss_history.writer.close()


# 加载配置文件，进行训练
import json
# 1. 加载配置文件  dataset/tasks/task_b6cb9a00-d868-11ee-bb66-552f1d10f6a7_0.json
with open('dataset/tasks/task_b6cb9a00-d868-11ee-bb66-552f1d10f6a7_0.json', 'r') as f:
    data = json.load(f)
    print(data)




# 2. 读取配置文件中的参数
# 3. 训练模型
