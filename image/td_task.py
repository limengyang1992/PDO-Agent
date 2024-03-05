
import os
import json
import numpy as np
import torch
import torch.nn.functional as F

def compute_6td(tensors):
    tensors_avg = tensors.mean()
    tensors_var = tensors.var()
    tensors_var_more2 = torch.sum((tensors - tensors_avg) > 2*tensors_var).item()/len(tensors)
    tensors_var_less2 = torch.sum((tensors - tensors_avg) < -2*tensors_var).item()/len(tensors)
    tensors_var_more3 = torch.sum((tensors - tensors_avg) > 3*tensors_var).item()/len(tensors)
    tensors_var_less3 = torch.sum((tensors - tensors_avg) < -3*tensors_var).item()/len(tensors)
    tensors_avg = tensors_avg.item()
    tensors_var = tensors_var.item()
    return tensors_avg,tensors_var,tensors_var_more2,tensors_var_less2,tensors_var_more3,tensors_var_less3


# 模拟数据

def generate_td_json(path,save_dir,num_classes=5,num_samples=10):

    forget = np.random.randint(0,2,num_samples)
    label = np.random.randint(0,num_classes,num_samples)
    model_predict = np.random.normal(2,1.3,num_samples*num_classes).reshape(num_samples,num_classes)

    # 计算 loss、grad、margin、possible、forget
    model_possible = F.softmax(torch.tensor(model_predict),dim=1)
    loss = F.cross_entropy(model_possible,torch.tensor(label),reduction='none')
    possible = model_possible[range(num_samples),label]
    grad = (1- model_possible)[range(num_samples),label]
    # margin：计算除了label位置以外的最大值
    model_possible[range(num_samples),label] = 0
    others_max = model_possible.max(dim=1).values
    margin = possible - others_max

    loss_avg,loss_var,loss_var_more2,loss_var_less2,loss_var_more3,loss_var_less3 = compute_6td(loss)
    grad_avg,grad_var,grad_var_more2,grad_var_less2,grad_var_more3,grad_var_less3 = compute_6td(grad)
    margin_avg,margin_var,margin_var_more2,margin_var_less2,margin_var_more3,margin_var_less3 = compute_6td(margin)
    possible_avg,possible_var,possible_var_more2,possible_var_less2,possible_var_more3,possible_var_less3 = compute_6td(possible)
    forget_avg,forget_var = forget.mean(),forget.var()

    # 生成json
    result = {"td":{"loss_avg":loss_avg,
                "loss_var":loss_var,
                "loss_var_more2":loss_var_more2,
                "loss_var_less2":loss_var_less2,
                "loss_var_more3":loss_var_more3,
                "loss_var_less3":loss_var_less3,
                "grad_avg":grad_avg,
                "grad_var":grad_var,
                "grad_var_more2":grad_var_more2,
                "grad_var_less2":grad_var_less2,
                "grad_var_more3":grad_var_more3,
                "grad_var_less3":grad_var_less3,
                "margin_avg":margin_avg,
                "margin_var":margin_var,
                "margin_var_more2":margin_var_more2,
                "margin_var_less2":margin_var_less2,
                "margin_var_more3":margin_var_more3,
                "margin_var_less3":margin_var_less3,
                "possible_avg":possible_avg,
                "possible_var":possible_var,
                "possible_var_more2":possible_var_more2,
                "possible_var_less2":possible_var_less2,
                "possible_var_more3":possible_var_more3,
                "possible_var_less3":possible_var_less3,
                "forget_avg":forget_avg,
                "forget_var":forget_var}}
    
    # 读取path的json，合并result组成新的json，保存到save_dir
    with open(path,"r") as f:
        data = json.load(f)
    new = {**data,**result}
    # 获取path文件名
    file_name = os.path.basename(path)
    with open(os.path.join(save_dir,file_name) ,"w") as f:
        json.dump(new,f,indent=4)



