
import os
import json
import numpy as np
import torch
import torch.nn.functional as F


class CosineSimilarity(torch.nn.Module):
    def __init__(self):
        super(CosineSimilarity, self).__init__()

    def forward(self, x1, x2):
        x2 = x2.t()
        x = x1.mm(x2)
    
        x1_frobenius = x1.norm(dim=1).unsqueeze(0).t()
        x2_frobenins = x2.norm(dim=0).unsqueeze(0)
        x_frobenins = x1_frobenius.mm(x2_frobenins)
    
        final = x.mul(1/x_frobenins)
        return final

device = torch.device("cuda")
model = CosineSimilarity().to(device)



def compute_6td(tensors):
    tensors = torch.tensor([x.item() for x in tensors])
    tensors_avg = tensors.mean()
    tensors_var = tensors.var()
    tensors_var_more2 = torch.sum((tensors - tensors_avg) > 2*tensors_var).item()/len(tensors)
    tensors_var_less2 = torch.sum((tensors - tensors_avg) < -2*tensors_var).item()/len(tensors)
    tensors_var_more3 = torch.sum((tensors - tensors_avg) > 3*tensors_var).item()/len(tensors)
    tensors_var_less3 = torch.sum((tensors - tensors_avg) < -3*tensors_var).item()/len(tensors)
    tensors_avg = tensors_avg.item()
    tensors_var = tensors_var.item()
    return tensors_avg,tensors_var,tensors_var_more2,tensors_var_less2,tensors_var_more3,tensors_var_less3


def generate_task_td_json(loss_list,grad_list,margin_list,possible_list,forget_list,similar_list):
    # 计算 loss、grad、margin、possible、forget

    loss_avg,loss_var,loss_var_more2,loss_var_less2,loss_var_more3,loss_var_less3 = compute_6td(loss_list)
    grad_avg,grad_var,grad_var_more2,grad_var_less2,grad_var_more3,grad_var_less3 = compute_6td(grad_list)
    margin_avg,margin_var,margin_var_more2,margin_var_less2,margin_var_more3,margin_var_less3 = compute_6td(margin_list)
    possible_avg,possible_var,possible_var_more2,possible_var_less2,possible_var_more3,possible_var_less3 = compute_6td(possible_list)
    similar_avg,similar_var,similar_var_more2,similar_var_less2,similar_var_more3,similar_var_less3 = compute_6td(similar_list)
    forget_list_cat = torch.tensor([x.item() for x in forget_list])
    forget_avg,forget_var = forget_list_cat.mean().item() ,forget_list_cat.var().item()

    # 生成json
    result = {"loss_avg":loss_avg,
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
                "similar_avg":similar_avg,
                "similar_var":similar_var,
                "similar_var_more2":similar_var_more2,
                "similar_var_less2":similar_var_less2,
                "similar_var_more3":similar_var_more3,
                "similar_var_less3":similar_var_less3,
                "forget_avg":forget_avg,
                "forget_var":forget_var}
    
    return result


def generate_class_td_json(loss_list,grad_list,margin_list,possible_list,forget_list,similar_list,targets_list):
    # 计算 loss、grad、margin、possible、forget
    class_list_json = []
    class_list = [int(x) for x in list(set(targets_list))]
    class_list.sort()
    
    for class_id in class_list:
        loss_list_class = [loss_list[i] for i in range(len(loss_list)) if targets_list[i] == class_id]
        grad_list_class = [grad_list[i] for i in range(len(grad_list)) if targets_list[i] == class_id]
        margin_list_class = [margin_list[i] for i in range(len(margin_list)) if targets_list[i] == class_id]
        possible_list_class = [possible_list[i] for i in range(len(possible_list)) if targets_list[i] == class_id]
        forget_list_class = [forget_list[i] for i in range(len(forget_list)) if targets_list[i] == class_id]
        similar_list_class = [similar_list[i] for i in range(len(similar_list)) if targets_list[i] == class_id]

        loss_avg,loss_var,loss_var_more2,loss_var_less2,loss_var_more3,loss_var_less3 = compute_6td(loss_list_class)
        grad_avg,grad_var,grad_var_more2,grad_var_less2,grad_var_more3,grad_var_less3 = compute_6td(grad_list_class)
        margin_avg,margin_var,margin_var_more2,margin_var_less2,margin_var_more3,margin_var_less3 = compute_6td(margin_list_class)
        possible_avg,possible_var,possible_var_more2,possible_var_less2,possible_var_more3,possible_var_less3 = compute_6td(possible_list_class)
        similar_avg,similar_var,similar_var_more2,similar_var_less2,similar_var_more3,similar_var_less3 = compute_6td(similar_list_class)
        forget_list_cat = torch.tensor([x.item() for x in forget_list_class])
        forget_avg,forget_var = forget_list_cat.mean().item() ,forget_list_cat.var().item()

        # 生成json
        result_class = {"loss_avg":loss_avg,
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
                    "similar_avg":similar_avg,
                    "similar_var":similar_var,
                    "similar_var_more2":similar_var_more2,
                    "similar_var_less2":similar_var_less2,
                    "similar_var_more3":similar_var_more3,
                    "similar_var_less3":similar_var_less3,
                    "forget_avg":forget_avg,
                    "forget_var":forget_var}
        class_list_json.append(result_class)
    
    result = dict(zip([f"c_{x}" for x in class_list],class_list_json))
    
    return result

