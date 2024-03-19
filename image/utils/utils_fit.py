import os
import torch
import torch.nn.functional as F
from torch import nn
from .utils import get_lr
from torch.cuda.amp import autocast
from .TD_task import generate_task_td_json,generate_class_td_json, CosineSimilarity

csmodel = CosineSimilarity()

def fit_one_epoch(model, optimizer, lossfunction, scaler, gen):
    total_loss      = 0
    total_accuracy  = 0
    model.train()
    for iteration, batch in enumerate(gen):
        images, targets = batch
        images  = images.cuda()
        targets = targets.cuda()
             
        optimizer.zero_grad()

        with autocast():
            outputs     = model(images)
            loss_value  = lossfunction(outputs, targets).mean()
        
        scaler.scale(loss_value).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss_value.item()

        with torch.no_grad():
            accuracy = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            total_accuracy += accuracy.item()

    return  total_loss / (iteration + 1), total_accuracy / (iteration + 1),  get_lr(optimizer)



def td_one_epoch(model, loss, optimizer, gen_td):

    loss_list = []
    grad_list = []
    margin_list = []
    possible_list = []
    forget_list = []
    similar_list = []
    targets_list = []
    grad_class_list = []
    model.eval()

    for iteration, batch in enumerate(gen_td):
        images, targets = batch
        targets_list.extend(targets.cpu().numpy())
        images  = images.cuda()
        targets = targets.cuda()

        optimizer.zero_grad()
        outputs     = model(images)
        
        # loss
        loss_value  = loss(outputs, targets)
        loss_list.extend(loss_value.detach().cpu())
        # possible
        possible_value = F.softmax(outputs,dim=1)[range(len(targets)),targets]
        possible_list.extend(possible_value.detach().cpu())
        # forget
        forget_value = (torch.argmax(F.softmax(outputs,dim=1),dim=1) != targets).type(torch.FloatTensor)
        forget_list.extend(forget_value.detach().cpu())
        # margin
        model_possible = F.softmax(outputs,dim=1)
        model_possible[range(len(targets)),targets] = 0
        others_max = model_possible.max(dim=1).values
        margin_value = possible_value - others_max
        margin_list.extend(margin_value.detach().cpu())
        # grad_total
        if iteration == 0:
            grad_compute = torch.autograd.grad(loss_value.mean(), model.parameters(), create_graph=False)
            grad_value = [torch.mean(grad).detach().cpu() for grad in grad_compute]
            grad_list.extend(grad_value)

        # grad_single
        c_grad = torch.norm(1- F.softmax(outputs,dim=1),dim=1)
        grad_class_list.extend(c_grad.detach().cpu())

        # similar
        if outputs.shape[0] >10: 
            sim_matrix = csmodel(outputs, outputs)
            value,_ = torch.topk(sim_matrix, k=6, dim=1, largest=True, sorted=True)
            value2 = (0.5-torch.mean(value[:,1:],dim=1)/2)
            similar_list.extend(value2.detach().cpu())

    # 生成TD数据
    result_task_json = generate_task_td_json(loss_list,grad_list,margin_list,possible_list,forget_list,similar_list)
    result_class_json = generate_class_td_json(loss_list,grad_class_list,margin_list,possible_list,forget_list,similar_list,targets_list)

    return result_task_json,result_class_json


def val_one_epoch(model, gen_val):
    val_loss        = 0
    val_accuracy    = 0
    model.eval()
    for iteration, batch in enumerate(gen_val):
        images, targets = batch
        images  = images.cuda()
        targets = targets.cuda()

        with torch.no_grad():
            outputs     = model(images)
            loss_value  = nn.CrossEntropyLoss()(outputs, targets)
        
        val_loss    += loss_value.item()
        accuracy        = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
        val_accuracy    += accuracy.item()
            
    return val_loss / (iteration + 1),val_accuracy / (iteration + 1)
                
