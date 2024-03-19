import torch
import torch.nn as nn
import torch.nn.functional as F



class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='none'):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        return F.cross_entropy(input, target, weight=self.weight, reduction=self.reduction)
    

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='none'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        return focal_loss


class SoftBootstrappingLoss(nn.Module):
    """
    ``Loss(t, p) = - (beta * t + (1 - beta) * p) * log(p)``

    Args:
        beta (float): bootstrap parameter. Default, 0.95
        reduce (bool): computes mean of the loss. Default, True.
        as_pseudo_label (bool): Stop gradient propagation for the term ``(1 - beta) * p``.
            Can be interpreted as pseudo-label.
    """
    def __init__(self, beta=0.95, reduce=True, as_pseudo_label=True):
        super(SoftBootstrappingLoss, self).__init__()
        self.beta = beta
        self.reduce = reduce
        self.as_pseudo_label = as_pseudo_label

    def forward(self, y_pred, y):
        # cross_entropy = - t * log(p)
        beta_xentropy = self.beta * F.cross_entropy(y_pred, y, reduction='none')

        y_pred_a = y_pred.detach() if self.as_pseudo_label else y_pred
        # second term = - (1 - beta) * p * log(p)
        bootstrap = - (1.0 - self.beta) * torch.sum(F.softmax(y_pred_a, dim=1) * F.log_softmax(y_pred, dim=1), dim=1)

        if False:
            return torch.mean(beta_xentropy + bootstrap)
        return beta_xentropy + bootstrap
    

class LabelSmoothingLoss(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='none'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = (-log_preds.sum(dim=-1)).mean()
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        ls = self.epsilon * (loss / n) + (1 - self.epsilon) * nll
        return ls
    

class LogitAdjustmentLoss(nn.Module):

    def __init__(self, cls_num_list, tau = 1.0):
        super().__init__()
        base_probs = [x/sum(cls_num_list) for x in cls_num_list]
        base_probs = torch.tensor(base_probs)
        scaled_class_weights = tau * torch.log(base_probs + 1e-12)
        scaled_class_weights = scaled_class_weights.reshape(1,-1) #[1,classnum]
        self.scaled_class_weights = scaled_class_weights.float().cuda()
        
    def forward(self, x, target):
        x += self.scaled_class_weights
        return F.cross_entropy(x, target, reduction='none')