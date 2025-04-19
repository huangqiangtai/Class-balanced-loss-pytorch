"""Pytorch implementation of Class-Balanced-Loss
   Reference: "Class-Balanced Loss Based on Effective Number of Samples" 
   Authors: Yin Cui and
               Menglin Jia and
               Tsung Yi Lin and
               Yang Song and
               Serge J. Belongie
   https://arxiv.org/abs/1901.05555, CVPR'19.
"""


import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class FocalLoss(nn.Module):
    """Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      gamma: A float scalar modulating loss from hard and easy examples.
      reduction: string. One of "mean", "sum".
    """
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, labels, sample_weight):
        """Compute the focal loss between `logits` and the ground truth `labels`.
            Args:
              logits: A float tensor of size [batch, num_classes].
              labels: A float tensor of size [batch, num_classes].
              sample_weight: A float tensor of size [batch_size, num_classes]
                specifying per-example weight for balanced cross entropy.

            Returns:
              focal_loss: A float32 scalar representing normalized total loss.
            """
        BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")
        if gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-self.gamma * labels * logits - self.gamma * torch.log(1 +
                torch.exp(-1.0 * logits)))

        loss = sample_weight * (modulator * BCLoss)
        loss = loss.sum(dim=1)

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

class WeightedBCELossWithLogit(nn.Module):
    """
   the weighted Binary Cross Entropy loss
    Args:
        prob_func: string. One of "sigmoid", "softmax".
        reduction: string. One of "mean", "sum".
    """
    def __init__(self, prob_func, reduction='mean'):
        super(WeightedBCELossWithLogit, self).__init__()
        self.reduction = reduction
        self.prob_func = prob_func
    def forward(self, logits, labels, sample_weight):
        """Compute the Binary Cross Entropy loss between `logits` and the ground truth `labels`.
            Args:
              logits: A float tensor of size [batch, num_classes].
              labels: A float tensor of size [batch, num_classes].
              sample_weight: A float tensor of size [batch_size, num_classes]
                specifying per-example weight for balanced cross entropy.
            Returns:
              loss: A float32 scalar representing normalized total loss.
            """
        if self.prob_func == 'sigmoid':
            probs = torch.sigmoid(logits)
        elif self.prob_func == 'softmax':
            probs = torch.softmax(logits, dim=1)
        loss = F.binary_cross_entropy(input=probs, target=labels, weight=sample_weight, reduction=self.reduction)
        return loss


class ClassBalancedLoss(nn.Module):
    """Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.

    Args:
        samples_num_per_cls: A python list of size [classes_num].
            Each element represents the number of samples of one class in the dataset.
        classes_num: total number of classes. int
        loss_type: string. One of "sigmoid", "focal", "softmax".
        beta: float. Hyperparameter for Class balanced loss.
        gamma: float. Hyperparameter for Focal loss.

    Returns:
        cb_loss: A float tensor representing class balanced loss
    """
    def __init__(self, samples_num_per_cls, classes_num, loss_type, beta, gamma):
        super(ClassBalancedLoss, self).__init__()
        self.classes_num = classes_num
        effective_num_per_cls = 1.0 - np.power(beta, samples_num_per_cls)
        weights = (1.0 - beta) / np.array(effective_num_per_cls)
        self.weights_per_cls = weights / np.sum(weights) * classes_num

        if loss_type == "focal":
            self.loss_func = FocalLoss(gamma, reduction='mean')
        elif loss_type == "sigmoid":
            self.loss_func = WeightedBCELossWithLogit('sigmoid', reduction='mean')
        elif loss_type == "softmax":
            self.loss_func = WeightedBCELossWithLogit('softmax', reduction='mean')

    def forward(self, logits, labels):
        """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Args:
          labels: A int tensor of size [batch].
          logits: A float tensor of size [batch, no_of_classes].
        """
        sample_weights = torch.tensor([self.weights_per_cls[i] for i in labels])
        sample_weights = sample_weights.unsqueeze(1).repeat(1, self.classes_num)
        sample_weights = sample_weights.to(labels.device)
        labels_one_hot = F.one_hot(labels, self.classes_num).float()
        cb_loss = self.loss_func(logits, labels_one_hot, sample_weights)

        return cb_loss

if __name__ == '__main__':
    num_classes = 5
    logits = torch.rand(10,num_classes).float()
    labels = torch.randint(0,num_classes, size = (10,))
    beta = 0.9999
    gamma = 2.0
    samples_per_cls = [2,3,1,2,2]
    loss_func = ClassBalancedLoss(samples_per_cls, num_classes, loss_type="focal", beta=beta, gamma=gamma)
    cb_loss = loss_func(logits, labels)
    print(cb_loss)
