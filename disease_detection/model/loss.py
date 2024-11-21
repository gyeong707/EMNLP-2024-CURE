import torch.nn as nn
import torch
from torch.nn import functional as F

# Multi-Class Classification
def cross_entropy_loss(output, target):
    criterion = nn.CrossEntropyLoss()
    return criterion(output, target)

# Multi-Label Classification
def bce_with_logits(output, target):
    criterion = nn.BCEWithLogitsLoss()
    return criterion(output, target)

# Multi-Label Classification without Sigmoid Function
def bce_loss(output, target):
    criterion = nn.BCELoss()
    return criterion(output, target)
