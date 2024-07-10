import torch
from torchvision import transforms as T
import torch.nn.functional as F


def compute_label_difference(label1, label2):
    greater_than = (label1 > label2).float() * 0
    equal_to = (label1 == label2).float() * 1
    less_than = (label1 < label2).float() * 2
    result = (greater_than + equal_to + less_than).type(torch.int64)
    return result


def compute_label_dissimiliarity(label1, label2):
    '''if the label of img1 is the same as img2 -> return 0
    elif the label of img1 is NOT the same as img2 -> return 1'''
    return (not (label1 == label2)).float()


def set_parameter_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad
