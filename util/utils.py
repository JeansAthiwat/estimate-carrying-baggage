import torch
from torchvision import transforms as T
import torch.nn.functional as F


def compute_label_difference(label1, label2):
    # Calculate label difference as described in your code
    greater_than = (label1 > label2).float() * 0
    equal_to = (label1 == label2).float() * 1
    less_than = (label1 < label2).float() * 2
    result = (greater_than + equal_to + less_than).type(torch.int64)
    return result
