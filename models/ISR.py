import torch
import torch.nn as nn
import timm
from torchvision import transforms as T
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import csv

model = timm.create_model("swin_base_patch4_window7_224", num_classes=0)
ckpt = torch.load("pretrained/converted_timm_ISR.pt")
model.load_state_dict(ckpt)


class ISR(nn.Module):
    def __init__(self):
        super(ISR, self).__init__()
        self.swin_transformer = timm.create_model(
            "swin_base_patch4_window7_224", num_classes=0
        )
        self.swin_transformer.load_state_dict(
            torch.load("pretrained/converted_timm_ISR.pt")
        )

    def forward(self, x):
        x = self.swin_transformer(x)
        return x


print(type(model))
# Check if the model is an instance of torch.nn.Module
print(isinstance(model, torch.nn.Module))  # This should print: True
