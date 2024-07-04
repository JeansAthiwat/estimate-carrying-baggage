import torch
from torchinfo import summary
import torch.nn as nn
import timm
from torchvision import transforms as T
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("pretrained/isr/converted_timm_ISR.pt", map_location=device)
# self.swin_transformer.load_state_dict(ckpt)


class ISR(nn.Module):
    def __init__(self, cut_last_avgpool=True):
        super(ISR, self).__init__()
        self.cut_last_avgpool = cut_last_avgpool
        self.swin_transformer = timm.create_model(
            "swin_base_patch4_window7_224", num_classes=0
        )

    def forward(self, x):
        x = self.swin_transformer.patch_embed(x)
        x = self.swin_transformer.pos_drop(x)
        x = self.swin_transformer.layers(x)
        x = self.swin_transformer.norm(x)

        if not self.cut_last_avgpool:
            x = self.swin_transformer.avgpool(x)

        x = self.swin_transformer.head(x)
        return x
