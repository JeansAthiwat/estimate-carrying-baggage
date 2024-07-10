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


class ISR_Siamese(nn.Module):
    def __init__(self, *, cut_last_avgpool, num_classes):
        super(ISR_Siamese, self).__init__()
        self.cut_last_avgpool = cut_last_avgpool
        self.num_classes = num_classes
        self.swin_transformer = timm.create_model("swin_base_patch4_window7_224", num_classes=0)
        self.classification_head = nn.Sequential(
            nn.Linear(1024 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, input1, input2):
        emb1 = self.swin_transformer(input1)
        emb2 = self.swin_transformer(input2)
        out = self.classification_head(torch.cat((emb1, emb2), dim=1))
        return out
