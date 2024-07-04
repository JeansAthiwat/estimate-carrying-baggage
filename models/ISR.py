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


class ISR(nn.Module):
    def __init__(self, cut_mlp_head=True):
        super(ISR, self).__init__()
        self.swin_transformer = timm.create_model(
            "swin_base_patch4_window7_224", num_classes=0
        )
        # Optionally cut MLP head and remove the last norm, avgpool, and head layers
        # if cut_mlp_head:
        #     with torch.no_grad():
        #         del self.swin_transformer.layers[3].blocks[-1].mlp
        #         del self.swin_transformer.norm
        #         del self.swin_transformer.avgpool
        #         del self.swin_transformer.head
        # else:
        #     with torch.no_grad():
        #         del self.swin_transformer.norm
        #         del self.swin_transformer.avgpool
        #         del self.swin_transformer.head

    def forward(self, x):
        x = self.swin_transformer.patch_embed(x)
        x = self.swin_transformer.pos_drop(x)
        x = self.swin_transformer.layers(x)
        x = self.swin_transformer.norm(x)
        x = self.swin_transformer.head(x)
        return x


isr = ISR(cut_mlp_head=False).to(device)
breakpoint()
# Print summary of the model
summary(isr, input_size=(4, 3, 224, 224))

# Optional: Load checkpoint if needed
# ckpt = torch.load("pretrained/isr/converted_timm_ISR.pt", map_location=device)
# isr.swin_transformer.load_state_dict(ckpt, strict=False)

# Optional: Define a breakpoint for debugging
