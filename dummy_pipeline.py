import torch
import torch.nn as nn
import timm
from torchvision import transforms as T
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import csv

from dataset import PersonWithBaggageDataset
from models.ISR import ISR
from models.H2L import ViT_face_model, ArcFace
from util.utils import compute_label_difference
from config import Config

cf = Config()
from tqdm import tqdm
from util.train import train

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    ds_train = PersonWithBaggageDataset(
        cf.TRAIN_CSV_FILE, os.path.join(cf.DATASET_ROOT_DIR, "train")
    )
    dl_train = DataLoader(
        ds_train, batch_size=8, shuffle=True, pin_memory=True, num_workers=1
    )

    ds_val = PersonWithBaggageDataset(
        cf.VAL_CSV_FILE, os.path.join(cf.DATASET_ROOT_DIR, "val")
    )
    dl_val = DataLoader(
        ds_val, batch_size=8, shuffle=True, pin_memory=True, num_workers=1
    )

    # Initialize model
    isr_model = ISR()

    h2l_model = ViT_face_model(**cf.VIT_face_model_params)

    if cf.CONTINUE_FROM_CHECKPOINT:
        isr_model.load_state_dict(
            torch.load("results/best_h2l_model_epoch_18_val_loss_0.1177.pth"),
            strict=True,
        )
        h2l_model.load_state_dict(
            torch.load("results/best_isr_model_epoch_18_val_loss_0.1177.pth"),
            strict=True,
        )
        print("loaded succ")
    else:
        isr_model.load_state_dict(torch.load("pretrained/isr/isr_model_weights.pth"))
        print("ERROR: load state dict failed")

    isr_model = isr_model.to(device)
    h2l_model = h2l_model.to(device)
    # Define loss function and optimizer
    criterion = F.cross_entropy

    params = list(isr_model.parameters()) + list(h2l_model.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2, verbose=True
    )
    # Set number of epochs
    num_epochs = 10

    train(
        isr_model,
        h2l_model,
        dl_train,
        dl_val,
        criterion,
        optimizer,
        scheduler,
        num_epochs,
        device,
    )

    # Optionally: save the trained model
    torch.save(isr_model.state_dict(), "isr_model_last_epoch.pth")
    torch.save(h2l_model.state_dict(), "h2l_model_last_epoch.pth")

    print("Data loading and model testing complete.")
