import torch
import torch.nn as nn
import timm
from torchvision import transforms as T
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import csv

from models.ISR import ISR
from models.H2L import ViT_face_model, ArcFace
from util.utils import compute_label_difference, compute_label_dissimiliarity
from config import Config

cf = Config()
from tqdm import tqdm

import wandb


def train(
    isr_model,
    dl_train,
    dl_val,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    device,
):

    # Initialize wandb
    wandb.init(project="estimate-carrying-baggage-ISR-Siamese-Network")
    # Log hyperparameters
    wandb.config.update(cf.wandb_config)

    best_val_loss = float("inf")
    best_val_acc = float("inf")

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        print("starting training...")
        isr_model.train()
        for batch in tqdm(iter(dl_train)):
            optimizer.zero_grad()
            img1, img2, label1, label2 = [v.to(device) for v in batch]

            with torch.no_grad():
                Y_dissimiliarity = compute_label_dissimiliarity(label1, label2)
            # print("result", result)

            ############# Forward pass #############
            emb1 = isr_model(img1)
            emb2 = isr_model(img2)

            contrastive_loss = criterion(emb1, emb2, Y_dissimiliarity)

            contrastive_loss.backward()
            optimizer.step()

            # Accumulate metrics
            total_loss += contrastive_loss.item() * img1.size(0)
            total_samples += img1.size(0)

    #         # Calculate accuracy
    #         _, predicted = torch.max(classy.data, 1)
    #         # print(predicted)
    #         total_correct += (predicted == compute_label_difference(label1, label2)).sum().item()

    #     # Calculate average training loss and accuracy
    #     avg_train_loss = total_loss / total_samples
    #     accuracy_train = total_correct / total_samples

    #     print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {accuracy_train:.4f}")
    #     # Update scheduler
    #     scheduler.step()

    #     print("starting validation...")
    #     isr_model.eval()
    #     h2l_model.eval()
    #     with torch.no_grad():
    #         total_loss = 0.0
    #         total_correct = 0
    #         total_samples = 0

    #         for batch in tqdm(dl_val):
    #             img1, img2, label1, label2 = [b.to(device) for b in batch]

    #             # Forward pass
    #             patch_emb1 = isr_model(img1)
    #             patch_emb2 = isr_model(img2)
    #             inputs = torch.cat((patch_emb1, patch_emb2), dim=1)
    #             classy = h2l_model(inputs)

    #             results = compute_label_difference(label1, label2)
    #             loss = criterion(classy, results).to(device)

    #             # Accumulate metrics
    #             total_loss += loss.item() * img1.size(0)
    #             total_samples += img1.size(0)

    #             # Calculate accuracy
    #             _, predicted = torch.max(classy.data, 1)
    #             total_correct += (predicted == compute_label_difference(label1, label2)).sum().item()

    #         # Average validation metrics
    #         avg_val_loss = total_loss / total_samples
    #         accuracy_val = total_correct / total_samples

    #         # Log validation metrics to wandb
    #         wandb.log(
    #             {
    #                 "epoch": epoch + 1,
    #                 "train_loss": avg_train_loss,
    #                 "train_accuracy": accuracy_train,
    #                 "val_loss": avg_val_loss,
    #                 "val_accuracy": accuracy_val,
    #             }
    #         )

    #         print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy_val:.4f}")

    #         # Save the model if validation loss decreases
    #         if avg_val_loss < best_val_loss:
    #             best_val_loss = avg_val_loss
    #             torch.save(
    #                 isr_model.state_dict(),
    #                 f"results/best_isr_model_e{epoch+1}_val_loss_{avg_val_loss:.3f}_acc_{accuracy_val:.3f}.pth",
    #             )
    #             torch.save(
    #                 h2l_model.state_dict(),
    #                 f"results/best_h2l_model_e{epoch+1}_val_loss_{avg_val_loss:.3f}_acc_{accuracy_val:.3f}.pth",
    #             )
    #             print(f"Epoch {epoch+1} Checkpoint saved.\navg_loss: {avg_val_loss}  val_accuracy: {accuracy_val}")

    # print("Training complete.")


import torch
import os
from torch.utils.data import DataLoader

from dataset import PersonWithBaggageDataset
from models.ISR import ISR
from models.H2L import ViT_face_model
from config import Config
from util.train import train
from torchvision import transforms as T
import torch.nn.functional as F

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

import numpy as np

torch.manual_seed(42)

cf = Config()

ISR_CKPT_PATH = "results/ISR_isr_frozen.pth"
H2L_CKPT_PATH = "results/H2L_isr_frozen.pth"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds_train = PersonWithBaggageDataset(
    cf.dataset_config.TRAIN_CSV_FILE,
    os.path.join(cf.dataset_config.DATASET_ROOT_DIR, "train"),
)
dl_train = DataLoader(
    ds_train,
    batch_size=cf.train_config.batch_size,
    shuffle=True,
    pin_memory=True,
)

ds_val = PersonWithBaggageDataset(
    cf.dataset_config.VAL_CSV_FILE,
    os.path.join(cf.dataset_config.DATASET_ROOT_DIR, "val"),
)
dl_val = DataLoader(
    ds_val,
    batch_size=cf.train_config.batch_size,
    shuffle=False,
    pin_memory=True,
)

# Initialize model
isr_model = ISR(cut_last_avgpool=True, num_classes=0)
isr_model = isr_model.to(device)

h2l_model = ViT_face_model(**cf.model_config.VIT_face_model_params)
h2l_model = h2l_model.to(device)

if cf.train_config.CONTINUE_FROM_CHECKPOINT:
    try:
        isr_model.load_state_dict(torch.load(ISR_CKPT_PATH))
        h2l_model.load_state_dict(torch.load(H2L_CKPT_PATH))

        print("loaded succ")
    except:
        isr_model.load_state_dict(torch.load("pretrained/isr/isr_model_weights.pth"))
        print("ERROR: fail to load model Train H2L From Scratch Instead")

else:
    isr_model.swin_transformer.load_state_dict(torch.load("pretrained/isr/converted_timm_ISR.pt"))
    print("Train H2L From Scratch")

# Define loss function
criterion = torch.nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.SGD(
    [
        {'params': h2l_model.parameters(), 'lr': cf.train_config.learning_rate_h2l},
        {'params': isr_model.parameters(), 'lr': cf.train_config.learning_rate_isr},
    ],
    momentum=0.9,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cf.train_config.num_epochs, verbose=True)

# Set number of epochs
num_epochs = cf.train_config.num_epochs

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
torch.save(isr_model.state_dict(), "ISR_model_last_epoch.pth")
torch.save(h2l_model.state_dict(), "H2L_model_last_epoch.pth")

print("Data loading and model testing complete.")
