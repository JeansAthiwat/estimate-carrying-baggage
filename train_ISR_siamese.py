import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import timm
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from models.ISR_siamese import ISR_Siamese
from dataset import PersonWithBaggageDataset
from config import Config
from util.utils import compute_label_difference, set_parameter_requires_grad

import os
import numpy as np
from tqdm import tqdm
import wandb
import csv

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cf = Config()
cf.train_config.CONTINUE_FROM_CHECKPOINT = True

ISR_CKPT_PATH = "results/isr_siamese/best_isr_model_e15_val_loss_1.021_acc_0.489.pth"

FREEZE_ISR = True
FREEZE_EPOCH = 0


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

    for epoch in range(num_epochs):

        # Freeze or unfreeze swin_transformer
        if epoch < FREEZE_EPOCH:
            set_parameter_requires_grad(isr_model.swin_transformer, False)
            print("Backbone FROZEN")
        else:
            set_parameter_requires_grad(isr_model.swin_transformer, True)
            print("Backbone UNFREEZED!")

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        # print("starting training...")
        isr_model.train()
        for batch in tqdm(iter(dl_train)):
            optimizer.zero_grad()
            img1, img2, label1, label2 = [v.to(device) for v in batch]

            with torch.no_grad():
                y_true = compute_label_difference(label1, label2)
            # print("result", result)

            ############# Forward pass #############
            classy = isr_model(img1, img2)
            loss = criterion(classy, y_true)

            loss.backward()
            optimizer.step()

            # Accumulate metrics
            total_loss += loss.item() * img1.size(0)
            total_samples += img1.size(0)

            # Calculate accuracy
            _, predicted = torch.max(classy.data, 1)
            # print(predicted)
            total_correct += (predicted == y_true).sum().item()

        # Calculate average training loss and accuracy
        avg_train_loss = total_loss / total_samples
        accuracy_train = total_correct / total_samples

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {accuracy_train:.4f}")
        # Update scheduler
        scheduler.step()

        # print("starting validation...")
        isr_model.eval()
        with torch.no_grad():
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for batch in tqdm(dl_val):
                img1, img2, label1, label2 = [b.to(device) for b in batch]

                y_true = compute_label_difference(label1, label2)
                classy = isr_model(img1, img2)
                loss = criterion(classy, y_true)

                # Accumulate metrics
                total_loss += loss.item() * img1.size(0)
                total_samples += img1.size(0)

                # Calculate accuracy
                _, predicted = torch.max(classy.data, 1)
                total_correct += (predicted == y_true).sum().item()

            # Average validation metrics
            avg_val_loss = total_loss / total_samples
            accuracy_val = total_correct / total_samples

            # Log validation metrics to wandb
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "train_accuracy": accuracy_train,
                    "val_loss": avg_val_loss,
                    "val_accuracy": accuracy_val,
                    # "ISR_state_dict": isr_model.state_dict(),
                    "learning_rate": scheduler.get_last_lr()[0],
                }
            )

            print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy_val:.4f}")

            # Save the model if validation loss decreases
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs('results/isr_siamese', exist_ok=True)
                torch.save(
                    isr_model.state_dict(),
                    f"results/isr_siamese/best_isr_model_e{epoch+1}_val_loss_{avg_val_loss:.3f}_acc_{accuracy_val:.3f}.pth",
                )
                print(
                    f"New best loss!\nEpoch {epoch+1} Checkpoint saved.\navg_loss: {avg_val_loss}  val_accuracy: {accuracy_val}\n",
                    ("=" * 60),
                )

    print("=" * 60, "Training complete.", "=" * 60)


##########################################

ds_train = PersonWithBaggageDataset(
    cf.dataset_config.TRAIN_CSV_FILE,
    cf.dataset_config.DATASET_ROOT_DIR,
)
dl_train = DataLoader(
    ds_train,
    batch_size=cf.train_config.batch_size,
    shuffle=True,
    pin_memory=True,
)

ds_val = PersonWithBaggageDataset(
    cf.dataset_config.VAL_CSV_FILE,
    cf.dataset_config.DATASET_ROOT_DIR,
)
dl_val = DataLoader(
    ds_val,
    batch_size=cf.train_config.batch_size,
    shuffle=False,
    pin_memory=True,
)

# Initialize model
siamese_model = ISR_Siamese(cut_last_avgpool=False, num_classes=3)
siamese_model = siamese_model.to(device)

if cf.train_config.CONTINUE_FROM_CHECKPOINT:
    try:
        siamese_model.load_state_dict(torch.load(ISR_CKPT_PATH))
        print("loaded succ")
    except:
        print("ERROR: fail to load model Train H2L From Scratch Instead")
        siamese_model.load_state_dict(torch.load("pretrained/isr/isr_model_weights.pth"))

else:
    print("Train H2L From Scratch")
    siamese_model.swin_transformer.load_state_dict(torch.load("pretrained/isr/converted_timm_ISR.pt"))


# Define loss function
criterion = torch.nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.SGD(
    [
        {'params': siamese_model.swin_transformer.parameters(), 'lr': 8e-5},
        {'params': siamese_model.classification_head.parameters(), 'lr': 8e-4},
    ],
    momentum=0.8,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cf.train_config.num_epochs, verbose=True)

# Set number of epochs
num_epochs = cf.train_config.num_epochs

train(
    siamese_model,
    dl_train,
    dl_val,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    device,
)

# Optionally: save the trained model
torch.save(siamese_model.state_dict(), "ISR_model_last_epoch.pth")

print("Data loading and model testing complete.")
