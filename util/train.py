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
from util.utils import compute_label_difference
from config import Config

cf = Config()
from tqdm import tqdm

import wandb


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True


def train(
    isr_model,
    h2l_model,
    dl_train,
    dl_val,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    device,
):

    # Initialize wandb
    wandb.init(project="estimate-carrying-baggage")
    # Log hyperparameters
    wandb.config.update(cf.wandb_config)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        if epoch < cf.train_config.isr_freeze_epoch:
            freeze_model(isr_model)
            unfreeze_model(h2l_model)
        else:
            unfreeze_model(isr_model)
            unfreeze_model(h2l_model)
            print("isr_model unfreezed")

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        print("starting training...")
        isr_model.train()
        h2l_model.train()
        for batch in tqdm(iter(dl_train)):
            optimizer.zero_grad()

            img1, img2, label1, label2 = [v.to(device) for v in batch]

            # calculate more less equal
            with torch.no_grad():
                result = compute_label_difference(label1, label2)
            # print("result", result)

            ############# Forward pass #############
            patch_emb1 = isr_model(img1)
            patch_emb2 = isr_model(img2)
            inputs = torch.cat((patch_emb1, patch_emb2), dim=1)

            classy = h2l_model(inputs)

            loss = criterion(classy, result)

            loss.backward()
            optimizer.step()

            # Accumulate metrics
            total_loss += loss.item() * img1.size(0)
            total_samples += img1.size(0)

            # Calculate accuracy
            _, predicted = torch.max(classy.data, 1)  # value, class_index
            # print(predicted)
            total_correct += (predicted == compute_label_difference(label1, label2)).sum().item()

        # Calculate average training loss and accuracy
        avg_train_loss = total_loss / total_samples
        accuracy_train = total_correct / total_samples

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {accuracy_train:.4f}")
        # Update scheduler
        scheduler.step()

        print("starting validation...")
        isr_model.eval()
        h2l_model.eval()
        with torch.no_grad():
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for batch in tqdm(dl_val):
                img1, img2, label1, label2 = [b.to(device) for b in batch]

                # Forward pass
                patch_emb1 = isr_model(img1)
                patch_emb2 = isr_model(img2)
                inputs = torch.cat((patch_emb1, patch_emb2), dim=1)
                classy = h2l_model(inputs)

                results = compute_label_difference(label1, label2)
                loss = criterion(classy, results).to(device)

                # Accumulate metrics
                total_loss += loss.item() * img1.size(0)
                total_samples += img1.size(0)

                # Calculate accuracy
                _, predicted = torch.max(classy.data, 1)
                total_correct += (predicted == compute_label_difference(label1, label2)).sum().item()

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
                }
            )

            print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy_val:.4f}")

            # Save the model if validation loss decreases
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(
                    isr_model.state_dict(),
                    f"results/best_isr_model_e{epoch+1}_val_loss_{avg_val_loss:.3f}_acc_{accuracy_val:.3f}.pth",
                )
                torch.save(
                    h2l_model.state_dict(),
                    f"results/best_h2l_model_e{epoch+1}_val_loss_{avg_val_loss:.3f}_acc_{accuracy_val:.3f}.pth",
                )
                print(f"Epoch {epoch+1} Checkpoint saved.\navg_loss: {avg_val_loss}  val_accuracy: {accuracy_val}")

    print("Training complete.")
