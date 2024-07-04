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
from tqdm import tqdm
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_IMAGES_SIZE = (224, 224)
TRAIN_CSV_FILE = "manifest/dummy-set/image_pairs_train.csv"
VAL_CSV_FILE = "manifest/dummy-set/image_pairs_val.csv"
TEST_CSV_FILE = "manifest/dummy-set/image_pairs_test.csv"
ROOT_DIR = "/home/jeans/internship/resources/datasets/mon"

CONTINUE_FROM_CHECKPOINT = False
CKPT_ROOT = None


# Initialize wandb
wandb.init(project="estimate-carrying-baggage")


# Log hyperparameters
wandb.config.update(
    {
        "input_image_size": INPUT_IMAGES_SIZE,
        "train_csv_file": TRAIN_CSV_FILE,
        "val_csv_file": VAL_CSV_FILE,
        "test_csv_file": TEST_CSV_FILE,
        "root_dir": ROOT_DIR,
        "continue_from_checkpoint": CONTINUE_FROM_CHECKPOINT,
        "checkpoint_root": CKPT_ROOT,
        "batch_size": 8,
        "num_epochs": 10,
        "learning_rate": 1e-5,
        "scheduler_step_size": 5,
        "scheduler_gamma": 0.1,
        "num_classes": 3,
        "image_size": 224,
        "depth_vit": 1,
        "heads": 4,
        "out_dim": 1024,
    }
)


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
    for epoch in range(num_epochs):
        isr_model.train()  # Set model to training mode
        h2l_model.train()  # Set model to training mode

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in tqdm(iter(dl_train)):
            img1, img2, label1, label2 = [v.to(device) for v in batch]

            # calculate more less equal
            with torch.no_grad():
                result = compute_label_difference(label1, label2)
            # print("result", result)

            ############# Forward pass #############
            with torch.no_grad():
                patch_emb1 = isr_model(img1)
                patch_emb2 = isr_model(img2)
            inputs = torch.cat((patch_emb1, patch_emb2), dim=1)

            classy = h2l_model(inputs)
            # print("classy shape:", classy)
            # print(classy)

            loss = criterion(classy, result).to(device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            total_loss += loss.item() * img1.size(0)
            total_samples += img1.size(0)

            # Calculate accuracy
            _, predicted = torch.max(classy.data, 1)
            # print(predicted)
            total_correct += (
                (predicted == compute_label_difference(label1, label2)).sum().item()
            )

        # Calculate average training loss and accuracy
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        # Log metrics to wandb
        wandb.log(
            {"epoch": epoch + 1, "train_loss": avg_loss, "train_accuracy": accuracy}
        )

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}"
        )
        # Update scheduler
        scheduler.step()

        print("starting validation...")
        isr_model.eval()  # Set ISR model to evaluation mode
        h2l_model.eval()  # Set H2L model to evaluation mode
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
                total_loss += loss.item()
                total_samples += img1.size(0)

                # Calculate accuracy
                _, predicted = torch.max(classy.data, 1)
                total_correct += (
                    (predicted == compute_label_difference(label1, label2)).sum().item()
                )

            # Average validation metrics
            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples

            # Log validation metrics to wandb
            wandb.log(
                {"epoch": epoch + 1, "val_loss": avg_loss, "val_accuracy": accuracy}
            )

            print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    print(torch.cuda.is_available())

    ds_train = PersonWithBaggageDataset(TRAIN_CSV_FILE, os.path.join(ROOT_DIR, "train"))
    dl_train = DataLoader(
        ds_train, batch_size=8, shuffle=True, pin_memory=True, num_workers=1
    )

    ds_val = PersonWithBaggageDataset(VAL_CSV_FILE, os.path.join(ROOT_DIR, "val"))
    dl_val = DataLoader(
        ds_val, batch_size=8, shuffle=True, pin_memory=True, num_workers=1
    )

    # Initialize model
    isr_model = ISR()
    isr_model = isr_model.to(device)

    if CONTINUE_FROM_CHECKPOINT:
        pass
    else:
        isr_model.load_state_dict(torch.load("pretrained/isr/isr_model_weights.pth"))

    NUM_CLASS = 3
    IMG_SIZE = 224
    depth_vit = 1
    heads = 4
    out_dim = 1024

    h2l_model = ViT_face_model(  # THIS IS THE ONE H2L or H2
        loss_type="ArcFace",
        num_class=NUM_CLASS,
        use_cls=False,
        use_face_loss=False,
        no_face_model=False,
        image_size=224,
        patch_size=7,
        ac_patch_size=12,
        pad=4,
        dim=1024,
        depth=depth_vit,
        heads=heads,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
        out_dim=out_dim,
        singleMLP=False,
        remove_sep=False,
    )
    h2l_model = h2l_model.to(device)
    # Define loss function and optimizer
    criterion = F.cross_entropy

    params = list(isr_model.parameters()) + list(h2l_model.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

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
    # torch.save(model.state_dict(), 'model.pth')

    print("Data loading and model testing complete.")
