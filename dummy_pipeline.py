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
from models.H2L import ViT_face_model


INPUT_IMAGES_SIZE = (224, 224)
TRAIN_CSV_FILE = "manifest/set1/image_pairs_train.csv"
VAL_CSV_FILE = "manifest/set1/image_pairs_val.csv"

TEST_CSV_FILE = "manifest/set1/image_pairs_test.csv"

ROOT_DIR = "/home/jeans/internship/resources/datasets/mon"

CONTINUE_FROM_CHECKPOINT = False
CKPT_ROOT = None


def train(model, dl_train, dl_val, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for batch in iter(dl_train):
            img1, img2, label1, label2 = batch
            ############# Forward pass #############
            # pass img throught isr
            patch_emb1 = model(img1)
            patch_emb2 = model(img2)
            inputs = torch.cat((patch_emb1, patch_emb2), dim=1)

            print(patch_emb1.size())
            print(patch_emb1.shape)
            print(type(patch_emb1))
            print(inputs.shape)
            print(type(inputs))
            # pack 2 images together and pass through h2l (classification modded)

            # Compute loss
            loss1 = criterion(patch_emb1, label1)
            loss2 = criterion(patch_emb2, label2)
            loss = loss1 + loss2

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print training statistics
            # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        for img1, img2, label1, label2 in dl_val:
            pass
        # Optionally: validate the model on a validation set and log metrics
        # scheduler.step()

    print("Training complete.")


if __name__ == "__main__":

    ds_train = PersonWithBaggageDataset(TRAIN_CSV_FILE, os.path.join(ROOT_DIR, "train"))
    dl_train = DataLoader(ds_train, batch_size=4, shuffle=False)

    ds_val = PersonWithBaggageDataset(VAL_CSV_FILE, os.path.join(ROOT_DIR, "val"))
    dl_val = DataLoader(ds_val, batch_size=4, shuffle=False)

    # Initialize model
    isr_model = ISR()
    if CONTINUE_FROM_CHECKPOINT:
        pass
    else:
        isr_model.load_state_dict(torch.load("pretrained/isr/isr_model_weights.pth"))

    HEAD_NAME = "Arc"
    NUM_CLASS = 3
    IMG_SIZE = 224
    depth_vit = 1
    heads = 4
    out_dim = 1024

    h2l_model = ViT_face_model(  # THIS IS THE ONE H2L or H2
        loss_type="ArcFace",
        GPU_ID=["0"],
        num_class=NUM_CLASS,
        use_cls=False,
        use_face_loss=True,
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

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # or any other appropriate loss function
    optimizer = torch.optim.Adam(isr_model.parameters(), lr=0.001)

    # Set number of epochs
    num_epochs = 10

    # Call the training function
    train(isr_model, dl_train, dl_val, criterion, optimizer, num_epochs)

    # Optionally: save the trained model
    # torch.save(model.state_dict(), 'model.pth')

    print("Data loading and model testing complete.")
