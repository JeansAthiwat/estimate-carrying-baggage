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


INPUT_IMAGES_SIZE = (224, 224)
CSV_FILE = "manifest/varied_image_pairs.csv"
ROOT_DIR = "/home/jeans/internship/resources/datasets/mon"


def train(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for img1, img2, label1, label2 in dataloader:
            ############# Forward pass #############
            # pass img throught isr
            output1 = model(img1)
            output2 = model(img2)
            # pack 2 images together and pass through h2l (classification modded)

            # Compute loss
            # loss1 = criterion(output1, label1)
            # loss2 = criterion(output2, label2)
            # loss = loss1 + loss2

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print training statistics
            # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Optionally: validate the model on a validation set and log metrics
        # scheduler.step()

    print("Training complete.")


if __name__ == "__main__":
    # Create dataset instance
    dataset = PersonWithBaggageDataset(csv_file=CSV_FILE, root_dir=ROOT_DIR)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Initialize model
    model = ISR()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # or any other appropriate loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Set number of epochs
    num_epochs = 10

    # Call the training function
    train(model, dataloader, criterion, optimizer, num_epochs)

    # Optionally: save the trained model
    # torch.save(model.state_dict(), 'model.pth')

    print("Data loading and model testing complete.")
