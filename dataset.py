import torch
import torch.nn as nn
import timm
from torchvision.transforms import v2 as T
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import csv
import cv2
import numpy as np

INPUT_IMAGES_SIZE = (224, 224)
CSV_FILE = "manifest/dummy-set-s/image_pairs_train.csv"
ROOT_DIR = "/home/jeans/internship/resources/datasets/mon/train"

# Define the transformations with RandomApply
TRANSFORM = T.Compose(
    [
        T.Resize(INPUT_IMAGES_SIZE),
        T.RandomApply(
            [
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
                # T.RandomRotation(degrees=(-20, 20)),
                T.RandomPerspective(distortion_scale=0.15, p=0.6),
                # T.GaussianNoise(mean=0.0, sigma=0.1, clip=True),
                T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            ],
            p=0.9,
        ),  # Apply these transformations with a probability of 0.5
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class PersonWithBaggageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=TRANSFORM):
        self.transform = transform
        self.images = []
        self.labels = []

        # Read CSV file and populate images and labels
        with open(csv_file, "r") as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # Skip header
            for row in csvreader:
                img1_path, img2_path, label1, label2 = row
                img1_path = os.path.join(root_dir, img1_path)
                img2_path = os.path.join(root_dir, img2_path)
                self.images.append([img1_path, img2_path])
                self.labels.append([int(label1), int(label2)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img1_path = self.images[idx][0]
        img2_path = self.images[idx][1]
        label1 = self.labels[idx][0]
        label2 = self.labels[idx][1]

        # Load images
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        # Apply transformations if specified
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label1, label2


def imshow(img, title=None):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    img = np.uint8(img * 255)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if title:
        cv2.imshow(title, img)
    else:
        cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage:
if __name__ == "__main__":
    csv_file = CSV_FILE
    dataset = PersonWithBaggageDataset(csv_file, ROOT_DIR)

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    for batch_index, (imgs1, imgs2, labels1, labels2) in enumerate(dataloader):
        print(batch_index, imgs1.shape, imgs2.shape, labels1.shape, labels2.shape)
        for i in range(len(imgs1)):
            imshow(imgs1[i], title=f'Image 1 - Label: {labels1[i]}')
            imshow(imgs2[i], title=f'Image 2 - Label: {labels2[i]}')
        break  # Display only the first batch for brevity
