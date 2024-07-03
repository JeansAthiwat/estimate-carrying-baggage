import torch
import torch.nn as nn
import timm
from torchvision import transforms as T
import torch.nn.functional as F
import os


INPUT_IMAGES_SIZE = (224, 224)
ROOT_DIR = "/home/jeans/internship/resources/datasets/mon"
TRANSFORM = T.Compose(
    [
        T.Resize(INPUT_IMAGES_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class Data_Processor(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.transform = T.Compose(
            [
                T.Resize((self.height, self.width)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, img):
        return self.transform(img).unsqueeze(0)


import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import csv


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
                img1_path = os.path.join(ROOT_DIR, img1_path)
                img2_path = os.path.join(ROOT_DIR, img2_path)
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


# Example usage:
if __name__ == "__main__":
    # Path to your varied_image_pairs.csv file
    csv_file = "varied_image_pairs.csv"

    # Create dataset instance
    dataset = PersonWithBaggageDataset(csv_file, ROOT_DIR)

    # # Example of accessing data from the dataset
    # for i in range(len(dataset)):
    #     img1, img2, label1, label2 = dataset[i]
    #     # Use img1, img2, label1, label2 as needed
    #     print(f"Pair {i+1}: Label1={label1}, Label2={label2}")

    # Example of using with DataLoader
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    for batch_index, (imgs1, imgs2, labels1, labels2) in enumerate(dataloader):
        print(batch_index, imgs1.shape, imgs2.shape, labels1.shape, labels2.shape)
