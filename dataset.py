import torch
import torch.nn as nn
import timm
from torchvision import transforms as T
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import csv

INPUT_IMAGES_SIZE = (224, 224)
CSV_FILE = "varied_image_pairs.csv"
ROOT_DIR = "/home/jeans/internship/resources/datasets/mon"

TRANSFORM = T.Compose(
    [
        T.Resize(INPUT_IMAGES_SIZE),
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


# Example usage:
if __name__ == "__main__":
    csv_file = CSV_FILE
    dataset = PersonWithBaggageDataset(csv_file, ROOT_DIR)

    # # Example of accessing data from the dataset
    # for i in range(len(dataset)):
    #     img1, img2, label1, label2 = dataset[i]
    #     # Use img1, img2, label1, label2 as needed
    #     print(f"Pair {i+1}: Label1={label1}, Label2={label2}")

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    for batch_index, (imgs1, imgs2, labels1, labels2) in enumerate(dataloader):
        print(batch_index, imgs1.shape, imgs2.shape, labels1.shape, labels2.shape)
