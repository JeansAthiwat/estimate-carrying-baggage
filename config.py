import torch
import torch.nn as nn
import timm
from torchvision import transforms as T
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import csv

# class ENV_Config:
#     def __init__(self):

# class TRAIN_CONFIG:
#     def __init__(self):


class Config:
    def __init__(self):

        self.DATASET_ROOT_DIR = "/home/jeans/internship/resources/datasets/mon"

        self.DATASET_MANIFEST = "manifest/dummy-set-m"
        self.TRAIN_CSV_FILE = f"{self.DATASET_MANIFEST}/image_pairs_train.csv"
        self.VAL_CSV_FILE = f"{self.DATASET_MANIFEST}/image_pairs_val.csv"
        self.TEST_CSV_FILE = f"{self.DATASET_MANIFEST}/image_pairs_test.csv"

        # Train config

        self.CONTINUE_FROM_CHECKPOINT = True
        self.INPUT_IMAGES_SIZE = (224, 224)

        # Model Config
        self.NUM_CLASS = 3
        self.IMG_SIZE = 224
        self.DEPTH_VIT = 1
        self.HEADS = 4
        self.OUT_DIM = 1024

        self.wandb_config = {
            "input_image_size": self.INPUT_IMAGES_SIZE,
            "train_csv_file": self.TRAIN_CSV_FILE,
            "val_csv_file": self.VAL_CSV_FILE,
            "test_csv_file": self.TEST_CSV_FILE,
            "root_dir": self.DATASET_ROOT_DIR,
            "continue_from_checkpoint": self.CONTINUE_FROM_CHECKPOINT,
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

        self.VIT_face_model_params = dict(
            loss_type="ArcFace",
            num_class=self.NUM_CLASS,
            use_cls=False,
            use_face_loss=False,
            no_face_model=False,
            image_size=224,
            patch_size=7,
            ac_patch_size=12,
            pad=4,
            dim=1024,
            depth=self.DEPTH_VIT,
            heads=self.HEADS,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            out_dim=self.OUT_DIM,
            singleMLP=False,
            remove_sep=False,
        )


self = Config()
