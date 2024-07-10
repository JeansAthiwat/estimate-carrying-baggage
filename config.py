import torch
import torch.nn as nn
import timm
from torchvision import transforms as T
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import csv


class ENV_Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DATASET_Config:
    def __init__(self):
        self.DATASET_ROOT_DIR = "/mnt/c/OxygenAi/resources/human_with_bag/ctw_re_uid_2024-07-01-2024-07-01.bag-images/image-samples-by-class/ctw_re_uid_2024-07-01-2024-07-01.bag-images/filtered"  # "/home/jeans/internship/resources/datasets/mon"
        self.DATASET_MANIFEST = 'manifest/intraclass_pair_with_label'  # "manifest/dummy-set-m"
        self.TRAIN_CSV_FILE = f"{self.DATASET_MANIFEST}/image_pairs_train.csv"
        self.VAL_CSV_FILE = f"{self.DATASET_MANIFEST}/image_pairs_val.csv"
        self.TEST_CSV_FILE = f"{self.DATASET_MANIFEST}/image_pairs_test.csv"
        self.INPUT_IMAGES_SIZE = (224, 224)


class TRAIN_Config:
    def __init__(self):
        self.CONTINUE_FROM_CHECKPOINT = False
        self.CKPT_ROOT = None
        self.batch_size = 16
        self.num_epochs = 40
        self.learning_rate_h2l = 1e-5
        self.learning_rate_isr = 1e-5
        self.scheduler_step_size = 5
        self.scheduler_gamma = 0.1
        self.learning_swap_epoch = 10
        self.h2l_learning_epoch = 7
        self.isr_freeze_epoch = 5


class MODEL_Config:
    def __init__(self):
        self.NUM_CLASS = 3
        self.IMG_SIZE = 224
        self.DEPTH_VIT = 1
        self.HEADS = 4
        self.OUT_DIM = 1024

        self.VIT_face_model_params = dict(
            loss_type="ArcFace",
            num_class=self.NUM_CLASS,
            use_cls=False,
            use_face_loss=False,
            no_face_model=False,
            image_size=self.IMG_SIZE,
            patch_size=7,
            ac_patch_size=12,
            pad=4,
            dim=self.OUT_DIM,
            depth=self.DEPTH_VIT,
            heads=self.HEADS,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            out_dim=self.OUT_DIM,
            singleMLP=False,
            remove_sep=False,
        )


class Config:
    def __init__(self):
        self.env_config = ENV_Config()
        self.dataset_config = DATASET_Config()
        self.train_config = TRAIN_Config()
        self.model_config = MODEL_Config()

        self.wandb_config = {
            "input_image_size": self.dataset_config.INPUT_IMAGES_SIZE,
            "train_csv_file": self.dataset_config.TRAIN_CSV_FILE,
            "val_csv_file": self.dataset_config.VAL_CSV_FILE,
            "test_csv_file": self.dataset_config.TEST_CSV_FILE,
            "root_dir": self.dataset_config.DATASET_ROOT_DIR,
            "continue_from_checkpoint": self.train_config.CONTINUE_FROM_CHECKPOINT,
            "checkpoint_root": self.train_config.CKPT_ROOT,
            "batch_size": self.train_config.batch_size,
            "num_epochs": self.train_config.num_epochs,
            "learning_rate_h2l": self.train_config.learning_rate_h2l,
            "learning_rate_isr": self.train_config.learning_rate_isr,
            "scheduler_step_size": self.train_config.scheduler_step_size,
            "scheduler_gamma": self.train_config.scheduler_gamma,
            "num_classes": self.model_config.NUM_CLASS,
            "image_size": self.model_config.IMG_SIZE,
            "depth_vit": self.model_config.DEPTH_VIT,
            "heads": self.model_config.HEADS,
            "out_dim": self.model_config.OUT_DIM,
        }


cf = Config()
