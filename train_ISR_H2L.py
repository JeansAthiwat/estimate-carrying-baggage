import torch
import os
from torch.utils.data import DataLoader

from dataset import PersonWithBaggageDataset
from models.ISR import ISR
from models.H2L import ViT_face_model
from config import Config
from util.train import train
from torchvision import transforms as T
import torch.nn.functional as F

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

import numpy as np

torch.manual_seed(42)

cf = Config()

ISR_CKPT_PATH = "results/ISR_isr_frozen.pth"
H2L_CKPT_PATH = "results/H2L_isr_frozen.pth"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds_train = PersonWithBaggageDataset(
    cf.dataset_config.TRAIN_CSV_FILE,
    os.path.join(cf.dataset_config.DATASET_ROOT_DIR, "train"),
)
dl_train = DataLoader(
    ds_train,
    batch_size=cf.train_config.batch_size,
    shuffle=True,
    pin_memory=True,
)

ds_val = PersonWithBaggageDataset(
    cf.dataset_config.VAL_CSV_FILE,
    os.path.join(cf.dataset_config.DATASET_ROOT_DIR, "val"),
)
dl_val = DataLoader(
    ds_val,
    batch_size=cf.train_config.batch_size,
    shuffle=False,
    pin_memory=True,
)

# Initialize model
isr_model = ISR()
isr_model = isr_model.to(device)

h2l_model = ViT_face_model(**cf.model_config.VIT_face_model_params)
h2l_model = h2l_model.to(device)

if cf.train_config.CONTINUE_FROM_CHECKPOINT:
    try:
        isr_model.load_state_dict(torch.load(ISR_CKPT_PATH))
        h2l_model.load_state_dict(torch.load(H2L_CKPT_PATH))

        print("loaded succ")
    except:
        isr_model.load_state_dict(torch.load("pretrained/isr/isr_model_weights.pth"))
        print("ERROR: fail to load model Train H2L From Scratch Instead")

else:
    isr_model.load_state_dict(torch.load("pretrained/isr/isr_model_weights.pth"))
    print("Train H2L From Scratch")

# Define loss function
criterion = torch.nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.SGD(
    [
        {'params': h2l_model.parameters(), 'lr': cf.train_config.learning_rate_h2l},
        {'params': isr_model.parameters(), 'lr': cf.train_config.learning_rate_isr},
    ],
    momentum=0.9,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cf.train_config.num_epochs, verbose=True)

# Set number of epochs
num_epochs = cf.train_config.num_epochs

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
torch.save(isr_model.state_dict(), "ISR_model_last_epoch.pth")
torch.save(h2l_model.state_dict(), "H2L_model_last_epoch.pth")

print("Data loading and model testing complete.")
