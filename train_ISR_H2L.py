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

cf = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    ds_train = PersonWithBaggageDataset(
        cf.dataset_config.TRAIN_CSV_FILE,
        os.path.join(cf.dataset_config.DATASET_ROOT_DIR, "train"),
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=cf.train_config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
    )

    ds_val = PersonWithBaggageDataset(
        cf.dataset_config.VAL_CSV_FILE,
        os.path.join(cf.dataset_config.DATASET_ROOT_DIR, "val"),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=cf.train_config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
    )

    # Initialize model
    isr_model = ISR()
    isr_model = isr_model.to(device)

    h2l_model = ViT_face_model(**cf.model_config.VIT_face_model_params)
    h2l_model = h2l_model.to(device)

    if cf.train_config.CONTINUE_FROM_CHECKPOINT:
        try:
            isr_model.load_state_dict(
                torch.load("results/best_h2l_model_epoch_18_val_loss_0.1177.pth"),
                strict=False,
            )
            h2l_model.load_state_dict(
                torch.load("results/best_isr_model_epoch_18_val_loss_0.1177.pth"),
                strict=False,
            )
            print("loaded succ")
        except:
            print("ERROR: fail to load model")
    else:
        isr_model.load_state_dict(torch.load("pretrained/isr/isr_model_weights.pth"))
        print("Train H2L From Scratch")

    # Define loss function
    criterion = F.cross_entropy

    # Define separate optimizers for each model
    optimizer_h2l = torch.optim.Adam(
        h2l_model.parameters(), lr=cf.train_config.learning_rate_h2l
    )
    optimizer_isr = torch.optim.Adam(
        isr_model.parameters(), lr=cf.train_config.learning_rate_isr
    )

    # Define separate schedulers for each model
    scheduler_h2l = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_h2l, mode="min", factor=0.1, patience=2, verbose=True
    )
    scheduler_isr = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_isr, mode="min", factor=0.1, patience=2, verbose=True
    )

    # Set number of epochs
    num_epochs = cf.train_config.num_epochs

    train(
        isr_model,
        h2l_model,
        dl_train,
        dl_val,
        criterion,
        optimizer_h2l,
        optimizer_isr,
        scheduler_h2l,
        scheduler_isr,
        num_epochs,
        device,
    )

    # Optionally: save the trained model
    torch.save(isr_model.state_dict(), "isr_model_last_epoch.pth")
    torch.save(h2l_model.state_dict(), "h2l_model_last_epoch.pth")

    print("Data loading and model testing complete.")
