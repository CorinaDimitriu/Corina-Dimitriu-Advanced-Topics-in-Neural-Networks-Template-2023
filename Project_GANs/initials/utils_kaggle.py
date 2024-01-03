import os
import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid


def plot_images(t_image, lines, title=""):
    t_image = t_image.detach().cpu() * 0.5 + 0.5
    grid = make_grid(t_image, nrow=lines).permute(1, 2, 0)
    plt.figure(figsize=(10, 7))
    plt.imshow(grid)
    plt.axis("off")
    plt.title(title)
    plt.show()


DEBUG = False
DM_CONFIG = {
    "monet_origin": os.path.join("../gan-getting-started/monet_jpg", "*.jpg"),
    "photos_origin": os.path.join("../gan-getting-started/photo_jpg", "*.jpg"),

    "loader_config": {
        "num_workers": 2,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": True,
    },
    "sample_size": 5,
    "batch_size": 30 if not DEBUG else 1,
}
MODEL_CONFIG = {
    "hidden_channels": 64,
    "optimizer": torch.optim.Adam,
    "lr": 2e-4,
    "betas": (0.5, 0.999),
    "lambda_idt": 0.5,
    "lambda_cycle": (10, 10),  # (M-P-M, P-M-P)
    "buffer_size": 100,
    # total number of epochs
    "num_epochs": 18 if not DEBUG else 2,
    # number of epochs before starting to decay the learning rate
    "decay_epochs": 18 if not DEBUG else 1
}
TRAIN_CONFIG = {
    "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
    "precision": "16-mixed" if torch.cuda.is_available() else 32,
    "devices": 1,
    "enable_checkpointing": True,
    "logger": False,
    "max_epochs": MODEL_CONFIG["num_epochs"],
    "limit_train_batches": 1.0 if not DEBUG else 2,
    "limit_predict_batches": 1.0 if not DEBUG else 5,
    "max_time": {"hours": 4, "minutes": 59},
    "limit_val_batches": 1,
    # "limit_test_batches": 5,
    "num_sanity_val_steps": 0,
    "check_val_every_n_epoch": 1 if not DEBUG else 1
}
