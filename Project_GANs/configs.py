import torch

DATALOADER_CONFIG = {
    "path_paintings": ".\\gan-getting-started\\monet_jpg",
    "path_photos": ".\\gan-getting-started\\photo_jpg",
    "num_workers": 2,
    "pin_memory": torch.cuda.is_available(),
    "persistent_workers": True,
    "sample_size": 5,
    "batch_size": 30,
    "val_batch_size": 30
}

MODEL_CONFIG = {
    "hidden_channels": 64,
    "in_channels": 3,
    "out_channels": 3,
    "optimizer": torch.optim.Adam,
    "lr": 2e-4,
    "betas": (0.5, 0.999),
    "lambda_idt": 0.5,
    "lambda_cycle": (10, 10),  # (M-P-M, P-M-P)
    "buffer_size": 100,
    # total number of epochs
    "num_epochs": 18,
    # number of epochs before starting to decay the learning rate
    "decay_epochs": 18,
    "kernel_size": 4,
    "stride": 2,
    "disc_stride": 1,
    "padding": 1,
    "mean_weight_init": 0.0,
    "std_weight_init": 0.02,
    "generator": "UNetGenerator",
    "discriminator": "Discriminator"
}

TRAIN_CONFIG = {
    "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
    "precision": "16-mixed" if torch.cuda.is_available() else 32,
    "devices": 1,
    "enable_checkpointing": True,
    "logger": False,
    "max_epochs": MODEL_CONFIG["num_epochs"],
    "max_time": {"hours": 4, "minutes": 59},
    "limit_val_batches": 1,
    "limit_train_batches": 1.0,
    "limit_predict_batches": 1.0,
    "num_sanity_val_steps": 0,
    "check_val_every_n_epoch": 6
}
