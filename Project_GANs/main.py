import os
import shutil
from multiprocessing import freeze_support
import pytorch_lightning as L
from torchvision.utils import save_image

from cycleGAN import CycleGAN
from loader import CustomLoader
from configs import DATALOADER_CONFIG, MODEL_CONFIG, TRAIN_CONFIG
from processing import Augmentation

if __name__ == '__main__':
    freeze_support()

    dm = CustomLoader(config=DATALOADER_CONFIG, transform=Augmentation())
    model = CycleGAN(config=MODEL_CONFIG)
    trainer = L.Trainer(**TRAIN_CONFIG)
    trainer.fit(model, datamodule=dm)
    # predictions = trainer.predict(model, datamodule=dm)
    # os.makedirs("../images", exist_ok=True)
    # idx = 0
    # for tensor in predictions:
    #     for monet in tensor:
    #         save_image(
    #             monet.float().squeeze() * 0.5 + 0.5,
    #             fp=f"../images/{idx}.jpg",
    #         )
    #         idx += 1
    #
    # shutil.make_archive(".\\images", "zip", ".")
