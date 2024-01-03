from multiprocessing import freeze_support
import pytorch_lightning as L

from initials.cycleGAN_kaggle import CycleGAN
from initials.loader_kaggle import CustomLoader
from initials.utils_kaggle import DM_CONFIG, MODEL_CONFIG, TRAIN_CONFIG

if __name__ == '__main__':
    freeze_support()

    dm = CustomLoader(**DM_CONFIG)
    model = CycleGAN(**MODEL_CONFIG)
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
