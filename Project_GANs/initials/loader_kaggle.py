import glob
import pytorch_lightning as L
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader

from dataset_kaggle import CustomDataset
from processing_kaggle import Augmentation


class CustomLoader(L.LightningDataModule):
    def __init__(self, monet_origin, photos_origin,
                 loader_config, sample_size, batch_size):
        super().__init__()
        self.valid_photo = None
        self.train_photo = None
        self.train_monet = None
        self.loader_config = loader_config
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.monet_files = sorted(glob.glob(monet_origin))
        self.photo_files = sorted(glob.glob(photos_origin))
        self.transform = Augmentation()

    def setup(self, stage):
        if stage == "fit":
            self.train_monet = CustomDataset(self.monet_files, self.transform, stage)
            self.train_photo = CustomDataset(self.photo_files, self.transform, stage)
        if stage in ["fit", "test", "predict"]:
            self.valid_photo = CustomDataset(self.photo_files, self.transform, None)

    def train_dataloader(self):
        loader_config = {
            "shuffle": True,
            "drop_last": True,
            "batch_size": self.batch_size,
            **self.loader_config
        }
        loader_monet = DataLoader(self.train_monet, **loader_config)
        loader_photo = DataLoader(self.train_photo, **loader_config)
        loaders = {"monet": loader_monet, "photo": loader_photo}
        return CombinedLoader(loaders, mode="max_size_cycle")

    def val_dataloader(self):
        return DataLoader(
            self.valid_photo,
            batch_size=self.sample_size,
            **self.loader_config
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return DataLoader(
            self.valid_photo,
            batch_size=self.batch_size,
            **self.loader_config
        )
