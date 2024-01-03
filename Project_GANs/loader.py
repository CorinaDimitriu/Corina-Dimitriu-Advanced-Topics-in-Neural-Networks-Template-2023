import pytorch_lightning as L
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader

import utils
from dataset import CustomDataset


class CustomLoader(L.LightningDataModule):
    def __init__(self, config, transform):
        super().__init__()
        self.config = config
        self.transform = transform

    def setup(self, stage):
        path_photos = self.config['path_photos']
        if stage == "fit":
            path_paintings = self.config['path_paintings']
            self.paintings = CustomDataset(path_paintings, utils.build_dataset, self.transform)
            self.photos = CustomDataset(path_photos, utils.build_dataset, self.transform)
        self.valid_photos = CustomDataset(path_photos, utils.build_dataset, self.transform, training=False)

    def train_dataloader(self):
        loader_monet = DataLoader(self.paintings, shuffle=True, drop_last=True,
                                  batch_size=self.config['batch_size'],
                                  pin_memory=self.config['pin_memory'],
                                  persistent_workers=self.config['persistent_workers'],
                                  num_workers=self.config['num_workers'])
        loader_photo = DataLoader(self.photos, shuffle=True, drop_last=True,
                                  batch_size=self.config['batch_size'],
                                  pin_memory=self.config['pin_memory'],
                                  persistent_workers=self.config['persistent_workers'],
                                  num_workers=self.config['num_workers'])
        loaders = {"monet": loader_monet, "photo": loader_photo}
        return CombinedLoader(loaders, mode="max_size_cycle")

    def val_dataloader(self):
        return DataLoader(self.valid_photos, batch_size=self.config['sample_size'],
                          shuffle=False, drop_last=False, persistent_workers=self.config['persistent_workers'],
                          num_workers=self.config['num_workers'], pin_memory=self.config['pin_memory'])

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return DataLoader(self.valid_photos, batch_size=self.config['val_batch_size'],
                          shuffle=False, drop_last=False, persistent_workers=self.config['persistent_workers'],
                          num_workers=self.config['num_workers'], pin_memory=self.config['pin_memory'])
