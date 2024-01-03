from torch.utils.data import Dataset
from torchvision.io import read_image


class CustomDataset(Dataset):
    def __init__(self, files, transform, step):
        self.files = files
        self.transform = transform
        self.step = step

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename = self.files[index]
        photo = read_image(filename) / 255.0  # basic normalization
        return self.transform(photo, step=self.step)
