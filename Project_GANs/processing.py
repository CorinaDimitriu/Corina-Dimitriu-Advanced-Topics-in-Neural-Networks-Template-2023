import torchvision.transforms as T


class Augmentation:
    def __init__(self, load_dim=286, target_dim=256):
        self.transform_train = T.Compose([T.Resize((load_dim, load_dim), antialias=True),
                                          T.RandomCrop((target_dim, target_dim)),
                                          T.RandomHorizontalFlip(p=0.5),
                                          T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                                          ])
        self.transform = T.Resize((target_dim, target_dim), antialias=True)

    def __call__(self, photo, training):
        if training:
            photo = self.transform_train(photo)
        else:
            photo = self.transform(photo)
        return photo * 2 - 1
