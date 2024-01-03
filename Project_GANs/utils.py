import glob
import os

from matplotlib import pyplot as plt
from torchvision.utils import make_grid


def build_dataset(root):
    files = sorted(glob.glob(os.path.join(root, "*.jpg")))
    return files, len(files)


def plot_images(t_images, lines, title=""):
    t_images = t_images.detach().cpu() * 0.5 + 0.5
    grid = make_grid(t_images, nrow=lines).permute(1, 2, 0)
    plt.figure(figsize=(10, 7))
    plt.imshow(grid)
    plt.axis("off")
    plt.title(title)
    plt.show()
