import glob
import os

import pandas as pd
import PIL
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
from torchvision.datasets.utils import extract_archive
from torchvision.io import read_image


def read_file(file_path: str):
    with open(file_path) as file:
        content = file.read()
        content = content.replace("\n", " ")
        return content


class ImageCaptionDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, start=0, end=None):
        caption_dir = os.path.join(root, "dataset", "text", "celeba-caption")
        number_of_records = len(os.listdir(caption_dir))
        l = [
            read_file(os.path.join(caption_dir, f"{file_number}.txt"))
            for file_number in range(number_of_records)
        ]
        self.captions = pd.DataFrame(l, columns=["caption"])
        self.root = root
        self.img_dir = os.path.join("image", "images")
        self.transform = transform
        self.target_transform = target_transform
        self.start = start
        self.end = end if end is not None else number_of_records

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError()
        idx = idx + self.start
        img_path = os.path.join(self.root, "dataset", self.img_dir, f"{idx}.jpg")
        image = read_image(img_path)
        text = self.captions.iloc[idx]["caption"]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            text = self.target_transform(text)
        return image, text


def load_local_image_caption(root, **kwargs):
    def generator(ds):
        for item in ds:
            yield {"img": torch.tensor(item[0]), "caption": item[1]}
    ds = ImageCaptionDataset(root=root, **kwargs)
    hds = Dataset.from_generator(
        lambda: generator(ds),
    )
    hds.set_format("pt")
    return hds


if __name__ == "__main__":
    ds = ImageCaptionDataset(root="data/MMCelebHQ", start=10, end=15)
    ob = ds[0]
    print(ob[0].shape)
    print(len(ds))
    for i, item in enumerate(ds):
        print(i)
    print("Done")
