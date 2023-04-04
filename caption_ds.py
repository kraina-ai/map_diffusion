import os
from tkinter import NO

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import datasets
from tqdm import tqdm


def read_file(file_path: str):
    with open(file_path) as file:
        content = file.read()
        content = content.replace("\n", " ")
        return content


class OSMTilesDataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        captions=False,
        cities: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.use_captions = captions
        self.cities = cities

        self.img_dir = os.path.join(root, "tiles")

        self.indexes = self._tile_id_to_idx()

        if captions:
            caption_dir = os.path.join(root, "text")
            self.captions = self.indexes.apply(
                lambda row: read_file(os.path.join(caption_dir, row["path"] + ".txt")),
                axis=1,
            )
            self.captions = pd.DataFrame(self.captions, columns=["caption"])

    def _tile_id_to_idx(self) -> pd.DataFrame:
        cities = os.listdir(self.img_dir) if self.cities is None else self.cities
        cities_list = []
        for city in cities:
            path = os.path.join(self.img_dir, city)
            city_list = [
                os.path.join(city, image).rsplit(".", 1) for image in os.listdir(path)
            ]
            cities_list.extend(city_list)
        return pd.DataFrame(cities_list, columns=["path", "img_type"])

    def __len__(self) -> int:
        return self.indexes.shape[0]

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError()
        path = self.indexes.iloc[0]["path"]
        img_type = self.indexes.iloc[0]["img_type"]
        img_path = os.path.join(self.img_dir, f"{path}.{img_type}")
        image = read_image(img_path, ImageReadMode.RGB)
        if self.transform:
            image = self.transform(image)
        if self.use_captions:
            text = self.captions.iloc[idx]["caption"]
            if self.target_transform:
                text = self.target_transform(text)
            return image, text
        else:
            return image

    def to_huggingface(self, **kwargs) -> datasets.Dataset:
        def generator(ds: OSMTilesDataset):
            for item in ds:
                if ds.use_captions:
                    yield {"img": item[0], "caption": item[1]}
                else:
                    yield {"img": item}

        hds = datasets.Dataset.from_generator(lambda: generator(self), **kwargs)
        assert isinstance(hds, datasets.Dataset)
        hds.set_format("pt")
        return hds


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


def load_local_image_caption(root, **kwargs) -> datasets.Dataset:
    def generator(ds):
        for item in ds:
            yield {"img": item[0], "caption": item[1]}

    ds = ImageCaptionDataset(root=root, **kwargs)
    print("generating")
    hds = datasets.Dataset.from_generator(lambda: generator(ds))
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
