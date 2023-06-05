from pathlib import Path
import torch.utils.data as t_data
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import glob
from functional import seq
import pandas as pd
import os
from sklearn.utils import shuffle

class SegmentationDataset(t_data.Dataset):
    def __init__(self, root: Path, root_contrain_cities: bool = False) -> None:
        super().__init__()

        self.root = root
        if not root_contrain_cities:
            dirs = [""]
        else:
            dirs = os.listdir(root)

        self.cities = []
        for city in dirs:
            files = self._get_images_for_location(root / city)
            masks = self._get_images_for_location(root / city / "mask")
            files = files & masks
            data = seq(files).map(lambda file: (file, city)).to_list()
            self.cities.append(data)
        self.indexes = pd.DataFrame(seq(self.cities).flatten())
        self.indexes = self.indexes.rename(columns={0: "file", 1: "city"})
        self.indexes = shuffle(self.indexes)

    def _get_images_for_location(self, dir: str | Path) -> list[str]:
        return (
            seq(glob.glob(str(dir / "*.png")))
            .map(lambda path: path.split("/")[-1])
            .to_set()
        )

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError()
        index = self.indexes.iloc[idx]
        img_path = self.root / index["city"] / index["file"]
        mask_path = self.root / index["city"] / "mask" / index["file"]
        image = read_image(str(img_path), ImageReadMode.RGB) / 255
        mask = read_image(str(mask_path), ImageReadMode.GRAY) / 255
        return image, mask
