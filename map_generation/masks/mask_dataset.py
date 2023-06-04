from pathlib import Path
import torch.utils.data as t_data
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import glob
from functional import seq
import pandas as pd


class SegmentationDataset(t_data.Dataset):
    def __init__(
        self,
        root: Path,
    ) -> None:
        super().__init__()

        self.root = root
        self.indexes = pd.Series(
            seq(glob.glob(str(root / "*.png")))
            .map(lambda path: path.split("/")[-1])
            .to_list()
        )

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError()
        img_path = self.root / self.indexes[idx]
        mask_path = self.root / "mask" / self.indexes[idx]
        image = read_image(str(img_path), ImageReadMode.RGB)
        mask = read_image(str(mask_path), ImageReadMode.GRAY)
        return image, mask
