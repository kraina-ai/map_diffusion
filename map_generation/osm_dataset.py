import datetime
import os
from pathlib import Path
from torch.utils.data import Dataset
from datasets import load_dataset
import pandas as pd
import torch
import torchvision.transforms as transforms
from transformers import CLIPTokenizer
from .config import BASE_MODEL_NAME


def get_columns(row, n_columns=5) -> pd.Series:
    not_zeros = row != 0
    columns = pd.Series(not_zeros.index[not_zeros])
    if columns.shape[0] <= n_columns:
        return columns
    else:
        return columns.sample(n_columns)


def create_sentence(
    row: pd.Series, n_columns: int = 5, placeholder_token: str="OSM"
) -> str:
    columns = get_columns(row, n_columns)
    space = " "
    underscore = "_"
    ls = [
        f"{value} {str(field).replace(underscore, space)}"
        + ("s " if value > 1 else space)
        for (field, value) in row[columns].items()
    ]
    name = placeholder_token
    return f"{name} of area containing: " + "".join(ls) + "."


class TextToImageDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        n_columns: int = 5,
        resolution: int = 256,
        center_crop: bool = False,
        random_flip: bool = False,
        save_texts: bool = False,
        tokenizer_path: str = BASE_MODEL_NAME,
        placeholder_token: str = "OSM",
    ) -> None:
        super().__init__()
        self.placeholder_token = placeholder_token
        self.texts = []
        self.save_texts = save_texts
        self.n_columns = n_columns
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    resolution, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(resolution)
                if center_crop
                else transforms.RandomCrop(resolution),
                transforms.RandomHorizontalFlip()
                if random_flip
                else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            tokenizer_path, subfolder="tokenizer"
        )
        self.dataset = load_dataset(path)
        self.dataset = self.dataset.map(
            self.prepare_data, batched=True, load_from_cache_file=(not self.save_texts)
        ).with_format("pt")
        if self.save_texts:
            pd.Series(self.texts).to_csv(
                os.path.join(path, f"texts_{str(datetime.datetime.now())}.csv")
            )

    def prepare_data(self, examples):
        examples["input_ids"] = self.prepare_text(examples)
        images = [image.convert("RGB") for image in examples["image"]]
        examples["pixel_values"] = [self.transform(image) for image in images]
        return examples

    def __len__(self):
        return self.dataset["train"].num_rows

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.dataset["train"][index]
        caption_tensor = record["input_ids"]
        img = record["pixel_values"]
        return (img, caption_tensor)

    def prepare_text(self, records):
        df = pd.DataFrame(dict(records)).drop(columns=["image"])
        captions = df.apply(
            lambda row: create_sentence(
                row, n_columns=self.n_columns, placeholder_token=self.placeholder_token
            ),
            axis=1,
        )
        captions_as_list = captions.tolist()
        if self.save_texts:
            self.texts.extend(captions_as_list)
        caption_tensor: torch.Tensor = self.tokenizer(
            captions_as_list,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze()

        return list(caption_tensor)

    def to_huggingface_dataset(self):
        return self.dataset.select_columns(["pixel_values", "input_ids"])
