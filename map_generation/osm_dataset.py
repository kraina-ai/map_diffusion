import datetime
import os

import pandas as pd
import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import CLIPTokenizer

from map_generation.config import BASE_MODEL_NAME


def get_columns(row, n_columns=5) -> pd.Series:
    not_zeros = row != 0
    columns = pd.Series(not_zeros.index[not_zeros])
    if columns.shape[0] <= n_columns:
        return columns
    else:
        return columns.sample(n_columns)


def create_sentence(row: pd.Series, n_columns: int = 5) -> str:
    geocode = row["geocode"]
    name = row["name"]
    row = row.dropna()
    row_only_digits = row[row.astype(str).str.isdigit()]
    columns = get_columns(row_only_digits, n_columns)
    ls = [
        _create_str_from_field(field, value) for (field, value) in row[columns].items()
    ]
    area_name = " " + name if name is not None else ""
    text_comma_end = f"OSM from {geocode} of{area_name} area containing: " + "".join(ls)
    return text_comma_end[:-2] + "."


def _create_str_from_field(field, value):
    space = " "
    underscore = "_"
    str_field = str(field).replace("_yes", "").replace(underscore, space)
    return (
        f"{value} {str_field}"
        + ("s " if value > 1 and len(str_field) != 0 else "")
        + ", "
    )


class TokenizedDataset(Dataset):
    def __init__(
        self,
        path,
        tokenizer_path: str = BASE_MODEL_NAME,
        resolution=256,
        center_crop=False,
        random_flip=False,
        cache_dir=None,
    ) -> None:
        super().__init__()
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
        self.dataset = load_dataset(path, cache_dir=cache_dir).with_transform(
            self.prepare_data
        )

    def prepare_data(self, examples):
        examples["input_ids"] = self.tokenize(examples)
        images = [image.convert("RGB") for image in examples["image"]]
        examples["pixel_values"] = [self.transform(image) for image in images]
        return examples

    def __len__(self):
        return self.dataset["train"].num_rows

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.dataset["train"][index]
        caption_tensor = record["input_ids"]
        img = record["image"]
        return (img, caption_tensor)

    def tokenize(self, records):
        captions = records["caption"]
        caption_tensor: torch.Tensor = self.tokenizer(
            captions,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

        return caption_tensor

    def to_huggingface_dataset(self):
        return self.dataset


class TextToImageDataset(Dataset):
    def __init__(
        self,
        path,
        n_columns=5,
        save_texts=False,
        cache_dir=None,
    ) -> None:
        super().__init__()
        print(cache_dir)
        self.texts = []
        self.save_texts = save_texts
        self.n_columns = n_columns
        self.dataset = load_dataset(path, cache_dir=cache_dir).map(
            self.prepare_data, batched=True
        )
        if self.save_texts:
            pd.Series(self.texts).to_csv(
                os.path.join(path, f"texts_{str(datetime.datetime.now())}.csv")
            )

    def prepare_data(self, examples):
        examples["caption"] = self.prepare_text(examples)
        return examples

    def __len__(self):
        return self.dataset["train"].num_rows

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.dataset["train"][index]
        caption = record["caption"]
        img = record["image"]
        return (img, caption)

    def prepare_text(self, records):
        df = pd.DataFrame(dict(records)).drop(columns=["image"])
        captions = df.apply(
            lambda row: create_sentence(row, n_columns=self.n_columns), axis=1
        )
        captions_as_list = captions.tolist()
        if self.save_texts:
            self.texts.extend(captions_as_list)
        return captions_as_list

    def to_huggingface_dataset(self):
        return self.dataset.select_columns(["image", "caption"])
