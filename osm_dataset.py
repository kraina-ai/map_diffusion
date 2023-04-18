from torch.utils.data import Dataset
from datasets import load_dataset
import pandas as pd
import torch
import torchvision.transforms as transforms
from transformers import CLIPTokenizer
from config import BASE_MODEL_NAME

tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_NAME, subfolder="tokenizer")


def get_columns(row, n_columns=5) -> pd.Series:
    not_zeros = row != 0
    columns = pd.Series(not_zeros.index[not_zeros])
    if columns.shape[0] <= n_columns:
        return columns
    else:
        return columns.sample(n_columns)


def create_sentence(row: pd.Series, n_columns: int = 5) -> str:
    columns = get_columns(row, n_columns)
    space = " "
    underscore = "_"
    ls = [
        f"{value} {str(field).replace(underscore, space)}"
        + ("s " if value > 1 else space)
        for (field, value) in row[columns].items()
    ]
    return "Map of area containing: " + "".join(ls) + "."


class TextToImageDataset(Dataset):
    def __init__(
        self,
        path,
        n_columns=5,
        resolution=256,
        center_crop=False,
        random_flip=False,
    ) -> None:
        super().__init__()
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
        self.dataset = load_dataset(path)
        self.dataset = self.dataset.map(self.prepare_data, batched=True).with_format(
            "pt"
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
            lambda row: create_sentence(row, n_columns=self.n_columns), axis=1
        )
        caption_tensor: torch.Tensor = tokenizer(
            captions.tolist(),
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze()

        return list(caption_tensor)

    def to_huggingface_dataset(self):
        return self.dataset.select_columns(["pixel_values", "input_ids"])
