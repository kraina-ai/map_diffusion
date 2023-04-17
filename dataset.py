from torch.utils.data import Dataset
from datasets import load_dataset, DatasetDict
import pandas as pd
import torch
from torchvision.transforms import ToTensor
from transformers import CLIPTokenizer


tokenizer = CLIPTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-2", subfolder="tokenizer"
)


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
    def __init__(self, path, n_columns=5, transform=None) -> None:
        super().__init__()
        self.n_columns = n_columns
        if transform is None:
            transform = ToTensor()
        self.transform = transform
        self.dataset = load_dataset(path)
        self.dataset = self.dataset.map(self.prepare_data, batched=True)

    def prepare_data(self, examples):
        examples["image"] = [image.convert("RGB") for image in examples["image"]]
        examples["text"] = [image.convert("RGB") for image in examples["image"]]
        return examples

    def __len__(self):
        return self.dataset["train"].num_rows

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.dataset["train"][index]
        caption_tensor = record["text"]
        return (self.transform(record["image"]), caption_tensor)

    def prepare_text(self, record):
        caption = create_sentence(
            pd.Series(record).drop(index=["image"]), n_columns=self.n_columns
        )
        caption_tensor: torch.Tensor = tokenizer(
            caption,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze()
        
        return caption_tensor
