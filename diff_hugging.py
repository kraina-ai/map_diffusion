import os

import torch

from caption_ds import load_local_image_caption
from config import (
    BASE_MODEL_NAME,
    CHECKPOINTS_DIR,
    DATA_DIR,
    VALID_DATA_DIR,
    EPOCHS,
    MODEL_NAME,
)
from dataset_lighting import LightingDataModule
from datasets import DatasetDict, load_dataset
from imagen.dataset import TextToImageDataset
from nn import LightingStableDiffusion
from torchvision import transforms
from train import train_model
from transformers import CLIPTokenizer


def main():
    torch.manual_seed(1)  
    print("getting data")
    train_ds = TextToImageDataset(DATA_DIR)
    test_ds = TextToImageDataset(VALID_DATA_DIR)
    dataset = LightingDataModule(train_ds, test_ds, batch_size=1)
    print("creating model")
    model = LightingStableDiffusion(BASE_MODEL_NAME)
    print("training...")
    train_model(model=model, datamodule=dataset, epochs=EPOCHS, name=MODEL_NAME)
    print("saving...")
    model = LightingStableDiffusion.load_from_checkpoint(
        os.path.join(CHECKPOINTS_DIR, f'{MODEL_NAME}.ckpt'), model_name=BASE_MODEL_NAME
    )
    model.save_pipeline(os.path.join(CHECKPOINTS_DIR, f'{MODEL_NAME}_pipeline'))


def apply_transform_on_images(data, transform, image_field="image"):
    data[image_field] = [transform(image.convert("RGB")) for image in data[image_field]]
    return data


def apply_transform_on_text(data, tokenizer, text_field="text"):
    tokens = [
        tokenizer(
            data[text_field],
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze()
    ]
    data[text_field] = tokens
    return data


# def get_dataset_custom_split(transform=None, dataset_name=DATASET_NAME):
#     if transform is None:
#         transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
#     tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_NAME, subfolder="tokenizer")

#     def trans_fun(data):
#         data = apply_transform_on_text(data=data, tokenizer=tokenizer)
#         return apply_transform_on_images(data=data, transform=transform)

#     ds_train = load_dataset(dataset_name, split="train[:80%]")
#     ds_train.set_transform(trans_fun)
#     ds_test = load_dataset(dataset_name, split="train[80%:]")
#     ds_test.set_transform(trans_fun)
#     return DatasetDict({"train": ds_train, "test": ds_test})


if __name__ == "__main__":
    main()
