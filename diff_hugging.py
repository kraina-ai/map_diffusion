import os

import torch
from config import (
    BASE_MODEL_NAME,
    CHECKPOINTS_DIR,
    DATA_DIR,
    EPOCHS,
    MODEL_NAME,
    VALID_DATA_DIR,
)
from dataset_lighting import LightingDataModule
from nn import LightingStableDiffusion
from osm_dataset import TextToImageDataset
from train import train_model


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
        os.path.join(CHECKPOINTS_DIR, f"{MODEL_NAME}.ckpt"), model_name=BASE_MODEL_NAME
    )
    model.save_pipeline(os.path.join(CHECKPOINTS_DIR, f"{MODEL_NAME}_pipeline"))


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


if __name__ == "__main__":
    main()
