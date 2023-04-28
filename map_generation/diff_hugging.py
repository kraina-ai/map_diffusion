import os

import torch
from .dataset_lighting import LightingDataModule
from .nn import LightingStableDiffusion
from .osm_dataset import TextToImageDataset
from .train import train_model


def main(config):
    torch.manual_seed(1)
    print("getting data")
    train_ds = TextToImageDataset(config.DATA_DIR)
    dataset = LightingDataModule(train_ds, train_ds, batch_size=config.BATCH_SIZE)
    print("creating model")
    model = LightingStableDiffusion(config.BASE_MODEL_NAME)
    print("training...")
    train_model(
        model=model, datamodule=dataset, epochs=config.EPOCHS, name=config.MODEL_NAME
    )
    print("saving...")
    model = LightingStableDiffusion.load_from_checkpoint(
        os.path.join(config.CHECKPOINTS_DIR, f"{config.MODEL_NAME}.ckpt"),
        model_name=config.BASE_MODEL_NAME,
    )
    model.save_pipeline(
        os.path.join(config.CHECKPOINTS_DIR, f"{config.MODEL_NAME}_pipeline")
    )


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
