import argparse

import torch
from diffusers import StableDiffusionPipeline
from stqdm import stqdm


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="models/model", required=False
    )
    args = parser.parse_args()
    return args


class Model:
    print("creating model")
    args = setup_args()
    model = StableDiffusionPipeline.from_pretrained(args.model_path, safety_checker=None).to("cuda")
    print("model created")

    def generate_image(self, caption: str):
        bar = stqdm(total=50)

        def callback(*_):
            bar.update()

        imgs = self.model(prompt=caption, callback=callback).images
        return imgs
