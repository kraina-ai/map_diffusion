from denoising_diffusion_pytorch import GaussianDiffusion, Unet
from nn import Diffusion
from torchvision import transforms
from dataset_lighting import get_celeba_module, LightingDataModule
from train import train_model

IMG_SIZE = 32


def main():
    print("Loading data...")
    transform = transforms.Compose(
        [
            transforms.Resize(size=IMG_SIZE),
            transforms.CenterCrop(size=IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset: LightingDataModule = get_celeba_module(
        transform=transform,
        subset_train=50000,
        subset_test=1000,
        num_workers=16,
        batch_size=32,
    )
    print("Training...")
    diffusion = build_diffusion(dim_mults=(1, 2, 2, 2, 2))
    model = Diffusion(diffusion, learning_rate=1e-6, count_fid=True)
    train_model(model=model, datamodule=dataset, epochs=50, name="32_size_big_final")


def build_diffusion(dim_mults=None):
    if dim_mults is None:
        dim_mults = (1, 2, 4)
    model = Unet(dim=64, dim_mults=(1, 2, 4))

    diffusion = GaussianDiffusion(
        model,
        image_size=IMG_SIZE,
        timesteps=1000,  # number of steps
        loss_type="l1",  # L1 or L2
    )

    return diffusion.cuda()  # (4, 3, 128, 128)


if __name__ == "__main__":
    main()
