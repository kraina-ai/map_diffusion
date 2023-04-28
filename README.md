# map_generation

## Project init

1. Install [PDM](https://pdm.fming.dev/latest) (only if not already installed)

    ```sh
    pip install pdm
    ```

2. Install package locally (will download all dev packages and create a local venv)

    ```sh
    # Optional if you want to create venv in a specific version. More info: https://pdm.fming.dev/latest/usage/venv/#create-a-virtualenv-yourself
    pdm venv create 3.10 # tested version

    pdm install
    ```

3. There is some problem with kaleido, you might have to install it with pip if you need it.


## Notebooks

To see workflow process use notebooks from workflow_raster. It also contains less memory consuming training procedure.

## Run code in container

- specify  configs in config.py (dirs inside container, base model, batch size, epoch, n_columns used to generate prompt, dataset)
- dataset can be located in data/tiles (specify in config data/tiles) or huggingface dataset (like mprzymus/osm_tiles_small which contains tiles from Wrocław)
- tiles can be downloaded from https://drive.google.com/drive/u/0/folders/1rIsECcHQNBj60905VhGzPUkeQuaBKgmI
- you can specify more arguments in run.sh according to https://huggingface.co/docs/diffusers/training/text2image
- `docker build --tag diffusion . `
- `docker run -it --gpus all -v  ./logs_cont:/app/result_path diffusion  `
- to monitor logs run `tensorboard --logdir logs_cont/logs `
- after training model is located inside log_cont directory