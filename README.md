# map_generation

##

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