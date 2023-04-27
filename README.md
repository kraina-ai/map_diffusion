# map_generation

## Run code in container

- specify  configs in config.py
- `docker build --tag diffusion . `
- `docker run -it --gpus all -v  ./logs_cont:/app/result_path diffusion  `