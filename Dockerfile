# build stage
FROM pytorch/pytorch

RUN python3 -m pip install --upgrade pip

RUN apt update && apt install -y git
# install PDM
RUN pip install -U pip setuptools wheel
RUN pip install pdm

# copy files

WORKDIR /app
COPY run.sh config_accelerate.yaml pyproject.toml pdm.lock README.md .
COPY map_generation/ /app/map_generation

# install dependencies and project into the local packages directory
RUN pdm install --prod --no-lock --no-editable

# FROM huggingface/accelerate-gpu

# retrieve packages from build stage
# ENV PYTHONPATH=/app/pkgs
# COPY --from=builder /app/__pypackages__/3.10/lib /app/pkgs
# RUN pip uninstall -y accelerate
RUN pip install accelerate

#create project files
ARG src="data/tiles/Athens, Greece"
COPY ${src} ./data
VOLUME result_path
# RUN accelerate config
# RUN chmod +x run.sh
ENTRYPOINT ["/bin/bash", "run.sh"]
# CMD ["/bin/bash"]

# ENTRYPOINT ls map_generation

