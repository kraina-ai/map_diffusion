# build stage
FROM python:3.10 AS builder

# # # install PDM
RUN pip3 install -U pip setuptools wheel
RUN pip3 install pdm

# # copy files

COPY run.sh config_accelerate.yaml pyproject.toml pdm.lock README.md /app/
COPY map_generation/ /app/map_generation

# # install dependencies and project into the local packages directory
WORKDIR /app
RUN mkdir __pypackages__ && pdm install --prod --no-lock --no-editable

FROM pytorch/pytorch

# retrieve packages from build stage
ENV PYTHONPATH=/app/pkgs

COPY --from=builder /app/__pypackages__/3.10/lib /app/pkgs
RUN pip uninstall -y accelerate
RUN pip install accelerate

#create project files
COPY run.sh config_accelerate.yaml pyproject.toml pdm.lock README.md /app/
COPY map_generation/ /app/map_generation
WORKDIR /app
ARG src="data/tiles/Athens, Greece"
COPY ${src} ./data
VOLUME ./result_path
ENTRYPOINT ["/bin/bash", "run.sh"]

