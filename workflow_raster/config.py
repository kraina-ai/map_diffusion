from srai.loaders.osm_loaders.filters import HEX2VEC_FILTER
from srai.loaders.osm_loaders.filters.popular import get_popular_tags

DEVICE = "cuda"
DATA_DIR = "mprzymus/osm_tiles_small"
CHECKPOINTS_DIR = "model"
LOGS_DIR = "logs"
BASE_MODEL_NAME = "OFA-Sys/small-stable-diffusion-v0"
BATCH_SIZE = 1
EPOCHS = 2
MODEL_NAME = "testing_workflow"
LEARNING_RAGE = 1e-5
N_COLUMNS = 5

road_tags = get_popular_tags(in_wiki_only=True)
tags_to_use = {
    "railway": road_tags["railway"],
    "highway": road_tags["highway"],
}

TAGS = HEX2VEC_FILTER | tags_to_use
