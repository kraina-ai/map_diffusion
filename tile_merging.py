from itertools import product
from typing import Any

import geopandas as gpd
import pandas as pd
from functional import seq
from PIL import Image
from shapely.ops import unary_union
from tqdm import tqdm


class SlippyMapRegionMerger:
    def __init__(self, zoom) -> None:
        self.zoom = zoom

    def to_id(self, x, y):
        return f"{x}_{y}_{self.zoom}"

    def tile_exists(self, x, y, regions):
        return self.to_id(x, y) in regions.index

    def merge_regions(self, regions: gpd.GeoDataFrame, num_merged: int = 2):
        all_list = []
        x_min = regions["x"].min()
        y_min = regions["y"].min()
        for x in tqdm(regions["x"].unique()):
            if x % num_merged == x_min % num_merged:
                gdf_this_x = regions[regions["x"] == x]
                for y in gdf_this_x["y"]:
                    if y % num_merged == y_min % num_merged:
                        if self.all_tiles_exists(x, y, num_merged, regions):
                            all_list.append(
                                {
                                    "region_id": self.to_id(x, y),
                                    "parts_id": [
                                        self.to_id(element_x, element_y)
                                        for element_x, element_y in self.all_product(
                                            x, y, num_merged
                                        )
                                    ],
                                    "geometry": unary_union(
                                        [
                                            regions.loc[
                                                self.to_id(element_x, element_y)
                                            ].geometry
                                            for element_x, element_y in self.all_product(
                                                x, y, num_merged
                                            )
                                        ]
                                    ),
                                    "xs": [x for x in range(x, x + num_merged)],
                                    "ys": [y for y in range(y, y + num_merged)],
                                }
                            )
        return gpd.GeoDataFrame(all_list).set_index("region_id")

    def all_product(self, x, y, num_merged):
        return product(
            range(x, x + num_merged),
            range(y, y + num_merged),
        )

    def all_tiles_exists(self, x_tile_min, y_tile_min, num_merged, regions):
        for x, y in product(
            range(x_tile_min, x_tile_min + num_merged),
            range(y_tile_min, y_tile_min + num_merged),
        ):
            if not self.tile_exists(x, y, regions):
                return False
        return True


class OSMTileMerger:
    def __init__(self, path, data_collector) -> None:
        self.path = path
        self.dc = data_collector

    def merge_tiles(self, pd_row: pd.DataFrame) -> Any:
        ids = pd_row["parts_id"]
        imgs = {x: {} for x in pd_row["xs"]}
        for img_id in ids:
            x, y, z = seq(img_id.split("_")).map(lambda i: int(i))
            imgs[x][y] = Image.open(self.path / f"{img_id}.png")
        width, height = imgs[x][y].width, imgs[x][y].height
        img = Image.new(
            "RGB", size=(width * len(pd_row["xs"]), height * len(pd_row["ys"]))
        )
        x_min = min(pd_row["xs"])
        y_min = min(pd_row["ys"])
        for x, y in product(sorted(pd_row["xs"]), sorted(pd_row["ys"])):
            x_pos = x - x_min
            y_pos = y - y_min
            img.paste(imgs[x][y], (width * x_pos, height * y_pos))
        return self.dc.store(pd_row.name, img)
