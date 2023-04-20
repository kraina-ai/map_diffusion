"""
OSM tile loader.

This module implements downloading tiles from given OSM tile server.
"""

from pathlib import Path
from typing import Any, Callable, Optional, Union
from urllib.parse import urljoin

import pandas as pd
import requests

import time
from datetime import datetime

from srai.regionizers.slippy_map_regionizer import SlippyMapRegionizer
from srai.utils import geocode_to_region_gdf

from vector_tile_data_collector import (VectorDataCollector, 
        SavingVectorDataCollector, InMemoryVectorDataCollector)


class VectorTileLoader:
    """
    Tile Loader.

    Downloads raster tiles from user specified tile server, like listed in [1]. Loader founds x, y
    coordinates [2] for specified area and downloads tiles. Address is built with schema
    {tile_server_url}/{zoom}/{x}/{y}.{resource_type}

    References:
        1. https://wiki.openstreetmap.org/wiki/Raster_tile_providers
        2. https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
    """

    def __init__(
        self,
        tile_server_url: str,
        zoom: int,
        verbose: bool = False,
        resource_type: str = "mvt",
        auth_token: Optional[str] = None,
        collector_factory: Optional[Union[str, Callable[[], VectorDataCollector]]] = None,
        storage_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.zoom = zoom
        self.verbose = verbose
        self.resource_type = resource_type
        self.base_url = urljoin(tile_server_url, "{0}/{1}/{2}." + resource_type)
        self.auth_token = auth_token
        self.save_path = storage_path
        self.collector_factory = (
            self._get_collector_factory(collector_factory)
            if collector_factory is not None
            else lambda: InMemoryVectorDataCollector()
        )
        self.regionizer = SlippyMapRegionizer(z=self.zoom)
        if self.verbose:
            print(self.base_url)

    def _get_collector_factory(
        self, storage_strategy: Union[str, Callable[[], VectorDataCollector]]
    ) -> Callable[[], VectorDataCollector]:
        if isinstance(storage_strategy, str):
            if storage_strategy == "save" and self.save_path is None:
                raise ValueError
            elif self.save_path is not None:
                save_path: Union[Path, str] = self.save_path
            return {
                "save": lambda: SavingVectorDataCollector(save_path, f_extension=self.resource_type),
                "return": lambda: InMemoryVectorDataCollector(),
            }[storage_strategy]
        else:
            return storage_strategy

    def get_tile_by_x_y(self, x: int, y: int) -> Any:
        """
        Downloads single tile from tile server.

        Args:
            x: x tile x coordinate
            y: y tile y coordinate
        """
        url = self.base_url.format(self.zoom, x, y)
        if self.verbose:
            print(f"Getting tile from url: {url}")
        content = None
        try:
            content = requests.get(url, params={"access_token": f"{self.auth_token}"}, timeout=5).content
        except Exception as e:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(f"{current_time}: failed at {x}, {y}")
                print(e)
                time.sleep(60)
        return content

    def get_tile_by_region_name(self, name: str) -> pd.DataFrame:
        """
        Returns all tiles of region.

        Args:
            name: area name, as in geocode_to_region_gdf
            return_rect: if true returns tiles out of area to keep rectangle shape of img of joined
                tiles.
        """
        tiles_collector = self.collector_factory()
        gdf = geocode_to_region_gdf(name)
        regions = self.regionizer.transform(gdf=gdf)
        data_series = regions.apply(
            lambda row: self._get_tile_for_area(row, tiles_collector), axis=1
        )
        return pd.DataFrame(data_series, columns=["tile"])

    def _get_tile_for_area(self, row: pd.Series, tiles_collector: VectorDataCollector) -> Any:
        x, y = row.name
        if tiles_collector.exists(x,y):
            return tiles_collector.restore(x, y)
        tile = self.get_tile_by_x_y(x, y)
        return tiles_collector.store(x, y, tile)
