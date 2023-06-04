import mapbox_vector_tile
import shapely

from PIL import Image
import geopandas as gpd
import geopandas as gpd
from PIL import Image, ImageDraw
from pathlib import Path


class BuildingMaskCreator:
    def __init__(self, zoom: int, path: str | Path, color=255) -> None:
        self.zoom = zoom
        self.path = Path(path)
        self.color = color
        self.width = 256
        self.height = 256
        self.width_scaling_factor = self.width / 4096
        self.height_scaling_factor = self.height / 4096

    def read_gdf(self, tile_to_read: str) -> gpd.GeoDataFrame:
        with open(self.path / f"{tile_to_read}", "rb") as tile:
            data = tile.read()
        d = mapbox_vector_tile.decode(data)
        if "building" in d.keys():
            features = d["building"]["features"]
            gdf = gpd.GeoDataFrame.from_features(features)
        else:
            gdf = gpd.GeoDataFrame()
        return gdf

    def draw_mask(self, tile_id: str) -> Image.Image:
        def draw_polygon(polygon, image):
            if isinstance(polygon, shapely.MultiPolygon):
                for p in list(polygon.geoms):
                    image = draw_polygon(p, image)
                return image
            elif isinstance(polygon, shapely.Polygon):
                draw = ImageDraw.Draw(image)
                coords = [(int(x), int(y)) for x, y in polygon.exterior.coords]
                draw.polygon(coords, fill=self.color, outline=self.color)
            return image

        buildings_gdf = self.read_gdf(tile_to_read=tile_id)
        image = Image.new("L", (self.width, self.height), color=(0))
        if buildings_gdf.empty:
            return image
        else:
            mapped = buildings_gdf.geometry.map(
                lambda polygon: shapely.ops.transform(
                    lambda x, y: (
                        x * self.width_scaling_factor,
                        y * -self.height_scaling_factor + self.height,
                    ),
                    polygon,
                )
            )
            for polygon in mapped.tolist():
                image = draw_polygon(polygon, image)
            return image
