import pandas as pd
import streamlit as st
from model_wrapper import Model
from PIL import Image

from map_generation.osm_dataset import create_sentence


class StreamlitApp:
    def __init__(self) -> None:
        self.placeholder = st.empty()
        self.columns = self.placeholder.columns(2)
        with self.columns[0]:
            self.data = {
                "building_yes": st.slider(
                    "residential buildings",
                    0,
                    50,
                ),
                "water_river": st.slider(
                    "river",
                    0,
                    5,
                ),
                "water": st.slider("natural_water", 0, 5),
                "park": st.slider("leisure_park", 0, 5),
            }
            self.button = st.button(label="Generate!", on_click=self.generate_image)

        img = self.get_or_default(Image.open("examples/tile.png"), "img")
        caption = self.get_or_default(
            "OSM of area containing: 1 shop convenience  , 2 shop lotterys  , 1 amenity social facility  , 2 amenity community centres  , 1 amenity parking entrance  .",
            "caption",
        )
        self.columns[1].image(img)
        self.columns[1].markdown(caption)

        self.model = Model()

    def get_or_default(self, default, field: str):
        return default if "img" not in st.session_state else st.session_state[field]

    def generate_image(self):
        new_caption = self.read_caption()
        generated_image = self.model.generate_image(new_caption)[0]
        st.session_state["img"] = generated_image
        st.session_state["caption"] = new_caption

    def read_caption(self) -> str:
        series = pd.Series(self.data)
        return create_sentence(series, n_columns=series.shape[0])


if __name__ == "__main__":
    StreamlitApp()
