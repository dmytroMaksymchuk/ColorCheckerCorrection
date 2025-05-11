from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class ColorSwatch:
    name: str
    hex_code: str

    def to_rgb(self) -> tuple[int, int, int]:
        """Convert hex code to RGB tuple."""
        hex_code = self.hex_code.lstrip("#")
        r = int(hex_code[0:2], 16)
        g = int(hex_code[2:4], 16)
        b = int(hex_code[4:6], 16)
        return r, g, b

    def to_lab(self)

class ColorCheckerReference:
    """Manages the standard ColorChecker reference swatches."""

    def __init__(self):
        self.swatches = [
            ColorSwatch("Dark Skin", "#735244"),
            ColorSwatch("Light Skin", "#C29682"),
            ColorSwatch("Blue Sky", "#627A9D"),
            ColorSwatch("Foliage", "#576C43"),
            ColorSwatch("Blue Flower", "#8580B1"),
            ColorSwatch("Bluish Green", "#67BDAA"),
            ColorSwatch("Orange", "#D67E2C"),
            ColorSwatch("Purplish Blue", "#505BA6"),
            ColorSwatch("Moderate Red", "#C15A63"),
            ColorSwatch("Purple", "#5E3C6C"),
            ColorSwatch("Yellow Green", "#9DBC40"),
            ColorSwatch("Orange Yellow", "#E0A32E"),
            ColorSwatch("Blue", "#383D96"),
            ColorSwatch("Green", "#469449"),
            ColorSwatch("Red", "#AF363C"),
            ColorSwatch("Yellow", "#E7C71F"),
            ColorSwatch("Magenta", "#BB5695"),
            ColorSwatch("Cyan", "#0885A1"),
            ColorSwatch("White", "#F3F3F2"),
            ColorSwatch("Neutral 8", "#C8C8C8"),
            ColorSwatch("Neutral 6.5", "#A0A0A0"),
            ColorSwatch("Neutral 5", "#7A7A79"),
            ColorSwatch("Neutral 3.5", "#555555"),
            ColorSwatch("Black", "#343434"),
        ]

    def get_all(self) -> List[ColorSwatch]:
        """Return all swatches."""
        return self.swatches

    def get_by_name(self, name: str) -> Optional[ColorSwatch]:
        """Find a swatch by name."""
        for swatch in self.swatches:
            if swatch.name.lower() == name.lower():
                return swatch
        return None




