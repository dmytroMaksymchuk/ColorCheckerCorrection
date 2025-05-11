from dataclasses import dataclass
from typing import Tuple, List, Optional

import cv2
import numpy as np
from skimage import io, color
def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    """
        Convert hex code to RGB tuple.
        sRGB color space is assumed.
    """
    hex_code = hex_code.lstrip("#")
    r = int(hex_code[0:2], 16)
    g = int(hex_code[2:4], 16)
    b = int(hex_code[4:6], 16)
    return r, g, b

def rgb_to_lab(rgb: tuple) -> tuple:
    """
        Convert RGB tuple to CIE Lab color space.
        sRGB color space is assumed.
    """
    # Normalize RGB values to [0, 1]
    r, g, b = [x / 255.0 for x in rgb]
    rgb_normalized = [[[(r), (g), (b)]]]  # shape (1, 1, 3)

    # Convert to Lab color space
    lab = color.rgb2lab(rgb_normalized)
    return tuple(lab[0][0])

def rgb_image_to_lab(image: np.ndarray) -> np.ndarray:
    """
        Convert an RGB image to CIE Lab color space.
    """
    # Normalize RGB values to [0, 1]
    rgb_normalized = image / 255.0

    # Convert to Lab color space
    lab_image = color.rgb2lab(rgb_normalized)
    return lab_image

def lab_image_to_rgb(lab_image: np.ndarray) -> np.ndarray:
    """
        Convert a CIE Lab image to RGB color space.
    """
    # Convert to RGB color space
    rgb_image = color.lab2rgb(lab_image)

    # Scale back to [0, 255]
    rgb_image = (rgb_image * 255).astype(np.uint8)
    return rgb_image

def bgr_to_rgb(bgr: tuple) -> tuple:
    """
        Convert BGR tuple to RGB tuple.
    """
    b, g, r = bgr
    return r, g, b





def compare_ref_photo_swatches(ref_swatches, photo_swatches, corrected_swatches):
    """
    Visualize reference and photo swatches in a horizontal comparison with borders and labels.
    :param ref_swatches: List of reference color swatches
    :param photo_swatches: List of photo color swatches
    :return: Combined image showing swatches horizontally with borders and labels
    """
    # Constants
    swatch_height = 60
    swatch_width = 60
    border_thickness = 1
    padding = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    font_thickness = 1

    # Calculate image dimensions
    max_swatches = max(len(ref_swatches), len(photo_swatches))
    image_width = (swatch_width + padding) * max_swatches + padding * 2 + 100
    image_height = swatch_height * 2 + padding * 4 + 200 # Extra space for labels

    # Create blank image with white background
    combined_image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255

    # Draw photo swatches with borders (bottom row)
    for i, color in enumerate(photo_swatches):
        x_start = padding + i * (swatch_width + padding)
        x_end = x_start + swatch_width

        # Position in bottom half of image
        y_start = 40
        y_end = y_start + swatch_height

        # Draw border
        cv2.rectangle(combined_image, (x_start, y_start), (x_end, y_end),
                      (0, 0, 0), border_thickness)

        # Draw color block
        cv2.rectangle(combined_image,
                      (x_start + border_thickness, y_start + border_thickness),
                      (x_end - border_thickness, y_end - border_thickness),
                      color, -1)

    # Draw reference swatches with borders (top row)
    for i, swatch in enumerate(ref_swatches):
        x_start = padding + i * (swatch_width + padding)
        x_end = x_start + swatch_width

        # Position in top half of image
        y_start = swatch_height + padding
        y_end = y_start + swatch_height

        # Draw border
        cv2.rectangle(combined_image, (x_start, y_start), (x_end, y_end),
                      (0, 0, 0), border_thickness)

        # Draw color block
        color = tuple(map(int, swatch.rgb))
        cv2.rectangle(combined_image,
                      (x_start + border_thickness, y_start + border_thickness),
                      (x_end - border_thickness, y_end - border_thickness),
                      color, -1)



    return combined_image

class ColorSwatch:
    name: str
    hex_code: str
    rgb: tuple[int, int, int]
    lab: tuple

    def __init__(self, name: str, hex_code: str):
        self.name = name
        self.hex_code = hex_code
        self.rgb = hex_to_rgb(hex_code)
        self.lab = rgb_to_lab(self.rgb)


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




