import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from ultralytics import YOLO

from utils import ColorCheckerReference
from color_correction import ColorCorrection


@dataclass
class ColorCheckerSwatch:
    """Data class representing a single color swatch"""
    color: Tuple[float, float, float]  # RGB values
    position: Tuple[int, int, int, int]  # x1, y1, x2, y2 coordinates
    image: np.ndarray  # Swatch image crop


class ColorCheckerDetector:
    def __init__(self, model_path: str = "colour-checker-detection-l-seg.pt"):
        """
        Initialize detector with YOLOv8 segmentation model.

        Args:
            model_path: Path to .pt model file
        """
        self.model = YOLO(model_path)
        self.last_results = None
        self.swatches = None

    def detect(self, image_path: str, conf: float = 0.5) -> bool:
        """
        Detect color checker in image.

        Args:
            image_path: Path to input image
            conf: Confidence threshold

        Returns:
            True if detection was successful
        """
        self.last_results = self.model.predict(image_path, conf=conf)
        return len(self.last_results[0]) > 0

    def extract_swatches(self, rows: int = 4, cols: int = 6, border: float = 0.02) -> List[ColorCheckerSwatch]:
        """
        Extract color swatches from detected color checker.

        Args:
            rows: Number of rows in color checker
            cols: Number of columns in color checker

        Returns:
            List of ColorCheckerSwatch objects
        """
        if not self.last_results or not self.last_results[0].masks:
            raise ValueError("No color checker detected or no mask available!")

        result = self.last_results[0]
        mask = result.masks[0].data[0].cpu().numpy()
        bbox = result.boxes[0].xyxy[0].cpu().numpy()

        # Crop the color checker region using the bounding box
        x1, y1, x2, y2 = map(int, bbox)
        color_checker_region = result.orig_img[y1:y2, x1:x2]
        # Remove borders

        border_h = int(color_checker_region.shape[0] * border)
        border_w = int(color_checker_region.shape[1] * border)
        color_checker_region = color_checker_region[border_h:-border_h, border_w:-border_w]

        cv2.imshow("Color Checker Region", color_checker_region)

        # Apply bilateral filter to smooth the image
        color_checker_region = self._bilateral_filter(color_checker_region)
        cv2.imshow("Filtered Color Checker Region", color_checker_region)

        # Divide into swatches
        self.swatches = self._divide_into_swatches(color_checker_region, rows, cols)
        return self.swatches

    def _bilateral_filter(self, image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """Apply bilateral filter to smooth the image while preserving edges"""
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    def _divide_into_swatches(self, image: np.ndarray, rows: int, cols: int, roi_size: int = 6) -> List[ColorCheckerSwatch]:
        """Divide color checker image into individual swatches"""
        swatches = []
        h, w = image.shape[:2]
        cell_h, cell_w = h // rows, w // cols
        for i in range(rows):
            for j in range(cols):
                center_x = int((j + 0.5) * cell_w)
                center_y = int((i + 0.5) * cell_h)

                x1 = max(0, center_x - roi_size // 2)
                x2 = min(w, center_x + roi_size // 2 + 1)
                y1 = max(0, center_y - roi_size // 2)
                y2 = min(h, center_y + roi_size // 2 + 1)
                roi = image[y1:y2, x1:x2]

                median_color = np.median(roi.reshape(-1, 3), axis=0)
                swatches.append(ColorCheckerSwatch(
                    color=median_color,
                    position=(x1, y1, x2, y2),
                    image=roi
                ))
        return swatches

    def visualize(self, show_swatches: bool = True) -> None:
        """Overlay detected swatch colors onto the color checker region with borders."""
        if not self.last_results or not self.swatches:
            print("No results or swatches to visualize!")
            return

        result = self.last_results[0]
        bbox = result.boxes[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, bbox)

        # Work on a copy of the original image
        overlay_img = result.orig_img.copy()

        # Draw filled rectangles + borders for each swatch
        for swatch in self.swatches:
            sx1, sy1, sx2, sy2 = swatch.position

            # Adjust swatch positions relative to the original image
            abs_x1 = x1 + sx1
            abs_y1 = y1 + sy1
            abs_x2 = x1 + sx2
            abs_y2 = y1 + sy2

            # Fill the swatch area with its median color
            color = tuple(map(int, swatch.color))  # Convert to int
            cv2.rectangle(overlay_img, (abs_x1, abs_y1), (abs_x2, abs_y2), color, thickness=-1)

            # Draw border (white or black)
            border_color = (255, 255, 255)  # White border
            cv2.rectangle(overlay_img, (abs_x1, abs_y1), (abs_x2, abs_y2), border_color, thickness=1)

        # Optionally: draw the whole color checker bounding box
        cv2.rectangle(overlay_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

        # Show the final visualization
        cv2.imshow("Color Checker with Swatch Overlay", overlay_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    detector = ColorCheckerDetector("colour-checker-detection-l-seg.pt")
    detector.detect("photos/card_3.jpg")
    if detector.last_results is not None:
        swatches = detector.extract_swatches()
        print(f"Detected {len(swatches)} color swatches")
        # detector.visualize()

    reference = ColorCheckerReference()
    colorCorrector = ColorCorrection(reference, detector.swatches, "photos/card_3.jpg")
    color_correction_matrix = colorCorrector.get_color_correction_matrix_lab()
    corrected_image = colorCorrector.apply_color_correction(detector.last_results[0].orig_img, color_correction_matrix)
    cv2.imshow("Corrected Image", corrected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


