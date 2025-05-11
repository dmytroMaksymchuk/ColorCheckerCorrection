import numpy as np

from utils import ColorCheckerReference, rgb_image_to_lab, lab_image_to_rgb, bgr_to_rgb
from utils import hex_to_rgb, rgb_to_lab


class ColorCorrection:
    def __init__(self, reference: ColorCheckerReference, photo_colors: list, bgr: bool = False):
        """
        Initialize ColorCorrection with reference colors and photo colors.

        Args:
            reference: ColorCheckerReference object
            photo_colors: List of detected colors in the photo color checker
            image_path: Path to the input image
        """
        self.reference = reference
        self.photo_colors = photo_colors

        if bgr:
            self.photo_colors = [bgr_to_rgb(swatch.color) for swatch in photo_colors]

    def get_color_correction_matrix_lab(self) -> np.ndarray:
        """
        Calculate the affine color correction matrix (Lab space) using the least squares.

        Returns:
            A (3, 4) matrix for affine Lab correction (includes bias term).
        """
        # Get reference Lab colors
        ref_swatches = self.reference.get_all()
        ref_lab = np.array([swatch.lab for swatch in ref_swatches])

        # Convert photo colors to Lab
        photo_lab = np.array([rgb_to_lab(swatch) for swatch in self.photo_colors])

        if ref_lab.shape != photo_lab.shape:
            raise ValueError("Reference and photo colors must be the same length")

        # Add bias column for affine transform (N, 4)
        ones = np.ones((photo_lab.shape[0], 1))
        photo_lab_aug = np.hstack([photo_lab, ones])  # shape: (N, 4)

        # Solve the least squares problem: A @ photo_lab_aug.T ≈ ref_lab.T
        A, _, _, _ = np.linalg.lstsq(photo_lab_aug, ref_lab, rcond=None)  # A shape: (4, 3)
        A = A.T  # Final shape: (3, 4)

        # Calculate error
        transformed_photo_lab = photo_lab_aug @ A.T  # shape: (N, 3)
        transformed_photo_rgb = lab_image_to_rgb(transformed_photo_lab)  # shape: (N, 3)
        ref_rgb = np.array([swatch.rgb for swatch in ref_swatches])  # shape: (N, 3)
        error = np.linalg.norm(transformed_photo_rgb - ref_rgb, axis=1).mean()
        print(f"Color correction error (in lab space): {error:.4f}")

        return A

    def apply_color_correction(self, image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        Apply color correction to the input image using the calculated matrix.

        Args:
            image: Input image as a numpy array

        Returns:
            Color-corrected image
        """

        # Convert image to Lab color space
        lab_image = rgb_image_to_lab(image)
        h, w, _ = lab_image.shape

        # Add bias channel of ones
        ones = np.ones((h, w, 1))
        lab_aug = np.concatenate([lab_image, ones], axis=-1)  # shape: (h, w, 4)
        corrected_lab_image = lab_aug @ matrix.T  # shape: (h*w, 3)


        # Convert back to RGB color space
        corrected_rgb_image = lab_image_to_rgb(corrected_lab_image)

        return corrected_rgb_image

    def calculate_swash_error(self) -> float:
        """
        Calculate the error between reference and photo colors.

        Returns:
            Error value
        """
        # Get reference Lab colors
        ref_swatches = self.reference.get_all()
        ref_lab = np.array([swatch.lab for swatch in ref_swatches])

        # Convert photo colors to Lab
        photo_lab = np.array([rgb_to_lab(swatch.color) for swatch in self.photo_colors])

        # Calculate the error (Euclidean distance)
        error = np.linalg.norm(ref_lab - photo_lab, axis=1).mean()
        return error

