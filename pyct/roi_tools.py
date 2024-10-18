import numpy as np
from dataclasses import dataclass


@dataclass
class ROIBounds:
    """Row, column, start, end indices for defining a rectangular ROI within an array."""

    row_start: int
    row_end: int
    col_start: int
    col_end: int

    @property
    def shape(self):
        return (self.row_end - self.row_start, self.col_end - self.col_start)


def get_roi(image: np.ndarray, roi_bounds: ROIBounds) -> np.ndarray:
    """Return ROI given image and ROI bounds"""
    return image[
        roi_bounds.row_start : roi_bounds.row_end,
        roi_bounds.col_start : roi_bounds.col_end,
    ]


def get_roi_from_centre_pixel(
    image: np.ndarray, centre_index: tuple[int], roi_dim: int
) -> ROIBounds:
    """
    Return square ROI from an image given centre pixel index and roi dimension.
    """
    row_start, row_end = int(centre_index[0] - roi_dim / 2), int(
        centre_index[0] + roi_dim / 2
    )
    col_start, col_end = int(centre_index[1] - roi_dim / 2), int(
        centre_index[1] + roi_dim / 2
    )
    return get_roi(image, ROIBounds(row_start, row_end, col_start, col_end))
