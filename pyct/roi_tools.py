from .processing import pixelate
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from dataclasses import dataclass
from matplotlib.axes import Axes


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


@dataclass
class CircularROIBounds:
    """Defines a circular ROI within an array."""

    centre_row: float
    centre_col: float
    radius: float

    @property
    def bounding_box(self):
        """Returns the bounding box (ROIBounds) of the circular ROI."""
        row_start = pixelate(self.centre_row - self.radius)
        row_end = pixelate(self.centre_row + self.radius + 1)
        col_start = pixelate(self.centre_col - self.radius)
        col_end = pixelate(self.centre_col + self.radius + 1)
        return ROIBounds(row_start, row_end, col_start, col_end)


def get_roi(image: np.ndarray, roi_bounds: ROIBounds | CircularROIBounds) -> np.ndarray:
    """Return ROI given image and ROI bounds"""
    if isinstance(roi_bounds, ROIBounds):
        return image[
            roi_bounds.row_start : roi_bounds.row_end,
            roi_bounds.col_start : roi_bounds.col_end,
        ]
    elif isinstance(roi_bounds, CircularROIBounds):
        rows, cols = np.indices(image.shape[:2])
        mask = (rows - roi_bounds.centre_row) ** 2 + (
            cols - roi_bounds.centre_col
        ) ** 2 <= roi_bounds.radius**2
        # Note the returned value will not be rectangular.
        return image[mask]
    else:
        raise TypeError("roi_bounds must be ROIBounds or CircularROIBounds object.")


def draw_rois(
    ax: Axes,
    rois: ROIBounds | CircularROIBounds | list[ROIBounds | CircularROIBounds],
    colors: str | list[str] = "red",
    linewidth: float = 2,
    alpha: float = 1.0,
) -> None:
    """
    Draw one or more ROIBounds on a matplotlib axis with an imshow plot.

    Parameters
    ----------
    ax : plt.Axes
        The matplotlib axes object where the image is displayed
    rois : ROIBounds or List[ROIBounds]
        Single ROIBounds object or list of ROIBounds to draw
    colors : str or List[str], optional
        Color(s) for the ROI boundaries. If None, uses default color cycle
    linewidth : float, optional
        Width of the boundary lines
    alpha : float, optional
        Transparency of the boundary lines (0.0 to 1.0)
    """
    # Convert single ROI to list for uniform processing
    if not isinstance(rois, list):
        rois = [rois]

    # Handle single color/label case
    if isinstance(colors, str):
        colors = [colors] * len(rois)
    elif colors is None:
        colors = [None] * len(rois)

    # Draw each ROI
    for roi, color in zip(rois, colors):
        if isinstance(roi, ROIBounds):
            # Create rectangle vertices
            vertices = np.array(
                [
                    [roi.col_start, roi.row_start],  # Top left
                    [roi.col_end, roi.row_start],  # Top right
                    [roi.col_end, roi.row_end],  # Bottom right
                    [roi.col_start, roi.row_end],  # Bottom left
                    [
                        roi.col_start,
                        roi.row_start,
                    ],  # Back to top left to close the rectangle
                ]
            )

            # Plot the boundary
            ax.plot(
                vertices[:, 0],
                vertices[:, 1],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
            )
        elif isinstance(roi, CircularROIBounds):
            circle = patches.Circle(
                (
                    roi.centre_col,
                    roi.centre_row,
                ),  # Note the order (x, y) for matplotlib
                roi.radius,
                edgecolor=color,
                facecolor="none",
                linewidth=linewidth,
                alpha=alpha,
            )
            ax.add_patch(circle)
        else:
            raise TypeError(f"Unsupported ROI type: {type(roi)}")


def get_roi_bounds_from_centre_pixel(
    centre_index: tuple[int, int], roi_dim: int
) -> ROIBounds:
    row_start, row_end = int(centre_index[0] - roi_dim / 2), int(
        centre_index[0] + roi_dim / 2
    )
    col_start, col_end = int(centre_index[1] - roi_dim / 2), int(
        centre_index[1] + roi_dim / 2
    )
    return ROIBounds(row_start, row_end, col_start, col_end)


def get_roi_from_centre_pixel(
    image: np.ndarray, centre_index: tuple[int, int], roi_dim: int
) -> np.ndarray:
    """
    Return square ROI from an image given centre pixel index and roi dimension.
    """
    roi_bounds = get_roi_bounds_from_centre_pixel(centre_index, roi_dim)
    return get_roi(image, roi_bounds)
