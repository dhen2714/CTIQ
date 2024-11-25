from ..processing import circle_centre_subpixel
from ..roi_tools import (
    get_roi,
    get_roi_bounds_from_centre_pixel,
    ROIBounds,
)
from skimage.transform import rescale
from skimage.feature import match_template
from skimage.filters import gaussian
import pandas as pd
import numpy as np

INSERT_DIAMETER_MM = 25
INSERT_DIST_FROM_CENTRE_MM = 50
INSERT_ANGULAR_POSITION_DEG = 45
TEMPLATE_SIZE_MM = 120
TEMPLATE_PIX_SIZE = 0.1
NPS_ROI_DIM_MM = 35


class PhantomDetectionError(ValueError):
    """Exception raised when phantom template cannot be reliably detected in an image."""

    pass


def _get_insert_inds(insert_centre_pos: tuple[float], FX: np.ndarray, FY: np.ndarray):
    FY, FX = FY - insert_centre_pos[0], FX - insert_centre_pos[1]
    return np.where(
        np.sqrt(FX**2 + FY**2) < (INSERT_DIAMETER_MM / 2 / TEMPLATE_PIX_SIZE)
    )


def get_insert_template():
    """
    Template for finding the centre of the CeLT phantom in an axial slice.
    """
    template_dim = int(TEMPLATE_SIZE_MM / TEMPLATE_PIX_SIZE)
    template = np.zeros((template_dim, template_dim))
    template_centre = (template_dim - 1) / 2
    angular_pos = np.radians(INSERT_ANGULAR_POSITION_DEG)
    row_offset = INSERT_DIST_FROM_CENTRE_MM * np.sin(angular_pos) / TEMPLATE_PIX_SIZE
    col_offset = INSERT_DIST_FROM_CENTRE_MM * np.cos(angular_pos) / TEMPLATE_PIX_SIZE
    centre = template_centre, template_centre
    top_left = template_centre - row_offset, template_centre - col_offset
    top_right = template_centre - row_offset, template_centre + col_offset
    bottom_left = template_centre + row_offset, template_centre - col_offset
    bottom_right = template_centre + row_offset, template_centre + col_offset
    FX, FY = np.meshgrid(np.arange(template_dim), np.arange(template_dim))

    for insert_centre_pos in [
        centre,
        top_left,
        top_right,
        bottom_left,
        bottom_right,
    ]:
        insert_indices = _get_insert_inds(insert_centre_pos, FX, FY)
        template[insert_indices] = 1
    return template


def get_phantom_centre(
    axial_slice: np.ndarray, template: np.ndarray, threshold: float = 0.5
) -> tuple[int]:
    """
    Matches given template to an axial slice, returns (row, column) index of best
    match between slice and template. template must have smaller dimensions than
    axial_slice.
    """
    # Attempt to only get mean of phantom HU vals
    phantom_mask = (axial_slice > -200) & (axial_slice < 200)
    axial_slice = np.abs(axial_slice - axial_slice[phantom_mask].mean())
    axial_slice = gaussian(axial_slice, sigma=2)
    result = match_template(axial_slice, template, pad_input=True)
    result = np.abs(result)
    corr_val = result.max()
    if corr_val > threshold:
        phantom_centre = np.where(result == result.max())
        phantom_centre = phantom_centre[0][0], phantom_centre[1][0]
    else:
        raise PhantomDetectionError(
            f"Failed to detect phantom in image. Best correlation ({corr_val:.3f}) "
            f"below threshold ({threshold}). This may indicate the image doesn't "
            "contain the expected phantom or has poor contrast/quality."
        )

    return phantom_centre


class Celt:

    def __init__(
        self,
        pixel_array3d: np.ndarray,
        slice_locations: np.array,
        pixel_dim_mm: tuple[float],
        template_match_threshold: float = 0.5,
    ) -> None:
        self.array = pixel_array3d
        self.slice_locations = slice_locations
        self.pixel_dim_mm = pixel_dim_mm
        self.match_threshold = template_match_threshold
        self.slice_idx_2_template_centre = dict()
        template = get_insert_template()
        # Rescale the template to the pixel size of the image
        self.template = rescale(template, TEMPLATE_PIX_SIZE / pixel_dim_mm[0])
        angular_pos = np.radians(INSERT_ANGULAR_POSITION_DEG)
        self.nps_roi_dim = int(NPS_ROI_DIM_MM / pixel_dim_mm[0])
        self.nps_roi_offset = int(
            INSERT_DIST_FROM_CENTRE_MM * np.sin(angular_pos) / pixel_dim_mm[0]
        )

        self.background_means = dict()
        self.background_stddevs = dict()

        self.background_roi_bounds = None
        self.contrast_insert_roi_bounds = None

    def measure_background(
        self,
        slice_indices: None | int | list[int] = None,
        roi_bounds: None | list[ROIBounds] = None,
    ) -> pd.DataFrame:
        if slice_indices is None:
            slice_indices = range(len(self.array))
        elif isinstance(slice_indices, int):
            slice_indices = [slice_indices]
        else:
            slice_indices = np.array(slice_indices)
        # Initialise roi bounds for background if they haven't been set previously
        self.background_roi_bounds = roi_bounds

        measurements = []
        for slice_index in slice_indices:
            rois = []
            slice_array = self.array[slice_index]
            if self.background_roi_bounds is None:
                self.background_roi_bounds = self.find_background_roi_bounds(
                    slice_index
                )
            for roi_bounds in self.background_roi_bounds:
                roi = get_roi(slice_array, roi_bounds)
                rois.append(roi)
            rois = np.array(rois)
            measurement = {
                "mean": rois.mean(),
                "standard_dev": rois.std(),
                "max": rois.max(),
                "min": rois.min(),
            }
            measurements.append(measurement)
        return pd.DataFrame(measurements, index=slice_indices)

    def find_background_roi_bounds(self, slice_index: int) -> list[ROIBounds]:
        axial_slice = self.array[slice_index]
        background_roi_bounds = []

        phantom_centre = get_phantom_centre(
            axial_slice, self.template, threshold=self.match_threshold
        )

        top = phantom_centre[0] - self.nps_roi_offset, phantom_centre[1]
        left = phantom_centre[0], phantom_centre[1] + self.nps_roi_offset
        bottom = phantom_centre[0] + self.nps_roi_offset, phantom_centre[1]
        right = phantom_centre[0], phantom_centre[1] - self.nps_roi_offset

        for background_roi_centre in [top, left, bottom, right]:
            roi_bounds = get_roi_bounds_from_centre_pixel(
                background_roi_centre, self.nps_roi_dim
            )
            background_roi_bounds.append(roi_bounds)
        return background_roi_bounds

    def measure_contrast_inserts(
        self,
        slice_indices: None | int | list[int] = None,
        roi_bounds: None | list[ROIBounds] = None,
    ) -> pd.DataFrame:
        if slice_indices is None:
            slice_indices = range(len(self.array))
        elif isinstance(slice_indices, int):
            slice_indices = [slice_indices]
        else:
            slice_indices = np.array(slice_indices)
        # Initialise roi bounds for contrast inserts if they haven't been set previously
        self.contrast_insert_roi_bounds = roi_bounds

        measurements = []
        for slice_index in slice_indices:
            slice_measurements = {}
            slice_array = self.array[slice_index]
            if self.contrast_insert_roi_bounds is None:
                self.contrast_insert_roi_bounds = self.find_contrast_insert_roi_bounds(
                    slice_index
                )
            for i, roi_bounds in enumerate(self.contrast_insert_roi_bounds):
                roi = get_roi(slice_array, roi_bounds)
                measurement_key = f"roi_{i}"
                slice_measurements[f"{measurement_key}_mean"] = roi.mean()
                slice_measurements[f"{measurement_key}_standard_dev"] = roi.std()
                slice_measurements[f"{measurement_key}_max"] = roi.max()
                slice_measurements[f"{measurement_key}_min"] = roi.min()

            measurements.append(slice_measurements)

        return pd.DataFrame(measurements, index=slice_indices)

    def find_contrast_insert_roi_bounds(self, slice_index: int) -> list[ROIBounds]:
        axial_slice = self.array[slice_index]
        insert_radius_px = (INSERT_DIAMETER_MM / 2) / self.pixel_dim_mm[0]
        outer_contrast_insert_roi_dim = int(4 * insert_radius_px)
        measurement_roi_dim = int(0.75 * insert_radius_px)
        px_offset = self.nps_roi_offset

        centre = get_phantom_centre(
            axial_slice, self.template, threshold=self.match_threshold
        )
        background_mean = self.measure_background(slice_index).loc[slice_index]["mean"]

        roi_centres = {
            "centre": (centre[0], centre[1]),
            "top_right": (centre[0] - px_offset, centre[1] + px_offset),
            "bottom_right": (centre[0] + px_offset, centre[1] + px_offset),
            "bottom_left": (centre[0] + px_offset, centre[1] - px_offset),
            "top_left": (centre[0] - px_offset, centre[1] - px_offset),
        }
        contrast_insert_roi_bounds = []
        for roi_centre in roi_centres.values():
            roi_bounds = get_roi_bounds_from_centre_pixel(
                roi_centre, outer_contrast_insert_roi_dim
            )
            roi = get_roi(axial_slice, roi_bounds)

            subpixel_centre = circle_centre_subpixel(
                roi - background_mean, insert_radius_px, supersample_factor=5
            )
            new_centre = (
                np.array(roi_centre) - outer_contrast_insert_roi_dim / 2
            ) + np.array(subpixel_centre, dtype=int)
            roi_bounds = get_roi_bounds_from_centre_pixel(
                new_centre,
                measurement_roi_dim,
            )
            contrast_insert_roi_bounds.append(roi_bounds)

        return contrast_insert_roi_bounds
