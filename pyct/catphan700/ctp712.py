from ..roi_tools import CircularROIBounds, get_roi
from ..processing import circle_centre_subpixel, pixelate
import numpy as np
from dataclasses import dataclass

PHANTOM_DIAMETER_MM = 200
ROI_RADIAL_POS = 70
ROI_DIAMETER_MM = 10


@dataclass
class UniformityROI:
    name: str
    bounds: CircularROIBounds
    mean: float
    std: float
    values: np.ndarray


def get_uniformity_rois(
    slice_image: np.ndarray,
    pixel_size_mm: tuple[float, float],
    roi_radial_pos_mm=70,
    roi_diameter_mm=25,
    phantom_centre: tuple[int, int] | None = None,
    template_match_padding_mm: float = -30,
) -> dict[str, UniformityROI]:
    if phantom_centre is not None:
        phantom_centre_px = phantom_centre
    else:
        phantom_centre_px = get_CTP712_centre_pixel(
            slice_image, pixel_size_mm, template_match_padding_mm
        )
    roi_radial_pos_px = roi_radial_pos_mm / pixel_size_mm[0]
    roi_diameter_px = roi_diameter_mm / pixel_size_mm[0]

    offsets = {
        "centre": [0, 0],
        "north": [-roi_radial_pos_px, 0],
        "east": [0, roi_radial_pos_px],
        "south": [roi_radial_pos_px, 0],
        "west": [0, -roi_radial_pos_px],
    }

    roi_dict = {}
    for roi_name, offset in offsets.items():
        r, c = offset
        roi_centre = [phantom_centre_px[0] + r, phantom_centre_px[1] + c]
        rbounds = CircularROIBounds(roi_centre[0], roi_centre[1], roi_diameter_px / 2)
        roi = get_roi(slice_image, rbounds)
        roi_dict[roi_name] = UniformityROI(
            roi_name, rbounds, roi.mean(), roi.std(), roi
        )
    return roi_dict


def get_CTP712_centre_pixel(
    slice_image: np.ndarray,
    pixel_size_mm: tuple[float, float],
    template_match_padding_mm: float = -30,
) -> tuple[int, int]:
    circle_radius_px = (PHANTOM_DIAMETER_MM / 2) / pixel_size_mm[0]
    template_padding_px = template_match_padding_mm / pixel_size_mm[0]
    row_centre, col_centre = circle_centre_subpixel(
        slice_image, circle_radius_px, template_padding_px, supersample_factor=2
    )
    return pixelate(row_centre), pixelate(col_centre)
