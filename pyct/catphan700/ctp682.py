from ..roi_tools import (
    ROIBounds,
    CircularROIBounds,
    get_roi,
    get_background_roi,
    get_roi_bounds_from_centre_pixel,
)
from ..processing import pixelate, circle_centre_subpixel
import numpy as np
from skimage.feature import match_template
from skimage.filters import gaussian
from dataclasses import dataclass
import warnings

PHANTOM_DIAMETER_MM = 200
INNER_DIAMETER_MM = 150
INSERT_ANGULAR_POSITIONS_DEG = [15, 60, 90, 120, 165, 195, 240, 270, 300, 345]
INSERT_NAMES_HF = [  # head first orientation corresponding to angular positions
    "Bone50",
    "Acrylic",
    "Air",
    "PMP",
    "Lung7112",
    "Delrin",
    "Polystyrene",
    "Teflon",
    "Bone20",
    "LDPE",
]
INSERT_NAMES_FF = [  # feet first orientation corresponding to angular positions
    "Lung7112",
    "PMP",
    "Air",
    "Acrylic",
    "Bone50",
    "LDPE",
    "Bone20",
    "Teflon",
    "Polystyrene",
    "Delrin",
]
INSERT2ANGLE_HF = {
    material: angular_pos
    for material, angular_pos in zip(INSERT_NAMES_HF, INSERT_ANGULAR_POSITIONS_DEG)
}
INSERT2ANGLE_FF = {
    material: angular_pos
    for material, angular_pos in zip(INSERT_NAMES_FF, INSERT_ANGULAR_POSITIONS_DEG)
}
INSERT_DIAMETER_MM = 12.2
INSERT_RADIAL_POS_MM = 58.4


@dataclass
class WireRampROI:
    name: str
    roi: np.ndarray
    bounds: ROIBounds


@dataclass
class ContrastInsertROI:
    name: str
    roi: np.ndarray
    pixel_size_mm: tuple[float, float]
    rod_centre: tuple[float, float]
    rod_radius_mm: float
    bounds: ROIBounds  # ROIBounds as defined in the original image

    @property
    def rod_radius_px(self) -> float:
        return self.rod_radius_mm / self.pixel_size_mm[0]

    def get_central_roi_bounds(self, roi_radius_mm: float) -> CircularROIBounds:
        roi_radius_px = roi_radius_mm / self.pixel_size_mm[0]
        return CircularROIBounds(self.rod_centre[0], self.rod_centre[1], roi_radius_px)

    def mean(self, measure_radius_mm: float = 5) -> float:
        measure_roi_bounds = self.get_central_roi_bounds(measure_radius_mm)
        return get_roi(self.roi, measure_roi_bounds).mean()

    def std(self, measure_radius_mm: float = 5) -> float:
        measure_roi_bounds = self.get_central_roi_bounds(measure_radius_mm)
        return get_roi(self.roi, measure_roi_bounds).std()

    def background_mean(self, exclusion_radius_mm: float = 8) -> float:
        exclusion_roi_bounds = self.get_central_roi_bounds(exclusion_radius_mm)
        return get_background_roi(self.roi, exclusion_roi_bounds).mean()

    def background_std(self, exclusion_radius_mm: float = 8) -> float:
        exclusion_roi_bounds = self.get_central_roi_bounds(exclusion_radius_mm)
        return get_background_roi(self.roi, exclusion_roi_bounds).std()

    def cnr(
        self, measure_radius_mm: float = 5, exclusion_radius_mm: float = 8
    ) -> float:
        foreground = self.mean(measure_radius_mm)
        background = self.background_mean(exclusion_radius_mm)
        contrast = foreground - background
        noise = self.background_std(exclusion_radius_mm)
        return contrast / noise


def create_CTP682_template(
    pixel_size_mm: float = 0.1,
    template_padding_mm: float = -30,
) -> np.ndarray:
    """
    Creates a binary template for matching CTP682 insert in the Catphan700.

    Parameters
    ----------
    pixel_size_mm : float
        Size of each pixel in mm.
    phantom_diameter_mm : float
        Diameter of the phantom in mm.
    insert_distance_mm : float
        Distance of inserts from phantom center in mm.
    insert_angles_deg : list[float]
        Angular positions of inserts in degrees.
    insert_diameter_mm : float
        Diameter of the contrast inserts in mm.
    template_padding_mm : float
        Additional padding around the template in mm. Negative values clip
        the template, useful when FOV is smaller than phantom diameter.

    Returns
    -------
    tuple[np.ndarray, tuple[float, float]]
        Binary template array and pixel coordinates of the template center.
    """
    # Calculate template dimensions in pixels
    template_size_mm = PHANTOM_DIAMETER_MM + 2 * template_padding_mm
    template_size_px = int(np.ceil(template_size_mm / pixel_size_mm))
    # Make sure template size is odd for proper centering
    if template_size_px % 2 == 0:
        template_size_px += 1

    # Create empty template
    template = np.zeros((template_size_px, template_size_px))

    # Calculate center point
    center_px = template_size_px / 2

    # Convert insert parameters to pixels
    insert_distance_px = INSERT_RADIAL_POS_MM / pixel_size_mm
    insert_radius_px = (INSERT_DIAMETER_MM / 2) / pixel_size_mm

    phantom_radius_px = (PHANTOM_DIAMETER_MM / 2) / pixel_size_mm

    # Create coordinate grids
    y, x = np.ogrid[:template_size_px, :template_size_px]
    y = y - center_px
    x = x - center_px

    # Add circles at each insert position
    for angle_deg in INSERT_ANGULAR_POSITIONS_DEG:
        # Convert angle to radians
        angle_rad = np.deg2rad(angle_deg)

        # Calculate insert center position
        insert_x = insert_distance_px * np.cos(angle_rad)
        insert_y = insert_distance_px * np.sin(angle_rad)

        # Create circle mask for this insert
        circle_mask = ((x - insert_x) ** 2 + (y - insert_y) ** 2) <= insert_radius_px**2
        template[circle_mask] = 1

    phantom_mask = (x**2 + y**2) > (phantom_radius_px) ** 2
    template[phantom_mask] = 1
    return template


def get_CTP682_centre_pixel(
    slice_image: np.ndarray,
    pixel_size_mm: tuple[float, float],
    template_match_padding_mm: float = -30,
    corr_warning_level: float = 0.7,
) -> tuple[int, int]:
    """
    Calculates the pixel index (row, column) centre of the CTP682 Catphan700 insert module.

    Parameters
    ----------
    slice_image : np.ndarray
        The image of the CTP682 module.
    pixel_size_mm : tuple[float, float]
        Pixel size in mm.
    template_padding_mm : float
        Additional padding around the template in mm. Negative values indicate that the template
        needs to be cropped. To perform template matching, the size of the template must be
        smaller than the dimensions of the slice_image.
    corr_warning_level: float
        Raises a warning if the template matching correlation is below this value.

    Returns
    -------
    tuple[int, int]
        Pixel coordinates of the centre of the CTP682 module.
    """
    t = create_CTP682_template(
        pixel_size_mm[0], template_padding_mm=template_match_padding_mm
    )
    if t.shape[0] > slice_image.shape[0] or t.shape[1] > slice_image.shape[1]:
        raise ValueError(
            f"Template dimensions {t.shape} exceed slice image dimensions {slice_image.shape}. "
            f"Try using a smaller template_match_padding_mm value."
        )

    phantom_mask = (slice_image > -200) & (slice_image < 200)
    match_slice = np.abs(slice_image - slice_image[phantom_mask].mean())
    match_slice = gaussian(match_slice, sigma=2)
    result = match_template(match_slice, t, pad_input=True)
    result = np.abs(result)
    corr_val = result.max()
    if corr_val < corr_warning_level:
        warning_message = f"Correlation between CTP682 template and slice array was less than {corr_warning_level}"
        warnings.warn(warning_message, UserWarning)
    phantom_centre = np.where(result == result.max())
    phantom_centre = phantom_centre[0][0], phantom_centre[1][0]  # row, column
    return phantom_centre


def get_angular_position(material: str, orientation: str = "HFS"):
    if orientation.lower().startswith("h"):
        mapping = INSERT2ANGLE_HF
    elif orientation.lower().startswith("f"):
        mapping = INSERT2ANGLE_FF
    else:
        raise ValueError("Orientation must be 'HF' or 'FF'.")

    if material not in INSERT_NAMES_HF:
        raise ValueError(f"Specified contrast material must be in {INSERT_NAMES_HF}")

    return mapping[material]


def get_contrast_roi(
    slice_image: np.ndarray,
    material: str,
    phantom_centre_px: tuple[int, int],
    pixel_size_mm: float,
    roi_margin_factor: float = 2.5,
    orientation: str = "HFS",
    supersample_factor: int = 10,
) -> ContrastInsertROI:

    angle_deg = get_angular_position(material, orientation)

    insert_distance_px = (
        INSERT_RADIAL_POS_MM / pixel_size_mm[0],
        INSERT_RADIAL_POS_MM / pixel_size_mm[1],
    )
    insert_radius_px = INSERT_DIAMETER_MM / 2 / pixel_size_mm[0]  # Using row pixel size

    roi_size_px = pixelate(INSERT_DIAMETER_MM * roi_margin_factor / pixel_size_mm[0])

    angle_rad = np.deg2rad(angle_deg)
    delta_row = -insert_distance_px[0] * np.sin(
        angle_rad
    )  # Negative for image coordinates
    delta_col = insert_distance_px[1] * np.cos(angle_rad)

    nominal_row = phantom_centre_px[0] + delta_row
    nominal_col = phantom_centre_px[1] + delta_col

    initial_bounds = ROIBounds(
        row_start=pixelate(nominal_row - roi_size_px / 2),
        row_end=pixelate(nominal_row + roi_size_px / 2),
        col_start=pixelate(nominal_col - roi_size_px / 2),
        col_end=pixelate(nominal_col + roi_size_px / 2),
    )

    # Extract ROI and refine centre position
    initial_roi = get_roi(slice_image, initial_bounds)

    # Fine-tune the centre position within the ROI
    local_row, local_col = circle_centre_subpixel(
        initial_roi, insert_radius_px, supersample_factor
    )

    return ContrastInsertROI(
        name=material,
        roi=initial_roi,
        pixel_size_mm=pixel_size_mm,
        rod_centre=(local_row, local_col),
        rod_radius_mm=INSERT_DIAMETER_MM / 2,
        bounds=initial_bounds,
    )


def get_CTP682_contrast_rois(
    slice_image: np.ndarray,
    pixel_size_mm: tuple[float, float] = (1.0, 1.0),
    orientation: str = "HFS",
    roi_margin_factor: float = 2.5,
    supersample_factor: int = 10,
    template_match_padding_mm: float = -30,
    corr_warning_level: float = 0.7,
    phantom_centre: tuple[int, int] | None = None,
) -> dict[str, ContrastInsertROI]:
    """
    Identifies and returns ROIs around contrast inserts in a CTP682 module.
    Initially finds ROIs depending on angular position from phantom centre.
    Refines calculation of centre pixel of contrast insert ROIs.

    Parameters
    ----------
    slice_image : np.ndarray
        2D array of the CT slice containing the phantom
    pixel_size_mm : Tuple[float, float]
        Pixel dimensions in mm (row_size, col_size)
    orientation : str
        The patient orientation during the scan. Defaults to "HFS" (Head First Supine).
        Use "FFS" for Feet First Supine.
    roi_margin_factor : float
        Factor to multiply insert diameter by for ROI size
    supersample_factor : int
        Factor for supersampling in subpixel position refinement
    template_padding_mm : float
        For finding phantom centre, additional padding around the template in mm. Negative
        values indicate that the templateneeds to be cropped. To perform template matching,
        the size of the template must be smaller than the dimensions of the slice_image.
    corr_warning_level: float
        For finding phantom centre, raises a warning if the template matching correlation is
        below this value.
    phantom_centre : tuple[int, int] | None
        Optional phantom centre coordinates as (row, col) pixels. If provided, skips
        automatic phantom centre calculation.

    Returns
    -------
    dict[str, ContrastInsertROI]
        Dictionary of ContrastInsertROI objects for each insert, keyed by contrast material name
    """
    if phantom_centre is not None:
        phantom_centre_px = phantom_centre
    else:
        phantom_centre_px = get_CTP682_centre_pixel(
            slice_image, pixel_size_mm, template_match_padding_mm, corr_warning_level
        )

    contrast_insert_dict = {}

    # Find each insert's position and create ROI
    for contrast_material in INSERT_NAMES_HF:

        # Create ContrastInsertROI object
        insert_roi = get_contrast_roi(
            slice_image,
            contrast_material,
            phantom_centre_px,
            pixel_size_mm,
            roi_margin_factor,
            orientation,
            supersample_factor,
        )

        contrast_insert_dict[contrast_material] = insert_roi

    return contrast_insert_dict


def get_wire_ramp_roi(
    slice_image: np.ndarray,
    phantom_centre: tuple[int, int],
    pixel_size_mm: float,
    which: str = "top",
) -> WireRampROI:
    centre_offset_mm = 40
    centre_offset_px = pixelate(centre_offset_mm / pixel_size_mm)

    roi_width_mm = 80
    roi_width_px = pixelate(roi_width_mm / pixel_size_mm)

    roi_height_mm = 5
    roi_height_px = pixelate(roi_height_mm / pixel_size_mm)

    if which == "top":
        roi_centre = [phantom_centre[0] - centre_offset_px, phantom_centre[1]]
    elif which == "bottom":
        roi_centre = [phantom_centre[0] + centre_offset_px, phantom_centre[1]]
    else:
        raise ValueError("'which' argument must be 'top' or 'bottom'.")

    roi_bounds = get_roi_bounds_from_centre_pixel(
        roi_centre, [roi_height_px, roi_width_px]
    )
    roi = get_roi(slice_image, roi_bounds)
    return WireRampROI(which, roi, roi_bounds)
