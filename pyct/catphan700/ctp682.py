from ..roi_tools import ContrastInsertROI, ROIBounds, get_roi
from ..processing import pixelate, circle_centre_subpixel, smooth
import numpy as np
from skimage.feature import match_template
from skimage.filters import gaussian
from dataclasses import dataclass
import warnings

PHANTOM_DIAMETER_MM = 200
CTP682_INSERT_ANGULAR_POSITIONS_DEG = [15, 60, 90, 120, 165, 195, 240, 270, 300, 345]
CTP682_INSERT_DIAMETER_MM = 12.2
CTP682_INSERT_RADIAL_POS_MM = 58.4


@dataclass
class PhantomSegment:
    name: str
    start_idx: int
    end_idx: int
    mean_variance: float
    center_location: float  # Center position in mm


def find_high_variance_segment(
    variance_profile: np.ndarray, min_segment_length: int = 30
) -> tuple[int, int]:
    """
    Locate the CTP721/CTP723 segment which has consistently high variance.
    Returns start and end indices.
    """
    # Smooth the variance to reduce noise
    smoothed = smooth(variance_profile)

    # Calculate the mean and std of variance values
    mean_var = np.mean(smoothed)
    std_var = np.std(smoothed)

    # Find regions with variance significantly above mean
    high_var_mask = smoothed > (mean_var + 1.5 * std_var)

    # Find the longest continuous segment of high variance
    current_length = 0
    max_length = 0
    max_segment = (0, 0)

    start_idx = 0
    for i in range(len(high_var_mask)):
        if high_var_mask[i]:
            if current_length == 0:
                start_idx = i
            current_length += 1
        else:
            if current_length > max_length and current_length >= min_segment_length:
                max_length = current_length
                max_segment = (start_idx, i)
            current_length = 0

    # Check the last segment
    if current_length > max_length and current_length >= min_segment_length:
        max_segment = (start_idx, len(high_var_mask))

    return max_segment


def locate_all_segments(
    variance: np.ndarray, slice_locations: np.ndarray, slice_thickness_mm: float = 1
) -> list[PhantomSegment]:
    """
    Identify all phantom segments based on the variance profile.
    Each segment is approximately 40mm thick.

    Args:
        variance: Array of variance values for each slice
        slice_locations: Array of z-coordinates for each slice in mm
        slice_thickness_mm: Slice thickness in mm
    """
    # Calculate number of slices for 40mm segment
    slices_per_segment = int(40 / slice_thickness_mm)

    # First locate the high variance segment (CTP721/CTP723)
    high_var_start, high_var_end = find_high_variance_segment(variance)

    # Create segments list
    segments = []

    # Working backwards from high variance segment to locate anterior segments
    current_start = high_var_start
    for segment_name in ["CTP721/CTP723", "CTP515", "CTP714", "CTP682"]:
        current_end = min(current_start + slices_per_segment, len(variance))
        current_start = max(0, current_start)  # Ensure start index isn't negative

        mean_var = np.mean(variance[current_start:current_end])
        center_loc = np.mean(slice_locations[current_start:current_end])

        segments.append(
            PhantomSegment(
                name=segment_name,
                start_idx=current_start,
                end_idx=current_end,
                mean_variance=mean_var,
                center_location=center_loc,
            )
        )

        current_start -= slices_per_segment

    # Add the posterior segment (CTP712)
    current_start = high_var_end
    current_end = min(current_start + slices_per_segment, len(variance))

    # Calculate center location for posterior segment
    center_loc = np.mean(slice_locations[current_start:current_end])

    segments.append(
        PhantomSegment(
            name="CTP712",
            start_idx=current_start,
            end_idx=current_end,
            mean_variance=np.mean(variance[current_start:current_end]),
            center_location=center_loc,
        )
    )

    # Sort segments by position
    segments.sort(key=lambda x: x.start_idx)
    return segments


def create_CTP682_template(
    pixel_size_mm: float = 0.1,
    template_padding_mm: float = -30,
) -> tuple[np.ndarray, tuple[float, float]]:
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
        Additional padding around the template in mm.

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
    insert_distance_px = CTP682_INSERT_RADIAL_POS_MM / pixel_size_mm
    insert_radius_px = (CTP682_INSERT_DIAMETER_MM / 2) / pixel_size_mm

    phantom_radius_px = (PHANTOM_DIAMETER_MM / 2) / pixel_size_mm

    # Create coordinate grids
    y, x = np.ogrid[:template_size_px, :template_size_px]
    y = y - center_px
    x = x - center_px

    # Add circles at each insert position
    for angle_deg in CTP682_INSERT_ANGULAR_POSITIONS_DEG:
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


def get_CTP682_contrast_rois(
    slice_image: np.ndarray,
    phantom_centre_px: tuple[float, float],
    pixel_size_mm: tuple[float, float] = (1.0, 1.0),
    roi_margin_factor: float = 2.5,
    supersample_factor: int = 10,
) -> list[ContrastInsertROI]:
    """
    Identifies and returns ROIs around contrast inserts in a CTP682 module.
    Initially finds ROIs depending on angular position from phantom centre.
    Refines calculation of centre pixel of contrast insert ROIs.

    Parameters
    ----------
    slice_image : np.ndarray
        2D array of the CT slice containing the phantom
    phantom_centre_px : Tuple[float, float]
        Centre position of the phantom in pixels (row, column)
    pixel_size_mm : Tuple[float, float]
        Pixel dimensions in mm (row_size, col_size)
    roi_margin_factor : float
        Factor to multiply insert diameter by for ROI size
    supersample_factor : int
        Factor for supersampling in subpixel position refinement

    Returns
    -------
    List[ContrastInsertROI]
        List of ContrastInsertROI objects for each insert
    """
    # Convert distances from mm to pixels
    insert_distance_px = (
        CTP682_INSERT_RADIAL_POS_MM / pixel_size_mm[0],
        CTP682_INSERT_RADIAL_POS_MM / pixel_size_mm[1],
    )
    insert_radius_px = (
        CTP682_INSERT_DIAMETER_MM / 2 / pixel_size_mm[0]
    )  # Using row pixel size

    # Calculate ROI size with margin
    roi_size_px = pixelate(
        CTP682_INSERT_DIAMETER_MM * roi_margin_factor / pixel_size_mm[0]
    )

    insert_locations = []

    # Find each insert's position and create ROI
    for angle_deg in CTP682_INSERT_ANGULAR_POSITIONS_DEG:
        # Convert angle to radians
        angle_rad = np.deg2rad(angle_deg)

        # Calculate expected insert centre position
        delta_row = -insert_distance_px[0] * np.sin(
            angle_rad
        )  # Negative for image coordinates
        delta_col = insert_distance_px[1] * np.cos(angle_rad)

        nominal_row = phantom_centre_px[0] + delta_row
        nominal_col = phantom_centre_px[1] + delta_col

        # Get initial ROI around expected position
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

        # Create ContrastInsertROI object
        insert_roi = ContrastInsertROI(
            name=f"Insert_{angle_deg}deg",
            roi=initial_roi,
            pixel_size_mm=pixel_size_mm,
            rod_centre=(local_row, local_col),
            rod_radius_mm=CTP682_INSERT_DIAMETER_MM / 2,
        )

        insert_locations.append(insert_roi)

    return insert_locations
