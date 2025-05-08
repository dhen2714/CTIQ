import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dataclasses import dataclass


@dataclass
class Catphan700Segment:
    name: str
    indices: np.array
    mean_variance: float
    centre_location_mm: float  # Centre position in mm

    @property
    def centre_index(self) -> int:
        return int(np.median(self.indices))


def visualise_segments(
    ct_array: np.ndarray,
    slice_locations: np.array,
    segments: list[Catphan700Segment],
    pixel_dimensions_mm: tuple[float, float],
    view: str = "coronal",
) -> None:
    """
    Visualises the identified Catphan 700 segments on a central slice (sagittal or coronal) of the CT volume.

    Args:
        ct_array (np.ndarray): A 3D numpy array representing the axial CT slices (num_slices, height, width).
        slice_locations (np.array): A 1D numpy array of z-coordinates (slice locations) in mm.
        segments (List[Catphan700Segment]): A list of Catphan700Segment objects, as returned by
            the locate_all_segments function.
        pixel_dimensions_mm (Tuple[float, float]): The pixel dimensions in mm (width, height).
        view (str, optional): The view to display.  Must be either "coronal" or "sagittal".
            Defaults to "coronal".
    """
    slice_index = ct_array.shape[1] // 2
    if view == "sagittal":
        central_slice = ct_array[:, :, slice_index]
    elif view == "coronal":
        central_slice = ct_array[:, slice_index, :]
    else:
        raise ValueError("View must be either 'coronal' or 'sagittal'.")

    plt.figure(figsize=(8, 6))
    plt.imshow(
        central_slice.T,
        cmap="gray",
        origin="lower",
        aspect=pixel_dimensions_mm[0] / (slice_locations[1] - slice_locations[0]),
    )

    segment_colors = {
        "CTP712": "red",
        "CTP721/CTP723": "green",
        "CTP515": "blue",
        "CTP714": "yellow",
        "CTP682": "magenta",
    }

    for segment in segments:
        segment_slice_indices = segment.indices[0]

        segment_min_z = slice_locations[segment_slice_indices.min()]
        segment_max_z = slice_locations[segment_slice_indices.max()]
        z_start_index = np.where(slice_locations == segment_min_z)[0][0]
        z_end_index = np.where(slice_locations == segment_max_z)[0][0]
        z_length = z_end_index - z_start_index

        y_pos = slice_index
        x_start = z_start_index
        width = z_length
        rect = Rectangle(
            (x_start, y_pos - 5),
            width,
            10,
            color=segment_colors[segment.name],
            alpha=0.3,
            label=segment.name,
            linewidth=1,
        )
        plt.gca().add_patch(rect)

    plt.xlabel("z-index")
    ylabel = "x" if view == "coronal" else "y"
    plt.ylabel(f"{ylabel}-index")
    plt.title(f"Catphan 700 module locations ({view} view)")
    plt.legend(loc="lower right")
    plt.show()


def find_high_variance_centre_location(
    slice_locations: np.array,
    variance_profile: np.array,
    upsample_factor: int = 10,
    min_segment_length_mm: float = 35,
) -> float:
    """
    Locate the CTP721/CTP723 segment which has consistently high variance.

    This function identifies the center location of the CTP721/CTP723 segment within a series of CT slices.
    It works by analyzing the variance profile across the slices to find a region of consistently high
    variance, which is characteristic of this segment.

    Args:
        slice_locations (np.array): An array of z-coordinates (slice locations) in mm for each slice.
        variance_profile (np.array): An array of variance values, one for each slice,
            representing the variance of pixel values within that slice.
        upsample_factor (int, optional):  A factor by which to upsample the slice locations and
            variance profile.  Defaults to 10, which increases precision.
        min_segment_length_mm (float, optional): The minimum length (in mm) of a high-variance segment
            to be considered valid. Defaults to 35. Note length of segements is 40 mm.

    Returns:
        float: The center location (in mm) of the identified high-variance segment.

    Raises:
        RuntimeError: If no high-variance segment is found that meets the minimum length requirement.

    """
    slice_locations_upsampled = np.linspace(
        slice_locations.min(),
        slice_locations.max(),
        upsample_factor * len(slice_locations),
    )
    variance_profile_upsampled = np.interp(
        slice_locations_upsampled, slice_locations, variance_profile
    )

    # Calculate the mean and std of variance values
    mean_var = np.mean(variance_profile_upsampled)
    std_var = np.std(variance_profile_upsampled)

    # Find regions with variance significantly above mean
    high_var_mask = variance_profile_upsampled > (mean_var + 1.5 * std_var)

    # Find the longest continuous segment of high variance
    min_segment_length = (
        np.floor(
            min_segment_length_mm / np.abs(slice_locations[1] - slice_locations[0])
        ).astype(int)
        * upsample_factor
    )
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

    location_range = slice_locations_upsampled[max_segment[0] : max_segment[1]]
    if len(location_range) < min_segment_length:
        raise RuntimeError(
            f"Could not find a high variance segment longer than {min_segment_length_mm}mm."
        )
    return location_range.mean()


def _get_catphan700_module_indices(
    slice_locations: np.array, orientation: str = "HFS"
) -> tuple[int]:
    """
    The holder for the Catphan, if in the axial scan volume, can have a higher
    variance than the CTP721/CTP723 module. This would negatively affect the
    accuracy of find_high_variance_center_location(). This function effectively trims
    one end of the axial scan in an attempt to remove the holder from the axial series.
    Assumes that the axial scan length has been chosen so that the end of the
    Catphan is the end of the scan.
    """
    catphan_modules_length_mm = 200  # combined catphan module length in mm
    loc_min, loc_max = slice_locations.min(), slice_locations.max()
    scan_length = loc_max - loc_min

    # If scan_length is smaller than the catphan module length, assume holder is not in view
    if scan_length < catphan_modules_length_mm:
        return (0, len(slice_locations))

    if orientation.lower().startswith("f"):  # Feet first, e.g, FFS, FFP
        loc_end = slice_locations.min() + catphan_modules_length_mm
        index_end = np.where(slice_locations < loc_end)[0][-1]
        index_start = 0
    else:  # default to Head first orientation
        loc_start = slice_locations.max() - catphan_modules_length_mm
        index_start = np.where(slice_locations > loc_start)[0][0]
        index_end = len(slice_locations)
    return (index_start, index_end)


def locate_all_segments(
    ct_array: np.ndarray,
    slice_locations: np.array,
    orientation: str = "HFS",
    segment_buffer_mm: float = 2,
) -> list[Catphan700Segment]:
    """
    Identify all phantom segments based on the variance profile.
    Each segment is approximately 40mm thick.

    This function identifies the locations of all relevant segments within a Catphan 700 CT
    image. It calculates the variance profile of the CT data and uses this information,
    along with the slice locations, to determine the start and end indices for each segment.
    The function handles both Feet First Supine (FFS) and Head First Supine (HFS) orientations.

    Args:
        ct_array (np.ndarray): A 3D numpy array representing the CT image data.
            The shape is expected to be (num_slices, height, width).
        slice_locations (np.array): A 1D numpy array of z-coordinates (slice locations) in mm,
            corresponding to the first dimension of the ct_array.
        orientation (str, optional): The patient orientation during the scan.
            Defaults to "HFS" (Head First Supine).  Use "FFS" for Feet First Supine.
        segment_buffer_mm (float, optional): A buffer in mm to apply when defining segment boundaries.
            Defaults to 2.  This helps to avoid partial volume effects in segment transition slices.

    Returns:
        list[Catphan700Segment]: A list of Catphan700Segment objects, where each object
            contains information about a detected segment, including its name, indices,
            mean variance, and center location.  The list is sorted by the center
            location of each segment.

    Raises:
        RuntimeError: If a segment is not found within the scan range, indicating a
            potential issue with the orientation or scan parameters.

    """
    variance_profile = np.var(ct_array, axis=(1, 2))

    # First locate the high variance segment (CTP721/CTP723)
    profile_start, profile_end = _get_catphan700_module_indices(
        slice_locations, orientation
    )
    ctp721_723_centre_mm = find_high_variance_centre_location(
        slice_locations[profile_start:profile_end],
        variance_profile[profile_start:profile_end],
    )

    # Create segments list
    segments = []

    # Working backwards from high variance segment to locate anterior segments
    current_centre = ctp721_723_centre_mm
    for segment_name in ["CTP721/CTP723", "CTP515", "CTP714", "CTP682"]:
        # Segments are 40mm, so specify start and end as 20mm from centre, with buffer
        current_end = current_centre + (20 - segment_buffer_mm)
        current_start = current_centre - (20 - segment_buffer_mm)
        segment_indices = np.where(
            (slice_locations >= current_start) & (slice_locations <= current_end)
        )
        if len(segment_indices[0]) < 1:
            raise RuntimeError(
                f"Segment {segment_name} not found within scan range. Make sure orientation is correct!"
            )

        mean_var = np.mean(variance_profile[segment_indices])

        segments.append(
            Catphan700Segment(
                name=segment_name,
                indices=segment_indices,
                mean_variance=mean_var,
                centre_location_mm=current_centre,
            )
        )

        if orientation.lower().startswith("f"):
            current_centre += 40  # feet first
        else:
            current_centre -= 40  # head first (default)

    # Add the posterior segment (CTP712)
    if orientation.lower().startswith("f"):
        current_centre = ctp721_723_centre_mm - 40
    else:
        current_centre = ctp721_723_centre_mm + 40

    current_end = current_centre + (20 - segment_buffer_mm)
    current_start = current_centre - (20 - segment_buffer_mm)
    segment_indices = np.where(
        (slice_locations >= current_start) & (slice_locations <= current_end)
    )
    if len(segment_indices[0]) < 1:
        raise RuntimeError(
            f"Segment {segment_name} not found within scan range. Make sure orientation is correct!"
        )

    segments.append(
        Catphan700Segment(
            name="CTP712",
            indices=segment_indices,
            mean_variance=mean_var,
            centre_location_mm=current_centre,
        )
    )

    # Sort segments by position
    segments.sort(key=lambda x: x.centre_location_mm)
    return segments
