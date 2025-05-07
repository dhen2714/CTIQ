import numpy as np
from dataclasses import dataclass


@dataclass
class Catphan700Segment:
    name: str
    indices: np.array
    mean_variance: float
    center_location_mm: float  # Center position in mm


def find_high_variance_centre_location(
    slice_locations: np.array,
    variance_profile: np.array,
    upsample_factor: int = 10,
    min_segment_length_mm: float = 35,
) -> tuple[int, int]:
    """
    Locate the CTP721/CTP723 segment which has consistently high variance.
    Returns start and end indices.
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
    accuracy of find_high_variance_segment(). This function effectively trims one
    end of the axial scan in an attempt to remove the holder from the axial series.
    Assumes that the axial scan length has been chosen so that the end of the
    Catphan is the end of the scan.
    """
    catphan_modules_length_mm = 200  # combined catphan module length in mm
    loc_min, loc_max = slice_locations.min(), slice_locations.max()
    scan_length = loc_max - loc_min

    # If scan_length is smaller than the catphan module length, assume holder is not in view
    if scan_length < catphan_modules_length_mm:
        return (0, len(slice_locations))

    if orientation.lower().startswith("f"):
        loc_end = slice_locations.min() + catphan_modules_length_mm
        index_end = np.where(slice_locations < loc_end)[0][-1]
        index_start = 0
    else:
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

    Args:
        variance: Array of variance values for each slice
        slice_locations: Array of z-coordinates for each slice in mm
        slice_thickness_mm: Slice thickness in mm
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
        # Segments are 40mm, so specify start and end as 20mm from centre, with buffer to avoid partial volume
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
                center_location_mm=current_centre,
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
            center_location_mm=current_centre,
        )
    )

    # Sort segments by position
    segments.sort(key=lambda x: x.center_location_mm)
    return segments
