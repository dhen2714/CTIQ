from ..processing import smooth
import numpy as np
from dataclasses import dataclass


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
