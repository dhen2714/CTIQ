from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from scipy.signal.windows import hann
from numba import njit, prange
import numpy as np
from dataclasses import dataclass
import warnings


@dataclass
class ESF:
    """Container class for calculated ESF."""

    esf: np.array
    r: np.array


@dataclass
class TTF:
    """Container class for calculated TTF."""

    mtf: np.array
    esf: np.array
    lsf: np.array
    f: np.array
    r: np.array

    @property
    def ttf(self) -> np.array:
        return self.mtf


@njit(parallel=True, fastmath=True)
def rebin_calc_esf(
    sample_positions: np.array, flattened_roi: np.array, upsampled_distances: np.array
) -> np.array:
    """
    Rebin ROI pixel values according to their distance from the edge.
    Returns ESF.
    Uses numba for faster looping.
    """
    n = len(sample_positions)
    esf = np.zeros(n)
    for i in prange(n):
        bin_val = sample_positions[i]
        inds = np.where(upsampled_distances == bin_val)[0]
        esf[i] = flattened_roi[inds].mean()
    return esf


def esf2ttf(esf: np.array, sample_positions: np.array) -> TTF:
    """
    Convert Edge Spread Function to Transfer Function.

    Parameters:
        esf: Edge Spread Function values
        sample_positions: Corresponding position values

    Returns:
        TTF object containing the transfer function and intermediate results
    """
    sample_period = sample_positions[1] - sample_positions[0]
    # Differentiate ESF to get LSF
    lsf = np.convolve(esf, [-1, 1], mode="valid")
    lsf = np.append([lsf[0]], lsf)  # make lsf same length as esf
    # Apply Hann windowing to flatten out tails of LSF
    hann_window = esf_hann_window(esf)
    lsf = lsf * hann_window

    # Compute FFT
    LSF = fft(lsf)

    # Safe normalization
    dc_component = np.abs(LSF)[0]
    if dc_component < 1e-10:  # Protect against division by very small numbers
        raise ValueError("DC component is too small for reliable TTF calculation")
    # Normalise MTF using DC component
    MTF = np.abs(LSF) / dc_component

    frequencies = fftfreq(len(lsf), sample_period)

    # Return only positive side of MTF
    fn_index = int(len(MTF) / 2)
    frequencies, MTF = frequencies[:fn_index], MTF[:fn_index]

    return TTF(MTF, esf, lsf, frequencies, sample_positions)


def esf_hann_window(esf: np.array, window_width: int = 15) -> np.array:
    """Return Hann window that flattens the tails of the LSF, using ESF as input."""
    # Find the centre of the edge and edge width
    esf_max, esf_min = esf.max(), esf.min()
    edge_positions = np.argsort(abs(esf - (esf_max + esf_min) / 2))
    edge_centre_index = edge_positions[0]
    # 0.15 and 0.85 values taken from imQuest
    if np.mean(esf[:edge_centre_index]) > np.mean(esf[edge_centre_index:]):
        edge_upper_index = np.where(esf > esf_min + 0.85 * (esf_max - esf_min))[0][-1]
        edge_lower_index = np.where(esf < esf_min + 0.15 * (esf_max - esf_min))[0][0]
    else:
        edge_upper_index = np.where(esf > esf_min + 0.85 * (esf_max - esf_min))[0][0]
        edge_lower_index = np.where(esf < esf_min + 0.15 * (esf_max - esf_min))[0][-1]
    # Use width of edge to define the Hann window width
    window_width = window_width * abs(edge_upper_index - edge_lower_index)
    hann_start_index = np.max([edge_centre_index - window_width, 0])
    hann_end_index = np.min([edge_centre_index + window_width, len(esf)])
    hann_window_length = hann_end_index - hann_start_index
    hann_window = np.zeros(len(esf))
    hann_window[hann_start_index:hann_end_index] = hann(hann_window_length)
    return hann_window


def calculate_radial_esf(
    roi: np.ndarray,
    subpixel_center: tuple[float],
    max_sample_radius: float,
    pixel_size_mm: float = 1,
    supersample_factor: int = 10,
) -> ESF:
    """ESF calculated within the ROI around the subpixel center (row, column)."""
    # Calculate each point's radial distance from center
    numrows_roi, numcols_roi = roi.shape
    row_coords, col_coords = np.arange(numrows_roi) + 0.5, np.arange(numcols_roi) + 0.5
    X, Y = np.meshgrid(col_coords, row_coords)
    dists_from_centre = np.sqrt(
        (X - subpixel_center[1]) ** 2 + (Y - subpixel_center[0]) ** 2
    )

    # Re-bin ESF only for pixels within max_sample_radius
    subroi_indices = np.where(dists_from_centre <= max_sample_radius)
    subroi_distances = dists_from_centre[subroi_indices]
    subroi_distances_upsampled = (
        np.round(subroi_distances * supersample_factor) / supersample_factor
    )
    subroi = roi[subroi_indices]
    sample_positions_px = np.arange(
        0, int(max_sample_radius) + 1 / supersample_factor, 1 / supersample_factor
    )
    sample_positions_mm = pixel_size_mm * sample_positions_px
    esf = rebin_calc_esf(
        np.unique(subroi_distances_upsampled),
        subroi.flatten(),
        subroi_distances_upsampled,
    )
    # Unique distances are less densely sampled near centre, so sampling is not uniform
    # Apply linear interpolation to get uniform ESF sampling
    esf_uniform_sample = np.interp(
        sample_positions_px, np.unique(subroi_distances_upsampled), esf
    )
    return ESF(esf_uniform_sample, sample_positions_mm)


def calculate_ttf(
    roi: np.ndarray,
    subpixel_center: tuple[float],
    max_sample_radius: float,
    pixel_size_mm: float = 1,
    supersample_factor: int = 10,
) -> TTF:
    """
    Radial ESF calculated within the ROI around the subpixel center (row, column).

    Computes the radial Edge Spread Function (ESF) within the ROI around a subpixel center,
    then converts it to a TTF measurement.

    Parameters
    ----------
    roi : numpy.ndarray
        Region of interest image data.
    subpixel_center : tuple[float]
        Coordinates of the center point (row, column) with subpixel precision.
    max_sample_radius : float
        Maximum distance in pixels from the center to sample the ESF. Set to ~2x insert
        radius in pixels.
    pixel_size_mm : float, optional
        Physical size of a pixel in millimeters, default=1.
    supersample_factor : int, optional
        Factor by which to supersample the data for subpixel resolution, default=10.

    Returns
    -------
    TTF : object
        Task transfer function container object containing the calculated TTF data.
    """
    esf = calculate_radial_esf(
        roi, subpixel_center, max_sample_radius, pixel_size_mm, supersample_factor
    )
    return esf2ttf(esf.esf, esf.r)


def get_cutoff_frequency(
    spatial_frequencies: np.array,
    mtf_values: np.array,
    cutoff_point: float,
    mtf_threshold: float = 0.01,
) -> float:
    """
    Finds the cutoff frequency at a specified MTF value using interpolation,
    considering only MTF values up to the point where it first drops below a threshold.

    Parameters
    ----------
    spatial_frequencies : np.array
        1D array of spatial frequency values (assumed to be sorted).
        mtf_values (np.array): 1D array of corresponding MTF values
        (assumed to be sorted according to spatial frequencies).
    cutoff_point : float
        The MTF value at which to find the cutoff frequency (e.g., 0.5 for 50%).
    mtf_threshold : float
        The minimum MTF value to consider for interpolation.
        Data points after the MTF first drops below this threshold
        will be excluded. Defaults to 0.01.

    Returns
    -------
    float or None
        The cutoff frequency at the specified MTF value, or None if the
        cutoff point is outside the relevant range of MTF values.
    """
    # Find the index of the first occurrence where MTF drops below the threshold
    below_threshold_index = np.argmax(mtf_values < mtf_threshold)

    # If MTF never drops below the threshold, use the entire array
    if below_threshold_index == 0 and not (mtf_values < mtf_threshold).any():
        valid_indices = np.arange(len(mtf_values))
    else:
        valid_indices = np.arange(
            below_threshold_index + 1
        )  # Include the point where it drops below

    if len(valid_indices) < 2:
        warnings.warn(
            f"The first MTF value is already below the threshold of {mtf_threshold}.",
            UserWarning,
        )
        return None

    # Use only the spatial frequencies and MTF values up to the threshold point
    valid_spatial_frequencies = spatial_frequencies[valid_indices]
    valid_mtf_values = mtf_values[valid_indices]

    # Create an interpolation function using the filtered data
    interpolation_function = interp1d(
        valid_mtf_values,
        valid_spatial_frequencies,
        bounds_error=False,
        fill_value="extrapolate",
    )

    # Find the cutoff frequency using the interpolation function
    cutoff_frequency = interpolation_function(cutoff_point)

    # It's good practice to check if the cutoff point was within the filtered MTF range
    if (cutoff_point < np.min(valid_mtf_values)) or (
        cutoff_point > np.max(valid_mtf_values)
    ):
        cutoff_warning_message = (
            f"The cutoff point of {cutoff_point} is outside"
            + "the range of filtered MTF values (down to"
            + " first occurrence below {mtf_threshold}). "
            + "The returned cutoff frequency might be an "
            + "extrapolation based on the filtered data."
        )
        warnings.warn(
            cutoff_warning_message,
            UserWarning,
        )

    return cutoff_frequency
