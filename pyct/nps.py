from .roi_tools import ROIBounds, get_roi
from .processing import cartesian2polar, detrend, rebin_by_pitch, rebin, smooth
import numpy as np
from dataclasses import dataclass


@dataclass
class NPS1D:
    """Container class for 1D noise power spectrum."""

    nps: np.ndarray
    f: np.ndarray
    num_rois: int = None  # Number of ROIs used to calculate NPS
    roi_dimensions: tuple[int] = None  # num_columns (nx), num_rows (ny)
    pad_size: int = None  # Size of padding used on subrois to calculate NPS

    def rebin_by_count(self, n):
        """
        Rebin the NPS to a specified number of bins.

        Parameters:
            n (int): Number of bins for rebinning.

        Returns:
            NPS1D: Rebinned noise power spectrum with n bins.
        """
        fnew, npsnew = rebin(self.f, self.nps, num_bins=n)
        return NPS1D(npsnew, fnew)

    def rebin_by_pitch(self, freq_pitch: float) -> "NPS1D":
        """
        Rebin the NPS to a specified frequency pitch.

        Parameters:
            freq_pitch (float): Desired frequency pitch (e.g., 0.1 mm^-1).
                              Must be larger than the original frequency spacing.

        Returns:
            NPS1D: Rebinned noise power spectrum with specified frequency pitch.

        Raises:
            ValueError: If freq_pitch is smaller than original frequency spacing.
        """
        fnew, npsnew = rebin_by_pitch(self.f, self.nps, freq_pitch)
        return NPS1D(npsnew, fnew)

    @property
    def favg(self):
        """Average spatial frequency of NPS"""
        probability_dist = np.abs(self.nps) / np.sum(np.abs(self.nps))
        return np.dot(probability_dist, np.abs(self.f))

    @property
    def fpeak(self):
        """Frequency at which NPS peaks"""
        smoothed_nps = smooth(self.nps, window_size=3)
        return np.abs(self.f[np.where(smoothed_nps == smoothed_nps.max())][0])

    def __len__(self):
        return len(self.nps)


@dataclass
class NPS2D:
    """Container class for 2D noise power spectrum."""

    nps: np.ndarray
    fx: np.array  # dimension across different columns
    fy: np.array  # dimension across different rows
    num_rois: int = None  # Number of ROIs used to calculate NPS
    roi_dimensions: tuple[int] = None  # num_columns (nx), num_rows (ny)
    pixel_value_mean: float = None
    pixel_value_var: float = None
    pad_size: int = None  # Size of padding used on subrois to calculate NPS

    def get_radial(self) -> NPS1D:
        """Returns radially averaged 1D NPS."""
        FX, FY = np.meshgrid(self.fx, self.fy)
        FR, _ = cartesian2polar(FX, FY)
        max_freq = np.min([FX.max(), FY.max()])
        nps_inds = np.where(FR <= max_freq)
        num_radial_samples = int(len(self.fx) / 2)
        fr, npsr = FR[nps_inds].flatten(), self.nps[nps_inds].flatten()
        fr, npsr = rebin(fr, npsr, num_bins=num_radial_samples)
        return NPS1D(npsr, fr, self.num_rois, self.roi_dimensions)

    def get_horizontal(
        self, num_slices: int = 15, exclude_zero_axis: bool = False
    ) -> NPS1D:
        """
        Returns one-sided 1D horizontal NPS. Averages over a stack in the centre of 2D NPS.

        Parameters:
            num_slices (int): Number of rows to average together.
            exclude_zero_axis (bool): If True, excludes the fy=0 axis when averaging.

        Returns:
            NPS1D: One-sided 1D horizontal noise power spectrum.
        """
        num_rows, num_cols = self.nps.shape
        centre_index = int(num_rows / 2)
        half_slices = int(num_slices / 2)

        # Define row indices to include
        if exclude_zero_axis:
            # Exclude the central row (fy=0 axis)
            upper_rows = slice(centre_index + 1, centre_index + half_slices + 1)
            lower_rows = slice(centre_index - half_slices, centre_index)
            rows_to_average = np.concatenate(
                [self.nps[lower_rows, :], self.nps[upper_rows, :]]
            )
        else:
            # Include all rows in the range
            rows_to_average = self.nps[
                centre_index - half_slices : centre_index + half_slices, :
            ]

        # Calculate averaged horizontal NPS
        horizontal_nps = rows_to_average.mean(axis=0)

        # Convert to one-sided spectrum
        mid_point = len(self.fx) // 2
        positive_freqs = self.fx[mid_point:]
        # Average the positive and negative frequency components
        one_sided_nps = (
            horizontal_nps[mid_point:] + horizontal_nps[mid_point - 1 :: -1]
        ) / 2

        return NPS1D(one_sided_nps, positive_freqs, self.num_rois, self.roi_dimensions)

    def get_vertical(
        self, num_slices: int = 15, exclude_zero_axis: bool = False
    ) -> NPS1D:
        """
        Returns one-sided 1D vertical NPS. Averages over a stack in the centre of 2D NPS.

        Parameters:
            num_slices (int): Number of columns to average together.
            exclude_zero_axis (bool): If True, excludes the fx=0 axis when averaging.

        Returns:
            NPS1D: One-sided 1D vertical noise power spectrum.
        """
        num_rows, num_cols = self.nps.shape
        centre_index = int(num_cols / 2)
        half_slices = int(num_slices / 2)

        # Define column indices to include
        if exclude_zero_axis:
            # Exclude the central column (fx=0 axis)
            right_cols = slice(centre_index + 1, centre_index + half_slices + 1)
            left_cols = slice(centre_index - half_slices, centre_index)
            cols_to_average = np.concatenate(
                [self.nps[:, left_cols], self.nps[:, right_cols]], axis=1
            )
        else:
            # Include all columns in the range
            cols_to_average = self.nps[
                :, centre_index - half_slices : centre_index + half_slices
            ]

        # Calculate averaged vertical NPS
        vertical_nps = cols_to_average.mean(axis=1)

        # Convert to one-sided spectrum
        mid_point = len(self.fy) // 2
        positive_freqs = self.fy[mid_point:]
        # Average the positive and negative frequency components
        one_sided_nps = (
            vertical_nps[mid_point:] + vertical_nps[mid_point - 1 :: -1]
        ) / 2

        return NPS1D(one_sided_nps, positive_freqs, self.num_rois, self.roi_dimensions)

    def get_radial_frequency_grid(self) -> np.ndarray:
        V, U = np.meshgrid(self.fy, self.fx)
        return np.sqrt(U**2 + V**2)

    @property
    def shape(self) -> tuple[int]:
        return self.nps.shape


def get_subroi_bounds(roi: np.ndarray, subroi_dim: int = 128) -> list[ROIBounds]:
    """Defines the bounds for each sub-ROI for NPS calculation."""
    numrows, numcols = roi.shape
    subroi_shape = (subroi_dim, subroi_dim)
    row_step, col_step = int(subroi_shape[0] / 2), int(subroi_shape[1] / 2)
    roibounds_list = []
    num_row_steps = (numrows - row_step) // row_step
    num_col_steps = (numcols - col_step) // col_step

    for i in range(num_row_steps):
        for j in range(num_col_steps):
            row_start = i * row_step
            row_end = row_start + subroi_shape[0]
            col_start = j * col_step
            col_end = col_start + subroi_shape[1]
            roibounds_list.append(ROIBounds(row_start, row_end, col_start, col_end))
    return roibounds_list


def subrois_from_rois(rois: list[np.ndarray], subroi_dim: int) -> np.ndarray:
    all_subrois = []
    for roi in rois:
        all_subrois.append(get_nps_subrois(roi, subroi_dim))
    return np.concatenate(all_subrois)


def get_nps_subrois(roi: np.ndarray, subroi_dim: int = 128) -> np.ndarray:
    """
    Splits square ROI into half overlapping sub-ROIs for NPS calculation.
    Returns N x (X x Y) array, where N is the number of sub-ROIs.
    """
    if (subroi_dim > np.array(roi.shape)).any():
        raise ValueError("Subroi must have dimensions smaller than ROI.")

    subroi_bounds_list = get_subroi_bounds(roi, subroi_dim)
    subroi_stack = []
    for roi_bounds in subroi_bounds_list:
        subroi = get_roi(roi, roi_bounds)
        subroi_stack.append(subroi)
    return np.array(subroi_stack, dtype=roi.dtype)


def nps2d_from_subrois(
    subrois: np.ndarray,
    pixel_dim_mm: tuple[float],
    detrend_method: str = "mean",
    pad_size: int = 0,
) -> NPS2D:
    """
    Calculate 2D noise power spectrum given a 3D array.
    """
    global_mean = subrois.mean()
    M, ny, nx = subrois.shape
    nps_stack = []
    for i in range(M):
        subroi = subrois[i]
        # Normalise pixel values in  subrois
        subroi = (global_mean / subroi.mean()) * subroi
        subroi_detrend = detrend(subroi, detrend_method)
        subroi_padded = np.pad(subroi_detrend, pad_size)
        F = np.fft.fft2(subroi_padded)
        F = np.fft.fftshift(np.abs(F) ** 2)
        nps_stack.append(F)

    ny_new, nx_new = subroi_padded.shape
    nps_stack = np.array(nps_stack)
    NPS = (pixel_dim_mm[0] * pixel_dim_mm[1]) / (nx * ny) * np.mean(nps_stack, axis=0)
    fx = np.fft.fftshift(np.fft.fftfreq(nx_new, pixel_dim_mm[1]))
    fy = np.fft.fftshift(np.fft.fftfreq(ny_new, pixel_dim_mm[0]))
    return NPS2D(
        NPS,
        fx,
        fy,
        num_rois=M,
        roi_dimensions=(nx, ny),
        pixel_value_mean=global_mean,
        pixel_value_var=subrois.var(),
        pad_size=pad_size,
    )


def calculate_nps2d_multiroi(
    rois: list[np.ndarray],
    pixel_dim_mm: tuple[float],
    subroi_dim: int = 64,
    detrend_method: str = "poly",
    pad_size: int = 0,
) -> NPS2D:
    """
    Calculate 2D noise power spectrum using multiple ROIs.
    ROIs can be from different slices or different parts of a slice.
    """
    subrois = subrois_from_rois(rois, subroi_dim)
    return nps2d_from_subrois(subrois, pixel_dim_mm, detrend_method, pad_size)


def calculate_nps2d(
    roi: np.ndarray,
    pixel_dim_mm: tuple[float],
    subroi_dim: int = 128,
    detrend_method: str = "mean",
    pad_size: int = 0,
) -> NPS2D:
    """
    Calculate 2D noise power spectrum for a single region of interest.
    """
    subroi_stack = get_nps_subrois(roi, subroi_dim)

    return nps2d_from_subrois(subroi_stack, pixel_dim_mm, detrend_method, pad_size)
