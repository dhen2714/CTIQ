from .roi_tools import ROIBounds, get_roi
from .processing import cartesian2polar, detrend, rebin_by_pitch, rebin, smooth
import numpy as np
from dataclasses import dataclass


@dataclass
class NPS1D:
    """Container class for 1D noise power spectrum."""

    nps: np.ndarray
    f: np.ndarray
    variance: float = None
    num_rois: int = None  # Number of ROIs used to calculate NPS
    roi_dimensions: tuple[int] = None  # num_columns (nx), num_rows (ny)
    pixel_value_mean: float = None
    pixel_value_var: float = None
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

    def _get_nps_at_frequency(self, frequency: float, nnps: bool = False) -> float:
        if frequency < self.f.min() or frequency > self.f.max():
            raise ValueError(
                f"Frequency {frequency} is outside the range of available "
                f"frequencies [{self.f.min()}, {self.f.max()}]"
            )
        if nnps:
            return np.interp(frequency, self.f, self.nnps)
        else:
            return np.interp(frequency, self.f, self.nps)

    def nps_value(self, frequency: float) -> float:
        """
        Get the NPS value at specified frequency.
        """
        return self._get_nps_at_frequency(frequency, nnps=False)

    def nnps_value(self, frequency: float) -> float:
        return self._get_nps_at_frequency(frequency, nnps=True)

    @property
    def nnps(self):
        """NPS normalised by squared pixel value."""
        if self.pixel_value_mean is None:
            raise ValueError("Mean pixel value not provided.")
        return self.nps / (self.pixel_value_mean**2)

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

    @property
    def noise(self) -> float:
        return np.sqrt(self.variance)

    def __len__(self):
        return len(self.nps)


@dataclass
class NPS2D:
    """Container class for 2D noise power spectrum."""

    nps: np.ndarray
    fx: np.array  # dimension across different columns
    fy: np.array  # dimension across different rows
    variance: float = None
    num_rois: int = None  # Number of ROIs used to calculate NPS
    roi_dimensions: tuple[int] = None  # num_columns (nx), num_rows (ny)
    pixel_value_mean: float = None
    pixel_value_var: float = None
    pad_size: int = None  # Size of padding used on subrois to calculate NPS

    def get_radial(self) -> NPS1D:
        """Returns radially averaged 1D NPS."""
        FX, FY = np.meshgrid(self.fx, self.fy)
        FR, _ = cartesian2polar(FX, FY)
        pitch = self.fx[1] - self.fx[0]
        fr, npsr = FR.flatten(), self.nps.flatten()
        fr, npsr = rebin_by_pitch(fr, npsr, pitch=pitch)
        return NPS1D(
            npsr,
            fr,
            self.variance,
            self.num_rois,
            self.roi_dimensions,
            self.pixel_value_mean,
            self.pixel_value_var,
            self.pad_size,
        )

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

        return NPS1D(
            one_sided_nps,
            positive_freqs,
            self.variance,
            self.num_rois,
            self.roi_dimensions,
            self.pixel_value_mean,
            self.pixel_value_var,
            self.pad_size,
        )

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

        return NPS1D(
            one_sided_nps,
            positive_freqs,
            self.variance,
            self.num_rois,
            self.roi_dimensions,
            self.pixel_value_mean,
            self.pixel_value_var,
            self.pad_size,
        )

    def get_radial_frequency_grid(self) -> np.ndarray:
        V, U = np.meshgrid(self.fy, self.fx)
        return np.sqrt(U**2 + V**2)

    @property
    def shape(self) -> tuple[int]:
        return self.nps.shape

    @property
    def noise(self) -> float:
        return np.sqrt(self.variance)


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
    ny_new, nx_new = ny + 2 * pad_size, nx + 2 * pad_size
    norm = (pixel_dim_mm[0] * pixel_dim_mm[1]) / (nx * ny)

    fx = np.fft.fftshift(np.fft.fftfreq(nx_new, pixel_dim_mm[1]))
    fy = np.fft.fftshift(np.fft.fftfreq(ny_new, pixel_dim_mm[0]))
    dfx = fx[1] - fx[0]
    dfy = fy[1] - fy[0]

    nps_stack = []
    variances = []
    for i in range(M):
        subroi = subrois[i]
        # Normalise pixel values in  subrois
        subroi = (global_mean / subroi.mean()) * subroi
        subroi_detrend = detrend(subroi, detrend_method)
        subroi_padded = np.pad(subroi_detrend, pad_size)
        F = np.fft.fft2(subroi_padded)
        subroi_nps = norm * np.fft.fftshift(np.abs(F) ** 2)
        nps_stack.append(subroi_nps)
        variance = np.sum(subroi_nps) * dfx * dfy
        variances.append(variance)

    variance = np.mean(variances)
    nps_stack = np.array(nps_stack)
    NPS = np.mean(nps_stack, axis=0)

    return NPS2D(
        NPS,
        fx,
        fy,
        variance=variance,
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
