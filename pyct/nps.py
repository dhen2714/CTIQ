from .roi_tools import ROIBounds, get_roi
from .processing import cartesian2polar, detrend, rebin, smooth
import numpy as np
from dataclasses import dataclass


@dataclass
class NPS1D:
    """Container class for 1D noise power spectrum."""

    nps: np.ndarray
    f: np.ndarray

    def rebin(self, n):
        fnew, npsnew = rebin(self.f, self.nps, num_bins=n)
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

    def get_radial(self) -> NPS1D:
        """Returns radially averaged 1D NPS."""
        FX, FY = np.meshgrid(self.fx, self.fy)
        FR, _ = cartesian2polar(FX, FY)
        max_freq = np.min([FX.max(), FY.max()])
        nps_inds = np.where(FR <= max_freq)
        num_radial_samples = int(len(self.fx) / 2)
        fr, npsr = FR[nps_inds].flatten(), self.nps[nps_inds].flatten()
        fr, npsr = rebin(fr, npsr, num_bins=num_radial_samples)
        return NPS1D(npsr, fr)

    def get_horizontal(self, num_slices: int = 15) -> NPS1D:
        """
        Returns 1D horizontal NPS. Averages over a stack in the centre of 2D NPS.

        Parameters:
            num_slices (int): Number of rows to average together.

        Returns:
            NPS1D: 1D horizontal noise power spectrum.
        """
        num_rows, _ = self.nps.shape
        centre_index = int(num_rows / 2)  # rounds down by default
        return NPS1D(
            self.nps[
                centre_index - int(num_slices / 2) : centre_index + int(num_slices / 2),
                :,
            ].mean(axis=0),
            self.fx,
        )

    def get_vertical(self, num_slices: int = 15) -> NPS1D:
        """
        Returns 1D horizontal NPS. Averages over a stack in the centre of 2D NPS.

        Parameters:
            num_slices (int): Number of columns to average together.

        Returns:
            NPS1D: 1D vertical noise power spectrum.
        """
        _, num_cols = self.nps.shape
        centre_index = int(num_cols / 2)
        return NPS1D(
            self.nps[
                :,
                centre_index - int(num_slices / 2) : centre_index + int(num_slices / 2),
            ].mean(axis=1),
            self.fy,
        )

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
    subrois: np.ndarray, pixel_dim_mm: tuple[float], detrend_method: str = "mean"
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
        F = np.fft.fft2(detrend(subroi, detrend_method))
        F = np.fft.fftshift(np.abs(F) ** 2)
        nps_stack.append(F)
    nps_stack = np.array(nps_stack)
    NPS = (pixel_dim_mm[0] * pixel_dim_mm[1]) / (nx * ny) * np.mean(nps_stack, axis=0)
    fx = np.fft.fftshift(np.fft.fftfreq(nx, pixel_dim_mm[1]))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, pixel_dim_mm[0]))
    return NPS2D(NPS, fx, fy)


def calculate_nps2d_multiroi(
    rois: list[np.ndarray],
    pixel_dim_mm: tuple[float],
    subroi_dim: int = 64,
    detrend_method: str = "poly",
) -> NPS2D:
    """
    Calculate 2D noise power spectrum using multiple ROIs.
    ROIs can be from different slices or different parts of a slice.
    """
    subrois = subrois_from_rois(rois, subroi_dim)
    return nps2d_from_subrois(subrois, pixel_dim_mm, detrend_method)


def calculate_nps2d(
    roi: np.ndarray,
    pixel_dim_mm: tuple[float],
    subroi_dim: int = 128,
    detrend_method: str = "mean",
) -> NPS2D:
    """
    Calculate 2D noise power spectrum for a single region of interest.
    """
    subroi_stack = get_nps_subrois(roi, subroi_dim)

    return nps2d_from_subrois(subroi_stack, pixel_dim_mm, detrend_method)
