from .ttf import TTF
from .nps import NPS1D
import numpy as np
from scipy.special import j1
from dataclasses import dataclass


def circular_task_function(
    contrast: float, radius_mm: float, fr: np.ndarray
) -> np.ndarray:
    """
    Generate frequency domain representation of a circular object of known contrast.

    Parameters
    ----------
    contrast : float
        Contrast value of the circle
    radius_mm : float
        Radius of the circle in mm
    fr : np.ndarray
        2D array of radial frequencies in cycles/mm


    Returns
    -------
    np.ndarray
        2D array containing the frequency domain representation of the circle

    Notes
    -----
    The Fourier transform of a circle of radius R is:
    F(rho) = πR² * [2*J₁(2*pi*rho*R)]/(2*pi*rho*R)
    where rho = sqrt(fx² + fy²) is the radial frequency
    and J₁ is the first-order Bessel function of the first kind
    """
    # Handle the special case at fr = 0 to avoid division by zero
    task = np.zeros_like(fr)
    nonzero_mask = fr > 0

    # Calculate 2πρR term
    two_pi_rho_R = 2 * np.pi * fr[nonzero_mask] * radius_mm

    # Calculate the frequency domain representation
    # The factor of contrast * π * radius_mm² comes from the circle's area and contrast
    task[nonzero_mask] = (
        contrast * np.pi * radius_mm**2 * 2 * j1(two_pi_rho_R) / two_pi_rho_R
    )

    # At fr = 0, the limit of the function is contrast * π * radius_mm²
    task[~nonzero_mask] = contrast * np.pi * radius_mm**2

    return task


def frequency_task_function_fft(task_function: np.ndarray) -> np.ndarray:
    """
    Convert frequency domain task function to spatial domain using inverse FFT.
    Useful for visualising task function in spatial domain.

    Parameters
    ----------
    task_function : np.ndarray
        2D array of frequency domain task function

    Returns
    -------
    np.ndarray
        2D array spatial domain representation
    """
    W_centred = np.fft.fftshift(task_function)
    Wx = np.fft.ifft2(W_centred)
    Wx = np.fft.fftshift(Wx)
    # Factor 1/2pi accounts for continuous vs discrete FT
    Wx_real = np.real(Wx) * (2 * np.pi)
    return Wx_real


def hvrf(fr: np.ndarray, fov: float, display: float, distance: float):
    a1 = 1.5
    a2 = 0.98
    a3 = 0.68
    rho = (fr * fov * distance * np.pi) / (display * 180)
    return (rho**a1) * np.exp(-a2 * (rho**a3))


def dprime(
    fr: np.ndarray,
    df: float,
    task_function: np.ndarray,
    ttf2d: np.ndarray,
    nps2d: np.ndarray,
    eye_filter: np.ndarray | None = None,
    internal_noise: np.ndarray | None = None,
) -> float:
    if eye_filter is None:
        eye_filter = np.ones(fr.shape)
    if internal_noise is None:
        internal_noise = np.zeros(fr.shape)

    d_numerator = np.sum((task_function**2 * ttf2d**2 * eye_filter**2) * df * df) ** 2
    d_denominator = np.sum(
        (task_function**2 * ttf2d**2 * nps2d * eye_filter**4 + internal_noise) * df * df
    )
    d = np.sqrt(d_numerator / d_denominator)
    return d


@dataclass
class ResampledData:
    df: float
    fr: np.ndarray
    ttf2d: np.ndarray
    nps2d: np.ndarray


def get_resampled_data(
    ttf_result: TTF, nps_result: NPS1D, num_pixels: int, pix_size_mm: float
) -> ResampledData:
    fx = np.fft.fftshift(np.fft.fftfreq(num_pixels, pix_size_mm))
    fx_grid, fy_grid = np.meshgrid(fx, fx)
    fr_grid = np.sqrt(fx_grid**2 + fy_grid**2)
    df = fx[1] - fx[0]

    ttf2d_new = np.interp(fr_grid, ttf_result.f, ttf_result.mtf)
    nps2d_new = np.interp(fr_grid, nps_result.f, nps_result.nps)
    return ResampledData(df, fr_grid, ttf2d_new, nps2d_new)


def calculate_dprime_npwei(
    contrast: float,
    task_radius: float,
    ttf_result: TTF,
    nps_result: NPS1D,
    fov_size_mm: float = 20,
    pix_size_mm: float = 0.1,
    display_zoom: float = 3,
    display_pixel_pitch_mm: float = 0.2,
    viewing_distance_mm: float = 500,
    alpha: float = 5,
) -> float:
    num_pixels = int(fov_size_mm / pix_size_mm)
    res = get_resampled_data(ttf_result, nps_result, num_pixels, pix_size_mm)

    W = circular_task_function(contrast, task_radius, res.fr)

    num_display_pixels = display_zoom * num_pixels
    display_size_mm = num_display_pixels * display_pixel_pitch_mm
    variance = nps_result.variance

    E = hvrf(res.fr, fov_size_mm, display_size_mm, viewing_distance_mm)
    N = alpha * ((viewing_distance_mm / 1000) ** 2) * variance * np.ones(res.fr.shape)

    return dprime(
        res.fr,
        res.df,
        W,
        res.ttf2d,
        res.nps2d,
        eye_filter=E,
        internal_noise=N,
    )


def calculate_dprime_npw(
    contrast: float,
    task_radius: float,
    ttf_result: TTF,
    nps_result: NPS1D,
    fov_size_mm: float = 20,
    pix_size_mm: float = 0.1,
) -> float:
    """
    Calculate the NPW observer detectability index for a circular detection task.

    This function handles the setup of 2D arrays and interpolation needed to
    calculate d' for a circular task of given contrast and radius.

    Parameters
    ----------
    contrast : float
        Contrast of the circular task relative to background
    task_radius : float
        Radius of the circular task in mm
    ttf_result : TTF
        Task transfer function result object containing frequency and MTF data
    nps_result : NPS2D
        2D noise power spectrum result object containing frequency and NPS data
    fov_size_mm : float, optional
        Field of view in which task object sits, size in mm.
    pixel_size_mm : float, optional
        Pixel size in mm.

    Returns
    -------
    float
        Detectability index d' for the NPW observer model

    Notes
    -----
    The NPW d' is calculated as:
    d' = sqrt((∫∫|W(f)·TTF(f)|²df)² / ∫∫|W(f)·TTF(f)|²·NPS(f)df)
    where W(f) is the task function, TTF(f) is the task transfer function,
    and NPS(f) is the noise power spectrum.
    """
    num_pixels = int(fov_size_mm / pix_size_mm)
    res = get_resampled_data(ttf_result, nps_result, num_pixels, pix_size_mm)

    W = circular_task_function(contrast, task_radius, res.fr)

    return dprime(res.fr, res.df, W, res.ttf2d, res.nps2d)
