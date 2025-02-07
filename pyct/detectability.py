from .ttf import TTF
from .nps import NPS2D
import numpy as np
from scipy.special import j1


def circular_task_function(
    contrast: float, radius_mm: float, fr: np.ndarray
) -> np.ndarray:
    """
    Generate frequency domain representation of a circular object of known contrast.

    Parameters
    ----------
    contrast : float, optional
        Contrast value of the circle (default=10)
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


def dprime_npw(
    df: float, task_function: np.ndarray, ttf2d: np.ndarray, nps2d: np.ndarray
) -> float:
    """
    Calculate the non-prewhitening (NPW) observer detectability index (d') for a given imaging task.

    The NPW observer model assumes the observer has knowledge of the signal (task function)
    and system transfer properties (TTF) but does not prewhiten the noise (NPS).

    Task function, ttf2d and nps2d must be sampled on the same spatial frequency grid.

    Parameters
    ----------
    df : float
        Frequency sampling interval in cycles/mm
    task_function : np.ndarray
        2D array containing the frequency domain representation of the detection task
    ttf2d : np.ndarray
        2D array containing the task transfer function (specific MTF for the task)
    nps2d : np.ndarray
        2D array containing the noise power spectrum

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
    d_numerator = np.sum((task_function**2 * ttf2d**2) * df * df) ** 2
    d_denominator = np.sum((task_function**2 * ttf2d**2 * nps2d) * df * df)
    d = np.sqrt(d_numerator / d_denominator)
    return d


def calculate_dprime_npw(
    contrast: float, task_radius: float, ttf_result: TTF, nps_result: NPS2D
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

    Returns
    -------
    float
        Detectability index d' for the NPW observer model
    """
    # Get 2D MTF
    fr = nps_result.get_radial_frequency_grid()
    ttf2d = np.interp(fr, ttf_result.f, ttf_result.mtf)

    W = circular_task_function(contrast, task_radius, fr)

    nps2d = nps_result.nps
    df = np.abs(nps_result.fx[1] - nps_result.fx[0])

    return dprime_npw(df, W, ttf2d, nps2d)
