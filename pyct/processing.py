from skimage.transform import rescale
from skimage.feature import match_template
import numpy as np


def rescale_pixels(
    image: np.ndarray, new_max: float = 255, new_min: float = 0
) -> np.ndarray:
    """
    Rescale pixel values in image.
    """
    old_max, old_min = image.max(), image.min()
    new_image = (image - old_min) * (
        (new_max - new_min) / (old_max - old_min)
    ) + new_min
    return new_image


def detrend(image: np.ndarray, detrend_method: str, mask=None) -> np.ndarray:
    """
    Detrends the input image using specified method.

    Parameters:
        image (np.ndarray): Input 2D image to be detrended.
        detrend_method (str): Method for detrending. Options are 'mean' or 'poly'.
        'mean' subtracts the mean, 'poly' substracts a second order polynomial fit.
        mask (np.ndarray, optional): Boolean or binary mask defining which pixels
                                   to use for detrending calculation. If None,
                                   uses all pixels. Must have same shape as image.

    Returns:
        np.ndarray: Detrended image.

    Raises:
        ValueError: If detrend_method is not 'mean' or 'poly', or if mask has
                   wrong shape or dtype.
    """
    if mask is None:
        mask = np.ones(image.shape, dtype=bool)
    else:
        if mask.shape != image.shape:
            raise ValueError(
                f"Mask shape {mask.shape} must match image shape {image.shape}"
            )

        if mask.dtype != bool:
            if np.issubdtype(mask.dtype, np.integer):
                mask = mask.astype(bool)
            else:
                raise ValueError("Mask must be boolean or integer type")

        if not np.any(mask):
            raise ValueError("Mask must contain at least one True value")

    if detrend_method == "mean":
        masked_mean = image[mask].mean()
        return image - masked_mean
    elif detrend_method == "poly":  # 2D polynomial detrend.
        ny, nx = image.shape
        y, x = np.arange(ny) - ny / 2, np.arange(nx) - nx / 2
        Y, X = np.meshgrid(y, x, copy=False, indexing="ij")
        X_masked = X[mask]
        Y_masked = Y[mask]
        b_masked = image[mask]
        A = np.column_stack(
            [
                np.ones_like(X_masked),  # constant term
                X_masked,  # x term
                Y_masked,  # y term
                X_masked * Y_masked,  # xy term
                X_masked**2,  # x^2 term
                Y_masked**2,  # y^2 term
            ]
        )
        coefs, _, _, _ = np.linalg.lstsq(A, b_masked, rcond=None)
        Y_flat, X_flat = Y.flatten(), X.flatten()
        A_full = np.column_stack(
            [
                np.ones_like(X_flat),
                X_flat,
                Y_flat,
                X_flat * Y_flat,
                X_flat**2,
                Y_flat**2,
            ]
        )

        trend_surface = np.dot(A_full, coefs).reshape(ny, nx)
        return image - trend_surface
    else:
        raise ValueError("Detrend method must either be 'mean' or 'poly'.")


def cartesian2polar(
    xarr: np.ndarray, yarr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts Cartesian coordinates to polar coordinates.

    Parameters:
        xarr (np.ndarray): Array of x coordinates. Can be 2D array output of meshgrid.
        yarr (np.ndarray): Array of y coordinates. Can be 2D array output of meshgrid.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing r (radius) and theta (angle) arrays.
    """
    r = np.sqrt(xarr**2 + yarr**2)
    theta = np.arctan2(yarr, xarr)
    return r, theta


def rebin_by_pitch(
    xarr: np.ndarray,
    yarr: np.ndarray,
    pitch: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rebin x and y arrays to a specified x-axis pitch (spacing).

    Parameters:
        xarr (np.ndarray): Original x-axis values
        yarr (np.ndarray): Original y-axis values
        pitch (float): Desired spacing between x values
                      Must be larger than original x spacing.

    Returns:
        tuple[np.ndarray, np.ndarray]: (xnew, ynew) Arrays with new binning

    Raises:
        ValueError: If pitch is smaller than original spacing
    """
    orig_spacing = np.mean(np.diff(xarr))
    if pitch < orig_spacing:
        raise ValueError(
            f"Requested pitch ({pitch}) must be larger than "
            f"original spacing ({orig_spacing})"
        )

    # Create bin edges at exact multiples of pitch
    xmin, xmax = xarr.min(), xarr.max()
    bin_edges = np.arange(xmin, xmax + pitch, pitch)

    # Calculate bin centers (these will be our new x points)
    xnew = bin_edges[:-1]

    # Digitize original x values into bins
    bin_indices = np.digitize(xarr, bin_edges)

    # Calculate mean y for each bin
    ynew = np.array(
        [
            np.mean(yarr[bin_indices == i]) if np.any(bin_indices == i) else np.nan
            for i in range(1, len(bin_edges))
        ]
    )

    # Remove any NaN values from gaps
    valid = ~np.isnan(ynew)
    xnew = xnew[valid]
    ynew = ynew[valid]

    return xnew, ynew


def rebin(
    xarr: np.ndarray, yarr: np.ndarray, num_bins: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rebins the y values corresponding to the x values into a new set of x values
    by averaging.

    Parameters:
        xarr (numpy.array): numpy array of original x values.
        yarr (numpy.array): numpy array of original y values.
        num_bins (int): Number of bins to rebin the data.

    Returns:
        xnew (numpy.narray): New set of x values.
        ynew (numpy.narray): Corresponding y values rebinned by averaging.
    """
    if num_bins > len(xarr):
        raise ValueError("Number of bins should be less than or equal to array length.")
    elif num_bins < 2:
        raise ValueError("Number of bins should be greater than 1.")
    # Combine x and y values into a single array
    combined_values = np.column_stack((xarr, yarr))
    # Sort the combined values based on x values
    combined_values = combined_values[combined_values[:, 0].argsort()]

    xnew = np.linspace(xarr.min(), xarr.max(), num_bins)

    # Calculate the width of each bin
    bin_width = np.diff(xnew).mean()

    # Initialize lists to store new x and y values
    ynew = []

    # Iterate through the number of new x values
    for i in range(num_bins):
        # Calculate the start and end points of the current bin
        bin_start = xnew[i] - bin_width / 2
        bin_end = xnew[i] + bin_width / 2

        # Filter the combined values within the current bin
        bin_values = combined_values[
            (combined_values[:, 0] >= bin_start) & (combined_values[:, 0] < bin_end)
        ]

        # Calculate the average y value for the current bin
        avg_y = np.mean(bin_values[:, 1])

        # Add the average y value to the new y values list
        ynew.append(avg_y)

    return xnew, np.array(ynew)


def smooth(array1d: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Implementation of MATLAB smooth function. Applies rolling average window.

    Parameters:
        array1d (numpy.array): One-dimensional numpy array to be smoothed.
        window_size (int): Size of averaging window. Must be odd integer.

    Returns:
        smoothed (numpy.array): Smoothed one-dimensional array.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")
    smoothed_array = (
        np.convolve(array1d, np.ones(window_size, dtype=int), "valid") / window_size
    )
    norm_factor = np.arange(1, window_size, 2)
    edge_case_left = np.cumsum(array1d[: window_size - 1])[::2] / norm_factor
    edge_case_right = (np.cumsum(array1d[:-window_size:-1])[::2] / norm_factor)[::-1]
    return np.concatenate((edge_case_left, smoothed_array, edge_case_right))


def pixelate(float_val: float) -> int:
    return np.round(float_val).astype(int)


def circle_template(
    circle_radius_px: float, template_padding_px: int = 10
) -> np.ndarray:
    """Returns a circle matching template."""
    # Add a small border around the circle template
    template_size = pixelate(2 * circle_radius_px + template_padding_px)
    # template = np.zeros((template_size, template_size))
    x, y = np.arange(template_size), np.arange(template_size)
    X, Y = np.meshgrid(x, y)
    X, Y = X + 0.5, Y + 0.5
    distances = np.sqrt((X - template_size / 2) ** 2 + (Y - template_size / 2) ** 2)
    # template = distances < circle_radius_px
    return (distances < circle_radius_px).astype(int)


def circle_centre_subpixel(
    roi: np.ndarray,
    circle_radius_px: float,
    template_padding_px: int = 10,
    supersample_factor: int = 10,
) -> tuple[float, float]:
    """Find subpixel centre location of a circle within the ROI."""
    template = circle_template(
        supersample_factor * circle_radius_px, template_padding_px
    )
    roi_upscaled = rescale(roi, supersample_factor)
    # roi_upscaled = rescale_pixels(roi_upscaled).astype(np.uint8)
    matched = match_template(roi_upscaled, template, pad_input=True)
    matched = np.abs(matched)
    circle_centre = np.where(matched == matched.max())
    row_centre = (circle_centre[0][0] - 0.5) / supersample_factor
    column_centre = (circle_centre[1][0] - 0.5) / supersample_factor
    return row_centre, column_centre


def window_level(image: np.ndarray, window: int, level: int) -> np.ndarray:
    image_wl = image.copy()
    window_start = level - window / 2
    window_end = level + window / 2
    left_mask = image < window_start
    right_mask = image > window_end
    image_wl[left_mask] = window_start
    image_wl[right_mask] = window_end
    return rescale_pixels(image_wl)
