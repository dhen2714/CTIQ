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


def detrend(image: np.ndarray, detrend_method: str) -> np.ndarray:
    """
    Detrends the input image using specified method.

    Parameters:
        image (np.ndarray): Input 2D image to be detrended.
        detrend_method (str): Method for detrending. Options are 'mean' or 'poly'.

    Returns:
        np.ndarray: Detrended image.
    """
    if detrend_method == "mean":
        return image - image.mean()
    elif detrend_method == "poly":  # 2D polynomial detrend.
        ny, nx = image.shape
        y, x = np.arange(ny) - ny / 2, np.arange(nx) - nx / 2
        Y, X = np.meshgrid(y, x, copy=False)
        Y, X = Y.flatten(), X.flatten()
        A = np.array([X**0, X, Y, X * Y, X**2, Y**2]).T
        b = image.flatten()
        coefs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return image - np.dot(A, coefs).reshape(ny, nx)
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


def rebin(
    xarr: np.array, yarr: np.array, num_bins: int = 20
) -> tuple[np.array, np.array]:
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


def smooth(array1d: np.array, window_size: int = 5) -> np.array:
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
