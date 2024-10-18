from .processing import rescale_pixels
import cv2
import pydicom
import numpy as np
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass


def measure_width(thresholded_image: np.ndarray) -> int:
    """
    Measures maximum width of thresholded region.

       Parameters:
    -----------
    thresholded_image : np.ndarray
        Thresholded image, background has pixel value 0.

    Returns:
    --------
    int
        Measured width in pixels.
    """
    image_profile = np.sum(thresholded_image, axis=0)
    non_zero = np.nonzero(image_profile)[0]
    lat_left, lat_right = non_zero[0], non_zero[-1]
    width_px = lat_right - lat_left + 1
    return width_px


def patient_width_from_scout(
    dcm: pydicom.FileDataset, background_threshold: int = 30
) -> float:
    """
    NB: MEASUREMENTS USING THIS FUNCTION ARE CRUDE.

    Calculation of patient width from a CT scout image.

    This function processes a DICOM CT scout image to measure the patient's width.
    It uses adaptive thresholding to segment the patient from the background,
    then measures the width of the segmented region.

    Parameters:
    -----------
    dcm : pydicom.FileDataset
        A pydicom FileDataset object containing the CT scout image and its metadata.

    background_threshold : int, optional (default=30)
        The initial threshold used to separate background from patient.
        Pixels with values below this are considered background.

    Returns:
    --------
    float
        The measured patient width in millimeters.

    Process:
    --------
    1. Rescales the pixel array of the input image.
    2. Creates an initial mask to separate background from patient.
    3. Calculates the mean pixel value of the patient region.
    4. Creates a refined threshold at 50% of the patient mean.
    5. Applies this threshold to create a binary image of the patient.
    6. Measures the width of the patient region in pixels.
    7. Converts the width from pixels to real-world units using DICOM metadata.

    Example:
    --------
    >> import pydicom
    >> dcm = pydicom.dcmread("path/to/scout_image.dcm")
    >> width = patient_width_from_scout(dcm)
    >> print(f"Patient width: {width:.2f} mm")
    """
    img = dcm.pixel_array
    img[img < 0] = 0  # Background border can have values of -2048
    img = rescale_pixels(img)
    # pt_orientation = dcm[(0x0020, 0x0020)].value
    pt_orientation = dcm.get((0x0020, 0x0020), None)
    if pt_orientation:
        lr_orientation = pt_orientation.value[0]
        if lr_orientation in ("H", "F"):
            img = img.T
            spacing, _ = np.array(dcm[(0x0028, 0x0030)].value).astype(float)
        else:
            _, spacing = np.array(dcm[(0x0028, 0x0030)].value).astype(float)
    else:
        _, spacing = np.array(dcm[(0x0028, 0x0030)].value).astype(float)
    background_masked = img > background_threshold
    patient_mean = img[background_masked].mean()
    mask_threshold = 0.5 * patient_mean
    thresholded_image = (img > mask_threshold).astype(int)
    width_px = measure_width(thresholded_image)
    return width_px * spacing


def find_image_histogram_peak(
    image: np.ndarray,
    num_bins: int = 2000,
    smoothing_sigma: float = 2,
) -> float:
    # Compute histogram of standard deviations
    hist, bin_edges = np.histogram(image, bins=num_bins)

    # Smooth the histogram
    smoothed_hist = gaussian_filter1d(hist, sigma=smoothing_sigma)

    # Get the bin center corresponding to the highest peak
    peak_bin_index = np.where(smoothed_hist == smoothed_hist.max())[0][0]
    peak_value = (bin_edges[peak_bin_index] + bin_edges[peak_bin_index + 1]) / 2

    if peak_value < 0:
        peak_value = 0
    return peak_value


@dataclass
class SliceNoiseResult:
    """Result container for single slice noise index calculation."""

    noise_index: float
    stddev_distribution: np.ndarray
    stddev_image: np.ndarray


@dataclass
class GNIResult:
    """Result and metadata container for global noise index calculation."""

    result: float  # The overall global noise index, from all slices.
    kernel_size: int  # Size of kernel used to calculate local standard deviation.
    stddev_images: np.ndarray  # Standard deviation images for all slices.
    all_indices: np.ndarray  # Calculated noise indices for each slice.
    all_stddev_distributions: (
        np.ndarray
    )  # Standard deviation distributions for all slices

    @property
    def result_dropna(self):
        return self.all_indices[~np.isnan(self.all_indices)].mean()


def calculate_slice_noise_index(
    slice_img: np.ndarray, kernel_size: int = 30
) -> SliceNoiseResult:
    """
    Calculate the Noise Index for a single CT slice and return the standard deviation distribution and image.

    Parameters:
    -----------
    slice_img : np.ndarray
        A 2D numpy array representing a single CT slice.
    kernel_size : int, optional (default=30)
        The size of the kernel used for local variance calculation.

    Returns:
    --------
    SliceNoiseResult
        A named tuple containing:
        - noise_index : float
            The Noise Index for the given slice.
        - stddev_distribution : np.ndarray
            The distribution of standard deviations in the soft tissue region.
        - stddev_image : np.ndarray
            The image of local standard deviations.
    """
    float_arr = slice_img.astype(np.float32)
    average_kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (
        kernel_size * kernel_size
    )

    local_average_image = cv2.filter2D(float_arr, -1, average_kernel)
    local_average_squared_image = cv2.filter2D(float_arr**2, -1, average_kernel)

    variance_image = local_average_squared_image - local_average_image**2
    stddev_image = np.sqrt(variance_image)

    soft_tissue_mask = (float_arr >= -300) & (float_arr <= 300)
    stddevs = stddev_image[soft_tissue_mask]
    stddevs = stddevs[~(stddevs == 0)]  # Ignore textureless parts of image.
    stddevs = stddevs[~np.isnan(stddevs)]

    if stddevs.size == 0:
        return SliceNoiseResult(np.nan, np.array([]), stddev_image)
    else:
        noise_index = find_image_histogram_peak(stddevs)
        return SliceNoiseResult(noise_index, stddevs, stddev_image)


def calculate_global_noise_index(
    ctseries: np.ndarray, kernel_size: int = 30
) -> GNIResult:
    """
    Calculate the Global Noise Index (GNI) for a series of CT images.

    This function computes the GNI for each slice in a CT series and returns the overall GNI
    along with additional information. The GNI is calculated using a local variance approach
    and focuses on the soft tissue region of the CT images.

    Parameters:
    -----------
    ctseries : np.ndarray
        A 3D numpy array representing the CT series. The first dimension is assumed to be
        the slice index, and the following two dimensions represent each 2D CT image.

    kernel_size : int, optional (default=30)
        The size of the kernel used for local variance calculation. This determines the
        local neighborhood for noise estimation.

    Returns:
    --------
    GNIResult
        A named tuple containing:
        - result : float
            The mean GNI across all slices.
        - kernel_size : int
            The kernel size used for calculation.
        - stddev_images : np.ndarray
            3D array of local standard deviation images for each slice.
        - all_indices : np.ndarray
            Array of GNI values for each individual slice.
        - all_stddev_distributions : np.ndarray
            Array of standard deviation distributions for each slice.

    Process:
    --------
    1. For each slice in the CT series:
        a. Calculate the noise index, standard deviation distribution, and standard deviation image
           using calculate_slice_noise_index function.
    2. Calculate the overall GNI as the mean of individual slice GNIs.

    Notes:
    ------
    - The function assumes CT values are in Hounsfield Units (HU).
    - Soft tissue is defined as the region between -300 and 300 HU.
    - The GNI for each slice is determined as the mode of local standard deviations.
    """
    num_slices = len(ctseries)
    gnis = np.zeros(num_slices)
    stddev_images = np.zeros(ctseries.shape, dtype=np.float32)
    all_stddev_distributions = []

    for i, slice_img in enumerate(ctseries):
        slice_result = calculate_slice_noise_index(slice_img, kernel_size)
        gnis[i] = slice_result.noise_index
        all_stddev_distributions.append(slice_result.stddev_distribution)
        stddev_images[i] = slice_result.stddev_image

    overall_gni = gnis.mean()
    return GNIResult(
        overall_gni, kernel_size, stddev_images, gnis, all_stddev_distributions
    )
