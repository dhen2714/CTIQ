import numpy as np
import pydicom
from dataclasses import dataclass
from pathlib import Path
import warnings

IMAGE_POS_PATIENT = (0x0020, 0x0032)


@dataclass
class SliceError:
    """Container for slice loading/validation errors"""

    file_path: str
    error: Exception
    message: str


class AxialSeries:
    """
    An iterable container for a series of axial CT DICOM slices.

    This class provides a convenient way to work with a directory of DICOM files
    containing a CT series. It reads the DICOM files, extracts relevant metadata,
    and stores the DICOM objects in a sortable, iterable container.

    By default, the class extracts the slice location, but you can also specify
    additional DICOM tags to extract and store in the class instance.

    Args:
        directory (str | Path): The path to the directory containing the DICOM files.
        load_images (bool, optional): If True, the full DICOM data, including pixel
            data, will be loaded. If False, only the metadata will be loaded.
            Defaults to True.
        custom_tags (dict[str, tuple[int, int]], optional): A dictionary mapping
            custom tag names to their corresponding DICOM tag locations (group, element).
            The extracted values will be stored in the `custom_tags` attribute.
            Defaults to None.
        skip_dimension_mismatch: If True, skip slices with mismatched dimensions
                                If False, raise ValueError on dimension mismatch
        validate_dimensions: If False, skips dimension validation entirely for
                            better performance. Use only when confident all
                            slices have matching dimensions.

    Attributes:
        slice_locations (list[float]): The sorted list of slice locations.
        file_paths (list[str]): The sorted list of file paths for the DICOM files.
        errors (list[tuple[str, Exception]]): A list of tuples containing the file
            path and the exception that occurred during DICOM file reading.
        custom_tags (dict[str, list[Any]]): A dictionary mapping custom tag names to
            their extracted values for each DICOM slice.
        pixel_size (np.ndarray): The pixel dimensions of the first DICOM slice.
    """

    def __init__(
        self,
        directory: str | Path,
        load_images: bool = True,
        custom_tags: dict[str, tuple[int, int]] = None,
        skip_dimension_mismatch: bool = True,
        validate_dimensions: bool = False,
    ) -> None:
        self._slices = []
        self.slice_locations = []
        self.file_paths = []
        self.errors = []
        self.dimension_errors = []
        self.custom_tags = {}
        self.expected_shape = None

        if custom_tags:
            for key in custom_tags.keys():
                self.custom_tags[key] = []

        if validate_dimensions and load_images:
            # Use the more thorough validation process
            self._load_with_validation(
                directory, load_images, skip_dimension_mismatch, custom_tags
            )
        else:
            # Fast path: skip dimension validation
            self._load_without_validation(directory, load_images, custom_tags)

    def _load_without_validation(
        self,
        directory: str | Path,
        load_images: bool,
        custom_tags: dict[str, tuple[int, int]] = None,
    ) -> None:
        """Fast loading path that skips dimension validation."""
        for dir_item in Path(directory).iterdir():
            try:
                dcm = pydicom.dcmread(dir_item, stop_before_pixels=(not load_images))

                # Set expected_shape from first slice with pixel data
                if (
                    load_images
                    and self.expected_shape is None
                    and hasattr(dcm, "pixel_array")
                ):
                    self.expected_shape = dcm.pixel_array.shape

                self._slices.append(dcm)
                self.slice_locations.append(float(dcm[IMAGE_POS_PATIENT].value[2]))
                self.file_paths.append(str(dir_item))

                if custom_tags:
                    for key, val in custom_tags.items():
                        tag = pydicom.tag.Tag(val[0], val[1])
                        tagval = dcm.get(tag, None).value
                        self.custom_tags[key].append(
                            str(tagval) if tagval is not None else None
                        )

            except Exception as e:
                self.errors.append((str(dir_item), e))

        if not self._slices:
            raise ValueError(f"No valid DICOM slices could be loaded from {directory}")

        # Sort slices by location
        sorted_indices = np.argsort(self.slice_locations).astype(int)
        self._slices = np.array(self._slices)[sorted_indices]
        self.slice_locations = np.array(self.slice_locations)[sorted_indices]
        self.file_paths = np.array(self.file_paths)[sorted_indices].astype(str)
        if custom_tags:
            for key, val in self.custom_tags.items():
                self.custom_tags[key] = np.array(val)[sorted_indices]

    def _load_with_validation(
        self,
        directory: str | Path,
        load_images: bool,
        skip_dimension_mismatch: bool,
        custom_tags: dict[str, tuple[int, int]] = None,
    ) -> None:
        dimension_counts = {}
        total_files = 0
        for dir_item in Path(directory).iterdir():
            try:
                dcm = pydicom.dcmread(dir_item, stop_before_pixels=(not load_images))
                if hasattr(dcm, "pixel_array"):
                    shape = tuple(dcm.pixel_array.shape)
                    if shape not in dimension_counts:
                        dimension_counts[shape] = []
                    dimension_counts[shape].append((dcm, dir_item))
                    total_files += 1
            except Exception as e:
                self.errors.append((str(dir_item), e))

        if not dimension_counts:
            raise ValueError(
                f"No valid DICOM files with pixel data found in {directory}"
            )

        # Determine the most common dimensions
        most_common_shape = max(dimension_counts.items(), key=lambda x: len(x[1]))[0]
        most_common_count = len(dimension_counts[most_common_shape])

        # Calculate percentage of slices with most common dimensions
        consistency_percentage = (most_common_count / total_files) * 100

        if consistency_percentage < 50:
            dimension_str = "\n".join(
                [
                    f"  {shape}: {len(slices)} slices"
                    for shape, slices in dimension_counts.items()
                ]
            )
            raise ValueError(
                f"No consistent slice dimensions found. Dimension distribution:\n{dimension_str}"
            )

        self.expected_shape = most_common_shape

        # Second pass: process all slices using the determined expected dimensions
        for shape, slice_info in dimension_counts.items():
            for dcm, dir_item in slice_info:
                if shape != self.expected_shape:
                    error = SliceError(
                        file_path=str(dir_item),
                        error=ValueError(
                            f"Dimension mismatch: expected {self.expected_shape}, got {shape}"
                        ),
                        message=f"Slice dimensions {shape} do not match expected dimensions {self.expected_shape}",
                    )
                    self.dimension_errors.append(error)
                    if not skip_dimension_mismatch:
                        raise error.error
                    continue

                try:
                    self._slices.append(dcm)
                    self.slice_locations.append(float(dcm[IMAGE_POS_PATIENT].value[2]))
                    self.file_paths.append(str(dir_item))

                    if custom_tags:
                        for key, val in custom_tags.items():
                            tag = pydicom.tag.Tag(val[0], val[1])
                            tagval = dcm.get(tag, None).value
                            self.custom_tags[key].append(
                                str(tagval) if tagval is not None else None
                            )

                except Exception as e:
                    error = SliceError(
                        file_path=str(dir_item),
                        error=e,
                        message=f"Error loading slice: {str(e)}",
                    )
                    self.errors.append((str(dir_item), e))

        if not self._slices:
            raise ValueError(f"No valid DICOM slices could be loaded from {directory}")

        if self.dimension_errors:
            warnings.warn(
                f"Skipped {len(self.dimension_errors)} slices due to dimension mismatch. "
                f"Using most common dimensions {self.expected_shape} "
                f"({consistency_percentage:.1f}% of slices). "
                f"Use .dimension_error_report() to see details."
            )

        # Sort slices by location
        sorted_indices = np.argsort(self.slice_locations).astype(int)
        self._slices = np.array(self._slices)[sorted_indices]
        self.slice_locations = np.array(self.slice_locations)[sorted_indices]
        self.file_paths = np.array(self.file_paths)[sorted_indices].astype(str)
        if custom_tags:
            for key, val in self.custom_tags.items():
                self.custom_tags[key] = np.array(val)[sorted_indices]

    def get_dimension_error_report(self) -> str:
        """
        Generate a detailed report of any dimension mismatches.

        Returns:
            str: A formatted report of dimension errors
        """
        if not self.dimension_errors:
            return "No dimension errors found."

        report = [f"Found {len(self.dimension_errors)} dimension errors:"]
        report.append(f"Expected dimensions: {self.expected_shape}")

        # Group errors by dimension
        dimension_groups = {}
        for error in self.dimension_errors:
            # Extract dimensions from error message
            import re

            dims = re.search(r"got \(([\d,\s]+)\)", error.message)
            if dims:
                dims_tuple = tuple(map(int, dims.group(1).split(",")))
                if dims_tuple not in dimension_groups:
                    dimension_groups[dims_tuple] = []
                dimension_groups[dims_tuple].append(error.file_path)

        report.append("\nError summary by dimensions:")
        for dims, files in dimension_groups.items():
            report.append(f"\nDimensions {dims}: {len(files)} files")
            for file_path in files:
                report.append(f"  - {file_path}")

        return "\n".join(report)

    @property
    def pixel_size(self):
        slice_pixel_dim = self[0][(0x0028, 0x0030)].value
        slice_pixel_dim = np.array(slice_pixel_dim).astype(float)
        return slice_pixel_dim

    @property
    def orientation(self):
        # (0018, 5100) Patient Position - HFP, HFS, FFP, FFS etc.
        return self[0].get((0x0018, 0x5100)).value

    def __len__(self):
        return len(self._slices)

    def __iter__(self):
        self._current_idx = 0
        return self

    def __getitem__(self, i):
        return self._slices[i]

    def __next__(self):
        if self._current_idx < len(self):
            current_index = self._current_idx
            self._current_idx += 1
            return self._slices[current_index]
        else:
            raise StopIteration

    def get_array(self, rescale: bool = True) -> np.ndarray:
        """
        Convert the DICOM pixel data into a 3D numpy array.

        Args:
            rescale (bool, optional): If True, applies the rescale slope and
                intercept to convert to proper CT numbers (Hounsfield Units).
                Defaults to True.

        Returns:
            np.ndarray: 3D array of pixel data, optionally rescaled to HU
        """
        array3d = []
        for dcm in self:
            # Get raw pixel array
            pixel_array = dcm.pixel_array

            if rescale:
                # Get rescale parameters
                rescale_slope = float(dcm.get((0x0028, 0x1053), 1.0).value)
                rescale_intercept = float(dcm.get((0x0028, 0x1052), 0.0).value)

                # Convert to float to avoid potential integer overflow
                pixel_array = pixel_array.astype(float)

                # Apply rescale formula: HU = pixel_value * slope + intercept
                pixel_array = pixel_array * rescale_slope + rescale_intercept

            array3d.append(pixel_array)

        return np.array(array3d)

    def get_slice_hu(self, index: int) -> np.ndarray:
        """
        Get a single slice converted to Hounsfield Units.

        Args:
            index (int): Index of the slice to retrieve

        Returns:
            np.ndarray: 2D array of pixel data in Hounsfield Units
        """
        dcm = self[index]
        pixel_array = dcm.pixel_array.astype(float)

        # Get rescale parameters
        rescale_slope = float(dcm.get((0x0028, 0x1053), 1.0).value)
        rescale_intercept = float(dcm.get((0x0028, 0x1052), 0.0).value)

        # Apply rescale formula: HU = pixel_value * slope + intercept
        hu_array = pixel_array * rescale_slope + rescale_intercept

        return hu_array
