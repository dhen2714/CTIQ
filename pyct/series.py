import numpy as np
import pydicom
from pathlib import Path

IMAGE_POS_PATIENT = (0x0020, 0x0032)


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
    ) -> None:
        self._slices = []
        self.slice_locations = []
        self.file_paths = []
        self.errors = []
        self.custom_tags = {}
        if custom_tags:
            for key in custom_tags.keys():
                self.custom_tags[key] = []

        for dir_item in Path(directory).iterdir():

            try:
                dcm = pydicom.dcmread(dir_item, stop_before_pixels=(not load_images))
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

        sorted_indices = np.argsort(self.slice_locations).astype(int)
        self._slices = np.array(self._slices)[sorted_indices]
        self.slice_locations = np.array(self.slice_locations)[sorted_indices]
        self.file_paths = np.array(self.file_paths)[sorted_indices].astype(str)
        if custom_tags:
            for key, val in self.custom_tags.items():
                self.custom_tags[key] = np.array(val)[sorted_indices]

    @property
    def pixel_size(self):
        slice_pixel_dim = self[0][(0x0028, 0x0030)].value
        slice_pixel_dim = np.array(slice_pixel_dim).astype(float)
        return slice_pixel_dim

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

    def get_array(self) -> np.ndarray:
        array3d = []
        for dcm in self:
            array3d.append(dcm.pixel_array)

        return np.array(array3d)
