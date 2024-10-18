import numpy as np
import pydicom
from pathlib import Path


def get_image_directory_paths(parent_dir: str | Path) -> list[Path]:
    """
    Returns list of directories that don't contain other directories.
    Assumption is that these directories contain the DICOM images.
    """
    stem_dirs = []
    has_subdirs = False
    for dir_entry in Path(parent_dir).iterdir():
        if dir_entry.is_dir():
            has_subdirs = True
            stem_dirs.extend(get_image_directory_paths(dir_entry))
    if not has_subdirs:
        stem_dirs.append(parent_dir)
    return stem_dirs


def get_dicom_tag_values(
    dcm: pydicom.FileDataset, tag_list: list[tuple[int, int]]
) -> list[str]:
    """
    Retrieve the values for specific DICOM tags from a given DICOM file.

    Parameters
    ----------
    dcm : pydicom.FileDataset
        The DICOM file dataset object from which to extract tag values.
    tag_list : List[Tuple[int, int]]
        A list of DICOM tags to retrieve values for, where each tag is represented as a tuple (group, element).

    Returns
    -------
    List[str]
        A list of values corresponding to the specified DICOM tags in the order they are provided.
        If a tag is not found, `None` is returned for that tag.

    Example
    -------
    >> dcm = pydicom.dcmread('path/to/dicom/file.dcm')
    >> tag_values = get_dicom_tag_values(dcm, [(0x0010, 0x0010), (0x0010, 0x0020)])
    >> print(tag_values)
    ['John Doe', '123456']
    """
    output_tag_values = []
    tag_set = set(element.tag for element in dcm)

    for dcm_tag in tag_list:
        tag = pydicom.tag.Tag(dcm_tag[0], dcm_tag[1])
        if tag not in tag_set:
            output_tag_values.append(None)
        else:
            value = dcm.get(tag, None).value
            output_tag_values.append(str(value) if value is not None else None)

    return output_tag_values


def get_series_dicom_tag_values(
    image_dir: str | Path, tag_list: list[tuple]
) -> list[str]:
    """
    Retrieves dicom tag values of a single, randomly sampled file from a
    directory containing DICOM images.
    """
    file_list = list(Path(image_dir).iterdir())
    try:
        file_sample = np.random.choice(file_list, size=1)[0]
    except ValueError:  # For empty directories.
        return [None]
    dcm = pydicom.dcmread(file_sample, stop_before_pixels=True)
    return get_dicom_tag_values(dcm, tag_list)
