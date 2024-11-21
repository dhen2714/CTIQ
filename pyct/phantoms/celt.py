from ..processing import circle_centre_subpixel
from ..roi_tools import get_roi_from_centre_pixel, ContrastInsertROI
from skimage.transform import rescale
from skimage.feature import match_template
from skimage.filters import gaussian
import numpy as np

INSERT_DIAMETER_MM = 25
INSERT_DIST_FROM_CENTRE_MM = 50
INSERT_ANGULAR_POSITION_DEG = 45
TEMPLATE_SIZE_MM = 120
TEMPLATE_PIX_SIZE = 0.1
NPS_ROI_DIM_MM = 35


def _get_insert_inds(insert_centre_pos: tuple[float], FX: np.ndarray, FY: np.ndarray):
    FY, FX = FY - insert_centre_pos[0], FX - insert_centre_pos[1]
    return np.where(
        np.sqrt(FX**2 + FY**2) < (INSERT_DIAMETER_MM / 2 / TEMPLATE_PIX_SIZE)
    )


def get_insert_template():
    """
    Template for finding the centre of the CeLT phantom in an axial slice.
    """
    template_dim = int(TEMPLATE_SIZE_MM / TEMPLATE_PIX_SIZE)
    template = np.zeros((template_dim, template_dim))
    template_centre = (template_dim - 1) / 2
    angular_pos = np.radians(INSERT_ANGULAR_POSITION_DEG)
    row_offset = INSERT_DIST_FROM_CENTRE_MM * np.sin(angular_pos) / TEMPLATE_PIX_SIZE
    col_offset = INSERT_DIST_FROM_CENTRE_MM * np.cos(angular_pos) / TEMPLATE_PIX_SIZE
    centre = template_centre, template_centre
    top_left = template_centre - row_offset, template_centre - col_offset
    top_right = template_centre - row_offset, template_centre + col_offset
    bottom_left = template_centre + row_offset, template_centre - col_offset
    bottom_right = template_centre + row_offset, template_centre + col_offset
    FX, FY = np.meshgrid(np.arange(template_dim), np.arange(template_dim))

    for insert_centre_pos in [
        centre,
        top_left,
        top_right,
        bottom_left,
        bottom_right,
    ]:
        insert_indices = _get_insert_inds(insert_centre_pos, FX, FY)
        template[insert_indices] = 1
    return template


def get_phantom_centre(
    axial_slice: np.ndarray, template: np.ndarray, threshold: float = 0.5
) -> tuple[int]:
    """
    Matches given template to an axial slice, returns (row, column) index of best
    match between slice and template. template must have smaller dimensions than
    axial_slice.
    """
    axial_slice = np.abs(axial_slice - axial_slice.mean())
    axial_slice = gaussian(axial_slice, sigma=2)
    result = match_template(axial_slice, template, pad_input=True)
    result = np.abs(result)
    corr_val = result.max()
    if corr_val > threshold:
        phantom_centre = np.where(result == result.max())
        phantom_centre = phantom_centre[0][0], phantom_centre[1][0]
    else:
        phantom_centre = None
    return phantom_centre


class Celt:

    def __init__(
        self,
        pixel_array3d: np.ndarray,
        slice_locations: np.array,
        pixel_dim_mm: tuple[float],
    ) -> None:
        self.array = pixel_array3d
        self.slice_locations = slice_locations
        self.pixel_dim_mm = pixel_dim_mm
        self.slice_idx_2_template_centre = dict()
        template = get_insert_template()
        # Rescale the template to the pixel size of the image
        self.template = rescale(template, TEMPLATE_PIX_SIZE / pixel_dim_mm[0])
        angular_pos = np.radians(INSERT_ANGULAR_POSITION_DEG)
        self.nps_roi_dim = int(NPS_ROI_DIM_MM / pixel_dim_mm[0])
        self.nps_roi_offset = int(
            INSERT_DIST_FROM_CENTRE_MM * np.sin(angular_pos) / pixel_dim_mm[0]
        )

        self.background_means = dict()
        self.background_stddevs = dict()

    def get_nps_rois(
        self, slice_index: int | list[int], match_threshold: float = 0.5
    ) -> np.ndarray:
        """
        Returns a 3D numpy array of ROIs.
        ROIs are drawn over the uniform parts of the CeLT phantom, near the
        tube inserts. For each slice, 4 ROIs will be returned.
        ROIs can either be from a single slice or multiple slices.
        """
        if type(slice_index) is int:
            nps_rois = self._get_nps_rois_single_slice(
                slice_index, match_threshold=match_threshold
            )
            return np.array(nps_rois)

        nps_rois = []
        for ind in slice_index:
            nps_rois_one_slice = self._get_nps_rois_single_slice(
                ind, match_threshold=match_threshold
            )
            nps_rois.extend(nps_rois_one_slice)
        return np.array(nps_rois)

    def _get_nps_rois_single_slice(
        self, slice_index: int, match_threshold: float = 0.5
    ) -> list[np.ndarray]:
        nps_rois = []
        axial_slice = self.array[slice_index]
        if slice_index in self.slice_idx_2_template_centre.keys():
            phantom_centre = self.slice_idx_2_template_centre[slice_index]
        else:
            phantom_centre = get_phantom_centre(
                axial_slice, self.template, threshold=match_threshold
            )
            self.slice_idx_2_template_centre[slice_index] = phantom_centre

        if phantom_centre:
            top = phantom_centre[0] - self.nps_roi_offset, phantom_centre[1]
            left = phantom_centre[0], phantom_centre[1] + self.nps_roi_offset
            bottom = phantom_centre[0] + self.nps_roi_offset, phantom_centre[1]
            right = phantom_centre[0], phantom_centre[1] - self.nps_roi_offset

            for nps_roi_centre in [top, left, bottom, right]:
                roi = get_roi_from_centre_pixel(
                    axial_slice, nps_roi_centre, self.nps_roi_dim
                )
                nps_rois.append(roi)

            if slice_index not in self.background_means.keys():
                self.background_means[slice_index] = np.array(nps_rois).mean()
                self.background_stddevs[slice_index] = np.array(nps_rois).std()
        else:
            nps_rois = None
        return nps_rois

    def get_ttf_rois(
        self, slice_index: int, match_threshold: float = 0.5
    ) -> dict[str, ContrastInsertROI]:
        """
        CeLT phantom insert ROIs for TTF calculation. First ROI in output list
        is centre, then top right, bottom right, bottom left, top left.
        """
        ttf_rois = dict()
        insert_radius_px = (INSERT_DIAMETER_MM / 2) / self.pixel_dim_mm[0]
        ttf_roi_dim = int(4 * insert_radius_px)
        px_offset = self.nps_roi_offset
        axial_slice = self.array[slice_index]
        if slice_index in self.slice_idx_2_template_centre.keys():
            centre = self.slice_idx_2_template_centre[slice_index]
        else:
            centre = get_phantom_centre(
                axial_slice, self.template, threshold=match_threshold
            )
            self.slice_idx_2_template_centre[slice_index] = centre

        if centre:
            roi_centres = {
                "centre": (centre[0], centre[1]),
                "top_right": (centre[0] - px_offset, centre[1] + px_offset),
                "bottom_right": (centre[0] + px_offset, centre[1] + px_offset),
                "bottom_left": (centre[0] + px_offset, centre[1] - px_offset),
                "top_left": (centre[0] - px_offset, centre[1] - px_offset),
            }
            for roi_name, roi_centre in roi_centres.items():
                roi = get_roi_from_centre_pixel(axial_slice, roi_centre, ttf_roi_dim)
                if slice_index not in self.background_means.keys():
                    self._get_nps_rois_single_slice(slice_index)
                # Subtracting background mean helps template matching
                background_mean = self.background_means[slice_index]
                subpixel_centre = circle_centre_subpixel(
                    roi - background_mean, insert_radius_px, supersample_factor=5
                )
                ttfroi = ContrastInsertROI(
                    name=f"{roi_name}_slice{slice_index}",
                    roi=roi,
                    pixel_size_mm=self.pixel_dim_mm[0],
                    rod_centre=subpixel_centre,
                    rod_radius_mm=(INSERT_DIAMETER_MM / 2),
                )
                ttf_rois[roi_name] = ttfroi
        else:
            ttf_rois = None
        return ttf_rois
