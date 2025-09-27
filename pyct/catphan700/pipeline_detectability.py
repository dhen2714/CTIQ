from .ctp682 import ContrastInsertROI, INNER_DIAMETER_MM, PHANTOM_DIAMETER_MM
from .ctp682 import get_CTP682_contrast_rois
from .localisation import locate_all_segments
from ..detectability import calculate_dprime_npw, calculate_dprime_npwei
from ..processing import detrend, circle_centre_subpixel
from ..roi_tools import ROIBounds, get_roi_bounds_from_centre_pixel, get_roi
from ..nps import NPS1D, NPS2D, subrois_from_rois, nps2d_from_subrois
from ..series import AxialSeries
from ..ttf import TTF, calculate_ttf, get_cutoff_frequency
import numpy as np
from dataclasses import dataclass
from pathlib import Path


class TTFResult:
    def __init__(
        self, result_name: str, contrast_roi: ContrastInsertROI, ttf_result: TTF
    ):
        self.name = result_name
        self.ttf = ttf_result.ttf
        self.mtf = ttf_result.mtf
        self.esf = ttf_result.esf
        self.lsf = ttf_result.lsf
        self.f = ttf_result.f
        self.r = ttf_result.r
        self.roi = contrast_roi.roi
        self.contrast = contrast_roi.contrast()
        self.cnr = contrast_roi.cnr()
        self.HU = contrast_roi.mean()
        self.esf_conditioning = ttf_result.esf_conditioning

    @property
    def f50(self) -> float:
        return get_cutoff_frequency(
            self.f, self.ttf, cutoff_point=0.5, mtf_threshold=0.49
        )

    @property
    def f10(self) -> float:
        return get_cutoff_frequency(
            self.f, self.ttf, cutoff_point=0.1, mtf_threshold=0.09
        )


class NPSResult:
    def __init__(
        self, result_name: str, nps_result: NPS2D, roi_bounds: list[list[ROIBounds]]
    ):
        self.name = result_name
        self.npsr = nps_result.get_radial()
        self.npsh = nps_result.get_horizontal()
        self.npsv = nps_result.get_vertical()
        self.roi_bounds = roi_bounds
        self.roi_dimensions = nps_result.roi_dimensions
        self.num_rois = nps_result.num_rois
        self.nps2d = nps_result

    @property
    def fpeak(self) -> float:
        return self.npsr.fpeak

    @property
    def favg(self) -> float:
        return self.npsr.favg

    @property
    def f(self) -> np.ndarray:
        return self.npsr.f

    @property
    def nps(self) -> np.ndarray:
        return self.npsr.nps


@dataclass
class TTFSettings:
    detrend_method: str = "poly"
    samples: str | int = "all"
    cnr_check: bool = True
    esf_conditioning: bool = False
    esf_radius_multiplier_px: float = 2.5
    window_width: int = 15


@dataclass
class NPSSettings:
    samples: str | int = "all"
    update_centre: bool = False
    pad_size: int = 64
    detrend_method: str = "poly"


@dataclass
class DetectabilitySettings:
    observer: str = "NPW"
    task_diameter_mm: float = 1
    task_fov_mm: float = 20
    task_pix_mm: float = 0.1
    display_zoom: float = 3
    display_pitch_mm: float = 0.2
    display_distance_mm: float = 500
    display_alpha: float = 5

    @property
    def task_radius_mm(self) -> float:
        return self.task_diameter_mm / 2


@dataclass
class DetectabilityResult:
    name: str
    d: float
    ttf: TTFResult
    nps: NPSResult
    settings: DetectabilitySettings
    ttf_settings: TTFSettings
    nps_settings: NPSSettings


def get_nps_ctp714_indices(
    segment_indices: np.ndarray, slice_interval: float, samples: str | int = "all"
) -> np.ndarray:
    # Take 10 mm in the centre of the segment and ignore these slices
    central_buffer_len = np.ceil(10 / slice_interval).astype(int)

    array_length = len(segment_indices)
    start_index = (array_length - central_buffer_len) // 2
    end_index = start_index + central_buffer_len
    valid_indices = np.concatenate(
        [segment_indices[:start_index], segment_indices[end_index:]]
    )
    if samples == "all":
        return valid_indices
    elif isinstance(samples, int):
        if samples <= 0:
            raise ValueError("Number of samples must be positive")
        if samples >= len(valid_indices):
            return valid_indices
        else:
            # Randomly sample without replacement
            return valid_indices[:samples]
    else:
        raise ValueError("samples must be 'all' or a positive integer")


def calculate_detectability(
    ttf_result: TTF,
    nps_result: NPS1D,
    task_contrast: float,
    settings: None | DetectabilitySettings = None,
) -> float:
    if settings is None:
        settings = DetectabilitySettings()

    if settings.observer.lower() == "npw":
        d = calculate_dprime_npw(
            task_contrast,
            settings.task_radius_mm,
            ttf_result,
            nps_result,
            settings.task_fov_mm,
            settings.task_pix_mm,
        )
    elif settings.observer.lower() == "npwei":
        d = calculate_dprime_npwei(
            task_contrast,
            settings.task_radius_mm,
            ttf_result,
            nps_result,
            settings.task_fov_mm,
            settings.task_pix_mm,
            settings.display_zoom,
            settings.display_pitch_mm,
            settings.display_distance_mm,
            settings.display_alpha,
        )
    else:
        raise ValueError("Observer model must be 'NPW' or 'NPWEi'.")
    return d


def catphan700_auto_detectability(
    series_dir: str | Path,
    detectability_settings: None | DetectabilitySettings = None,
    ttf_settings: None | TTFSettings = None,
    nps_settings: None | NPSSettings = None,
):
    if detectability_settings is None:
        detectability_settings = DetectabilitySettings()  # use default settings
    if ttf_settings is None:
        ttf_settings = TTFSettings()
    if nps_settings is None:
        nps_settings = NPSSettings()

    ctseries = AxialSeries(series_dir, validate_dimensions=True)
    ctarray = ctseries.get_array()
    pix_dim = ctseries.pixel_size
    slice_locs = ctseries.slice_locations
    slice_interval = ctseries.slice_interval
    orientation = ctseries.orientation

    catphan_segments = locate_all_segments(
        ctarray, slice_locs, orientation, segment_buffer_mm=2
    )
    segment_dict = {
        segment_object.name: segment_object for segment_object in catphan_segments
    }
    ttf_segment = segment_dict["CTP682"]
    ttf_segment_averaged = ctarray[ttf_segment.indices].mean(axis=0)
    ttf_results_dict = calculate_all_ttfs(
        ttf_segment_averaged,
        pix_dim,
        orientation,
        detrend_method=ttf_settings.detrend_method,
        cnr_check=ttf_settings.cnr_check,
        esf_conditioning=ttf_settings.esf_conditioning,
        esf_radius_multiplier_px=ttf_settings.esf_radius_multiplier_px,
        window_width=ttf_settings.window_width,
    )

    nps_segment = segment_dict["CTP714"]
    nps_indices = get_nps_ctp714_indices(
        nps_segment.indices, slice_interval, nps_settings.samples
    )
    nps_array = ctarray[nps_indices]
    nps_result = calculate_nps_ctp714(
        nps_array, pix_dim, nps_settings.update_centre, nps_settings.pad_size
    )
    nps1d = nps_result.npsr

    detectability_results_dict = dict()

    for material_name, ttf_result in ttf_results_dict.items():
        contrast = ttf_result.contrast
        d = calculate_detectability(ttf_result, nps1d, contrast, detectability_settings)
        dresult = DetectabilityResult(
            material_name,
            d,
            ttf_result,
            nps_result,
            detectability_settings,
            ttf_settings,
            nps_settings,
        )
        detectability_results_dict[material_name] = dresult

    return detectability_results_dict


def calculate_insert_ttf(
    contrast_insert: ContrastInsertROI,
    detrend_method: str = "poly",
    cnr_check: bool = True,
    esf_conditioning: bool = False,
    esf_radius_multiplier_px: float = 2.5,
    window_width: int = 15,
) -> TTF:
    if cnr_check:
        cnr = contrast_insert.cnr()
        if abs(cnr) < 15:
            esf_conditioning = True

    roi = contrast_insert.roi
    background_roi_bounds = contrast_insert.get_central_roi_bounds(8)
    rows, cols = np.indices(roi.shape[:2])
    background_mask = (rows - background_roi_bounds.centre_row) ** 2 + (
        cols - background_roi_bounds.centre_col
    ) ** 2 > background_roi_bounds.radius**2
    detrended_roi = detrend(roi, detrend_method, background_mask)

    esf_sample_radius_px = esf_radius_multiplier_px * contrast_insert.rod_radius_px

    return calculate_ttf(
        detrended_roi,
        contrast_insert.rod_centre,
        esf_sample_radius_px,
        pixel_size_mm=contrast_insert.pixel_size_mm[0],
        esf_conditioning=esf_conditioning,
        window_width=window_width,
    )


def calculate_nps_ctp714(
    ctarray: np.ndarray,
    pixel_size_mm: tuple[float, float],
    update_centre_pixel: bool = False,
    pad_size: int = 64,
):
    phantom_radius_px = (PHANTOM_DIAMETER_MM / 2) / pixel_size_mm[0]
    inner_diameter_px = INNER_DIAMETER_MM / pixel_size_mm[0]
    if inner_diameter_px > 3 * 92:
        roi_dim = 92
    elif inner_diameter_px > 3 * 64:
        roi_dim = 64
    else:
        raise ValueError(
            "Pixel size too large, cannot automatically select large enough ROIs for NPS."
        )
    phantom_centre = circle_centre_subpixel(
        ctarray[0], phantom_radius_px, template_padding_px=-30, supersample_factor=2
    )
    all_rois = []
    roi_bound_record = []
    for i, ctslice in enumerate(ctarray):
        slice_roi_bounds = []
        slice_rois = []
        roi_bounds = get_roi_bounds_from_centre_pixel(phantom_centre, roi_dim)
        slice_roi_bounds.append(roi_bounds)
        slice_rois.append(get_roi(ctslice, roi_bounds))

        if i > 0 and update_centre_pixel:
            phantom_centre = circle_centre_subpixel(
                ctarray[0],
                phantom_radius_px,
                template_padding_px=-30,
                supersample_factor=2,
            )
        for offset in (-roi_dim, roi_dim):

            new_roi_centre = [phantom_centre[0] + offset, phantom_centre[1]]
            roi_bounds = get_roi_bounds_from_centre_pixel(new_roi_centre, roi_dim)
            slice_roi_bounds.append(roi_bounds)
            slice_rois.append(get_roi(ctslice, roi_bounds))

            new_roi_centre = [phantom_centre[0], phantom_centre[1] + offset]
            roi_bounds = get_roi_bounds_from_centre_pixel(new_roi_centre, roi_dim)
            slice_roi_bounds.append(roi_bounds)
            slice_rois.append(get_roi(ctslice, roi_bounds))

        all_rois.extend(slice_rois)
        roi_bound_record.append(slice_roi_bounds)

    all_subrois = subrois_from_rois(all_rois, 64)
    nps_result = nps2d_from_subrois(
        all_subrois, pixel_size_mm, detrend_method="poly", pad_size=pad_size
    )
    return NPSResult("CTP712", nps_result, roi_bound_record)


def calculate_all_ttfs(
    ctslice: np.ndarray,
    pixel_size_mm: tuple[float, float],
    orientation: str = "HFS",
    detrend_method: str = "poly",
    cnr_check: bool = True,
    esf_conditioning: bool = False,
    esf_radius_multiplier_px: float = 2.5,
    window_width: int = 15,
) -> dict[TTFResult]:
    contrast_insert_dict = get_CTP682_contrast_rois(ctslice, pixel_size_mm, orientation)
    results_dict = dict()
    for insert_name, contrast_insert in contrast_insert_dict.items():
        ttf_val = calculate_insert_ttf(
            contrast_insert,
            detrend_method,
            cnr_check,
            esf_conditioning,
            esf_radius_multiplier_px,
            window_width,
        )
        ttf_result = TTFResult(insert_name, contrast_insert, ttf_val)
        results_dict[insert_name] = ttf_result

    return results_dict
