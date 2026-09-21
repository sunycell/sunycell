"""Tissue analysis routines.

This module contains functions related to whole slide image manipulation.
"""

from histomicstk.saliency.tissue_detection import (
    get_slide_thumbnail,
    get_tissue_mask
)
from histomicstk.preprocessing.color_normalization.\
    deconvolution_based_normalization import deconvolution_based_normalization
import girder_client
import numpy as np
from scipy import ndimage as ndi
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from skimage import measure
from skimage.color import rgb2hsv
from skimage.morphology import (
    disk,
    remove_small_objects,
    dilation,
)
from skimage.transform import resize
from typing import Union


def close_contour(contour: list):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def get_slide_thumb_ratio(
        slide_info: dict,
        thumb_height: int,
        thumb_width: int):
    """Given a slide metadata from the DSA, as well as a target height and
    width, return the ratio between them.
    """

    slide_width = slide_info['sizeX']
    slide_height = slide_info['sizeY']

    height_ratio = slide_height / thumb_height
    width_ratio = slide_width / thumb_width

    return height_ratio, width_ratio


def binary_mask_to_bounds(
        binary_mask: np.ndarray,
        tolerance: int = 0) -> list:
    """Converts a binary mask to COCO polygon representation

    Args:
        binary_mask: 2D binary numpy array where 1 represents the object
        tolerance: Max dist from original points of polygon to approximated
            polygonal chain. If tolerance == 0, use original coordinate array

    """
    pts = []

    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(
        binary_mask,
        pad_width=1,
        mode='constant',
        constant_values=0)

    contours = measure.find_contours(padded_binary_mask, 0.5)

    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue

        contour = np.flip(contour, axis=1)
        seg = contour.ravel().tolist()

        # after pad and subtract 1, might have -0.5
        seg = [0 if i < 0 else i for i in seg]

        xy = [(float(x), float(y)) for x, y in zip(seg[0::2], seg[1::2])]

        for i in range(len(xy)):
            xy[i] += (0,)

        pts.append(xy)

    return pts


def tissue_segmentation(
        conn: girder_client.GirderClient,
        slide: dict,
        max_size: int = 1000,
        disk_size: int = 1) -> Union[MultiPolygon, None]:
    """Given a DSA connection and a slide object, return a list of polygons
    for each tissue area in the image.

    Note that this code does not attempt to identify separate pieces of tissue,
    only regions of the WSI that are likely to contain tissue. Some close
    tissue areas may overlap.
    """

    W_target_Qupath = np.array([
        [0.6511078257574492, 0.7011930431234068, 0.29049426072255424],
        [0.2158989356208711, 0.8011960501132094, 0.5580972485873468],
        [0.315510575173205, -0.5981592020094376, 0.736653681186286],
    ])

    stain_unmixing_routine_params = {
        "stains": ["hematoxylin", "eosin"],
        "stain_unmixing_method": "macenko_pca",  # xu_snmf, macenko_pca
    }

    # Get thumbnail
    slide_thumbnail = get_slide_thumbnail(conn, slide["_id"])
    thumb_height, thumb_width = np.shape(slide_thumbnail)[:2]

    # Perform normalization
    slide_thumbnail_normalized = deconvolution_based_normalization(
        slide_thumbnail,
        W_target=W_target_Qupath,
        stain_unmixing_routine_params=stain_unmixing_routine_params)

    # Get the tissue mask
    mask_out_normalized, _ = get_tissue_mask(
        slide_thumbnail_normalized,
        deconvolve_first=True,
        n_thresholding_steps=1,
        sigma=1,
        min_size=30)

    mask_out_normalized = resize(
        mask_out_normalized == 0,
        output_shape=slide_thumbnail.shape[:2],
        order=0, preserve_range=True) == 1

    # Check whether the mask is "inside" or "outside" the tissue
    # Calculate the whiteness of the areas and make a call
    slide_sat = rgb2hsv(slide_thumbnail)[:, :, 1]
    pixels_in = slide_sat[np.where(mask_out_normalized > 0)]
    pixels_out = slide_sat[np.where(mask_out_normalized <= 0)]

    if np.mean(pixels_in) > np.mean(pixels_out):
        # Values inside the mask represent the tissue
        mask_out = mask_out_normalized > 0
    else:
        # Values outside the mask represent the tissue
        mask_out = mask_out_normalized <= 0

    # Slightly edit the mask and assign labels to each individual part
    mask_out = ndi.binary_fill_holes(mask_out)
    mask_out = remove_small_objects(mask_out, max_size=max_size)
    mask_labeled = measure.label(mask_out)

    # Convert mask to polygon
    slide_info = conn.get(f"/item/{slide['_id']}/tiles")

    height_ratio, width_ratio = get_slide_thumb_ratio(
        slide_info,
        thumb_height,
        thumb_width)

    tissue_idxes = np.unique(mask_labeled.ravel())

    tissue_polygons = []

    # Cycle through each tissue object (leave out first index, background)
    for tissue_idx in tissue_idxes[1:]:
        tissue_area = mask_labeled == tissue_idx

        # Restore labeled tissue to its original size
        tissue_area = dilation(tissue_area,
                               footprint=disk(disk_size))
        # Eliminate holes again
        tissue_area = ndi.binary_fill_holes(tissue_area)

        # Get the boundaries of the tissue area
        tissue_bounds = binary_mask_to_bounds(
            tissue_area)  # , mpp_ratio=width_ratio)

        for tissue_bound in tissue_bounds:
            tissue_x = [int(x[0] * width_ratio) for x in tissue_bound]
            tissue_y = [int(x[1] * width_ratio) for x in tissue_bound]
            tissue_coordinates = [[x, y] for x, y in zip(tissue_x, tissue_y)]

            # Combine into a polygon
            tissue_polygons.append(Polygon(tissue_coordinates))

    # Ensure all polygons are valid and merge them
    tissue_polygons = [p.buffer(0) for p in tissue_polygons]
    tissue_polygons = unary_union(tissue_polygons)

    # Ensure this is a MultiPolygon (even if we have one object)
    if type(tissue_polygons) is list:
        tissue_polygons = MultiPolygon(tissue_polygons)

    if type(tissue_polygons) is Polygon:
        tissue_polygons = MultiPolygon([tissue_polygons])

    return tissue_polygons
