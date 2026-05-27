"""Digital Slide Archive interactions.

This package contains helper functions for interfacing with the Digital Slide
Archive.
Most of these functions are wrappers or helpers for the histomicstk library.
"""

import girder_client
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_scale_factor_and_appendStr,
    get_image_from_htk_response,
    get_bboxes_from_slide_annotations,
)
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import rasterio.features
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from typing import Union


def dsa_connection(api_url: str, api_key: str) -> girder_client.GirderClient:
    """Connect to a DSA server.

    Parameters
    ----------
    api_url : str
        URL to the API endpoint.

    api_key : str
        API key for the DSA server.

    Returns
    -------
    gc : girder_client.GirderClient
        An authenticated Girder client object.
    """
    gc = girder_client.GirderClient(apiUrl=api_url)
    gc.authenticate(apiKey=api_key)
    return gc


def get_collection_id(conn: girder_client.GirderClient,
                      collection_name: str) -> str:
    """Retrieve the ID of a given collection.

    Parameters
    ----------
    conn : girder_client.GirderClient
        An open, authenticated connection to the Digital Slide Archive
    collection_name :str
        The name of the collection.
        
    Returns
    -------
    collection_id : str
        The collection ID in string format.
    """
    collection_id = None

    # List all collections and find the target one
    collection_list = list(conn.listCollection())

    assert len(collection_list) > 0, \
        f"Cannot find collection named {collection_name} on Histomics. " \
        + "Please check that the server connection is working, that you have " \
        + "access to the collection, and that you are spelling everything " \
        + "correctly."

    for collection in collection_list:
        if collection['name'] == collection_name:
            collection_id = collection['_id']
    assert collection_id, f"Connected, but could not find {collection_name}."

    return collection_id


def get_folder_id(conn: girder_client.GirderClient,
                  folder_path: str,
                  search_limit: int = 1000) -> str | None:
    """Given a folder name and connection, return the folder ID number.

    Parameters
    ----------
    conn : girder_client.GirderClient
        An open, authenticated connection to the Digital Slide Archive
    folder_path : str
        Path to the folder, starting from the collection name. Note that this 
        should not include a leading or trailing '/', these will be handled 
        internally.
    search_limit : int, optional
        How many folders for the DSA to look through. The search function is
        fuzzy, so we have to limit the amount of returned results. Should be
        high enough that we don't have to worry about accidentally missing
        the target folder.

    Returns
    -------
    folder_result_id : str or None
        If found, returns the ID of the first folder that matches the input 
        string for `folder_path`. If not found, return None.
    """

    folder_name = folder_path.split('/')[-1]

    # Get a list of all folders that match the target (terminal) folder name
    folder_results = conn.get(f'/folder?parentType=folder&text={folder_name}&limit={search_limit}&sort=lowerName&sortdir=1')

    # Cycle through the results and validate that the paths match with our list of folder parts
    for folder_result in folder_results:
        folder_result_id = folder_result['_id']

        # Get the folder path for this folder
        folder_root_objects = conn.get(f'/folder/{folder_result_id}/rootpath')

        folder_root_path_parts = []

        for folder_root_object in folder_root_objects:
            if 'login' in folder_root_object['object'].keys():
                folder_root_path_parts.append(folder_root_object['object']['login'])
            elif 'name' in folder_root_object['object'].keys():
                folder_root_path_parts.append(folder_root_object['object']['name'])
            else:
                print(f'WARNING: Cannot identify the type of {folder_root_object}')
                return None
            
        # Append the folder name as well
        folder_root_path_parts.append(folder_result['name'])

        if '/'.join(folder_root_path_parts) == folder_path:
            return folder_result_id
    
    return None


def get_sample_id(conn: girder_client.GirderClient,
                  sample_name: str,
                  folder_path: str) -> str | None:
    """Retrieve the ID of a sample, given a name and a folder path.

    Parameters
    ----------
    conn : girder_client.GirderClient
        An open, authenticated connection to the Digital Slide Archive.
    sample_name : str
        The name of the desired sample.
    folder_path : str
        The containing folder path, starting with the collection name.
    
    Returns
    -------
    id : str or None
        Sample ID (if found) or None (if not found).
    """
    id_list, item_list = ids_names_from_htk(conn, folder_path)

    for (id, item) in zip(id_list, item_list):
        if item == sample_name:
            return id
    print(f'Did not find sample {sample_name} in {folder_path}. Please recheck your access and spelling of the path.')
    return None


def image_metadata(conn: girder_client.GirderClient,
                   sample_id: str) -> dict:
    """Return the image metadata given sample ID.

    This is just a wrapper around the girder API call which can be hard to 
    remember.

    Parameters
    ----------
    conn : girder_client.GirderClient
        An open, authenticated connection to the Digital Slide Archive.
    sample_id : str
        Sample ID of the desired slide metadata.

    Returns
    -------
    resp : dict
        Set of metadata available for the image. Keys may vary depending on
        the slide format.
    """
    return conn.get(f'/item/{sample_id}/tiles')


def ids_names_from_htk(conn: girder_client.GirderClient,
                       folder_path: str) -> tuple[list, list]:
    """Retrieve all item IDs and item names in a folder.
    
    Parameters
    ----------
    conn : girder_client.GirderClient
        An open, authenticated connection to the Digital Slide Archive.
    folder_path : str
        Path of the folder, including collection name and without a trailing 
        slash.

    Returns
    -------
    item_ids : list
        List of the ID values in string format.
    item_names : list
        List of the item names in string format.
    """
    
    folder_id = get_folder_id(conn, folder_path)
    
    item_list_htk = conn.listItem(folder_id)

    # Parse the retrieved item list
    item_ids = []
    item_names = []
    for item in item_list_htk:
        item_ids.append(item['_id'])
        item_names.append(item['name'])

    return item_ids, item_names


def slide_annotations(conn: girder_client.GirderClient,
                      slide_id: str,
                      target_mpp: float,
                      group_list: list = []) -> tuple[pd.Series | pd.DataFrame | None, float, str] | None:
    """Return a single slide's annotations.

    Accepts an optional annotation_list parameter indicating the groups to get.
    If not provided, grab everything.

    Parameters
    ----------
    conn : girder_client.GirderClient
        An open, authenticated connection to the Digital Slide Archive
    slide_id : str
        The slide ID number as a string.
    target_mpp : float
        The desired target microns per pixel of the annotation objects.
    group_list : list or None
        If specified, filter the set of retrieved annotations by the given 
        group / class names.
    """
    # Case-insensitive group list!
    if group_list is not None:
        gl = [x.lower() for x in group_list]
    else:
        gl = []

    # Pull down the annotation objects
    try:
        annotations_resp = conn.get('/annotation/item/' + slide_id)
    except girder_client.HttpError:
        # The server couldn't find the object
        print(f'Could not find an item on the server with id {slide_id}.')
        return None
    except Exception:
        # Unclear why this failure happened
        print(f'Something went wrong getting annotations for {slide_id}.')
        return None

    # Do they exist? If not, return false
    if len(annotations_resp) == 0:
        print(f'No annotations were found for {slide_id}.')
        return None

    # Get the scale factor and string for this slide
    scale_factor, appendStr = get_scale_factor_and_appendStr(conn,
                                                             slide_id,
                                                             MPP=float(target_mpp),
                                                             MAG=None)

    # Get the info of the elements based on the now-scaled annotations
    element_infos = get_bboxes_from_slide_annotations(annotations_resp)

    # Filter the dataframe if we asked for a specific set of classes
    if len(gl) > 0:
        element_infos = element_infos[element_infos['group'].isin(gl)]

    return element_infos, scale_factor, appendStr


def slide_elements(conn: girder_client.GirderClient,
                   item_id: str,
                   group_list: list | None = None) -> tuple[list, dict] | None:
    """Retrieve a list of elements from the HTK annotation response.

    Each element in this list corresponds to a polygon.
    Optionally, ask for elements that belong to a specific group or list of groups.

    Parameters
    ----------
    conn : girder_client.GirderClient
        An open, authenticated connection to the Digital Slide Archive
    item_id : str
        The ID value of the item from which to grab elements.
    group_list : list, optional
        List of groups (classes) to retrieve. If None (default), retrieve all classes.

    Returns
    -------
    target_elements : list
        List of the target annotation elements, filtered by `group_list` (if included).
    img_metadata : dict
        Dictionary of the image metadata, used to identify source spatial resolution.
    """
    # Pull down the annotation objects
    annotations_resp = conn.get(f'annotation/item/{item_id}')
    img_metadata = conn.get(f'item/{item_id}/tiles')

    if len(annotations_resp) == 0:
        print('The annotation response had a length of Zero')
        return None

    # Initialize the gt annotation holder
    target_elements = []

    # Cycle through each annotation on this item
    for annotation in annotations_resp:
        elements = annotation['annotation']['elements']
        # Cycle through each of the annotation elements
        for element in elements:
            # Check that this item has a group (i.e. a class)
            if 'group' in element.keys():
                # If this group is what we're looking for, then pull it
                if group_list is not None and element['group'] in group_list:
                    target_elements.append(element)
                elif group_list is None:
                    target_elements.append(element)

    return target_elements, img_metadata


def image_data(conn: girder_client.GirderClient,
               sample_id: str,
               bounds_dict: dict,
               appendStr: str | None = None) -> NDArray | None:
    """Return a numpy image defined by the connection, sample_id, and ROI.

    Parameters
    ----------
    conn : girder_client.GirderClient
        An open, authenticated connection to the Digital Slide Archive
    sample_id : str
        The sample ID in string form.
    bounds_dict : dict
        A dictionary with 'xmin', 'ymin', 'xmax', 'ymax' keys.
    appendStr : str, optional
        String mostly used to define desired spatial resolution. If not given,
        download at original spatial resolution.
    """
    # Convert the keys of the bounds_dict to lowercase
    bounds_dict = {k.lower(): v for k, v in bounds_dict.items()}

    # Ensure the keys are present for the bounding box
    assert 'xmin' in bounds_dict and \
        'xmax' in bounds_dict and \
        'ymin' in bounds_dict and \
        'ymax' in bounds_dict, \
        'bounds_dict is not formatted properly. ' \
        'Please make sure xmin, xmax, ymin, ymax is included in the keys.'

    getStr = f"item/{sample_id}/tiles/region?" + \
        f"left={bounds_dict['xmin']}&" + \
        f"right={bounds_dict['xmax']}&" + \
        f"top={bounds_dict['ymin']}&" + \
        f"bottom={bounds_dict['ymax']}"

    if appendStr is not None:
        getStr += appendStr

    # Get the image raw response
    try:
        resp = conn.get(getStr, jsonResp=False)
    except Exception:
        print(f'{Exception}: Could not retrieve image response with {getStr}.')
        return None

    # Sometimes this fails, and I'm not sure why
    try:
        img_roi = np.array(get_image_from_htk_response(resp))
    except Exception:
        print(f'{Exception}: Could not convert image response {resp} to image')
        return None

    return img_roi


def slide_roi(conn: girder_client.GirderClient,
              sample_id: str,
              bounds: dict,
              target_mpp: Union[float, None] = None) -> NDArray:
    """Use HTK to pull an ROI from this image as a numpy array.

    Parameters
    ----------
    conn : girder_client.GirderClient
        An open, authenticated connection to the Digital Slide Archive

    """
    if target_mpp is None:
        return np.array(image_data(conn, sample_id, bounds_dict=bounds))
    else:
        _, appendStr = get_scale_factor_and_appendStr(conn,
                                                      sample_id,
                                                      MPP=target_mpp,
                                                      MAG=None)
        return np.array(image_data(conn, sample_id, bounds_dict=bounds, appendStr=appendStr))


def tile_polygon(slide_resolution: float,
                 polygon: Union[Polygon, MultiPolygon],
                 tile_size: int = 1024,
                 target_mpp: Union[float, None] = None,
                 stride: Union[int, float, None] = None,
                 edges: str = "within") -> list:
    """Retrieve a list of possibly overlapping tiles within a surrounding polygon.

    Parameters
    ----------
    slide_resolution : float
        The resolution of the slide on the Digital Slide Archive.
    polygon : Polygon or MultiPolygon
        The enclosing polygon object you wish to tile.
    tile_size : int
        Desired resulting tile size.
    target_mpp : float or None
        Desired resulting spatial resolution. If None, it will be set to the 
        slide resolution.
    edges : {'within', 'overlaps', 'both'}
        How to handle tiles at the border of the defining polygon.
    stride : int, float, or None
        Step size between tile origins.

        If None, uses non-overlapping tiles.

        If 0 < stride < 1, it is interpreted as a fraction of tile_size.
        Example: stride=0.5 gives 50% overlap.

        If stride >= 1, it is interpreted as an absolute stride in output
        tile pixels before conversion to slide-resolution coordinates.

    Returns
    -------
    tile_polygons : list
        List of the tile polygon objects in shapely format.
    """

    assert edges in ['within', 'overlaps', 'both'], (
        f'{edges} is not a valid overlap designation, use "within", "overlaps", or "both".'
    )

    if target_mpp is None:
        target_mpp = slide_resolution

    mpp_ratio = target_mpp / slide_resolution
    mod_tile_size = int(np.ceil(tile_size * mpp_ratio))

    if stride is None:
        mod_stride = mod_tile_size
    else:
        stride = float(stride)

        if stride <= 0:
            raise ValueError("stride must be > 0")

        if stride < 1:
            # Fractional stride, e.g. 0.5 means move half a tile each step.
            mod_stride = int(np.ceil(tile_size * stride * mpp_ratio))
        else:
            # Absolute stride in output tile pixels.
            mod_stride = int(np.ceil(stride * mpp_ratio))

    mod_stride = max(1, mod_stride)

    # Get the bounding box of the polygon
    minx, miny, maxx, maxy = polygon.bounds

    # Use stride instead of tile size for overlapping tiles
    left_coords = np.arange(minx, maxx, mod_stride)
    top_coords = np.arange(miny, maxy, mod_stride)

    tile_polygons = []

    polygon_union = unary_union(polygon)

    for col in left_coords:
        for row in top_coords:
            tile_polygon = Polygon([
                (col, row),
                (col, row + mod_tile_size),
                (col + mod_tile_size, row + mod_tile_size),
                (col + mod_tile_size, row)
            ])

            if edges == "within":
                if tile_polygon.within(polygon_union) and not tile_polygon.overlaps(polygon_union):
                    tile_polygons.append(tile_polygon)

            elif edges == "overlaps":
                if tile_polygon.overlaps(polygon_union) and not tile_polygon.within(polygon_union):
                    tile_polygons.append(tile_polygon)

            elif edges == "both":
                if tile_polygon.overlaps(polygon_union) or tile_polygon.within(polygon_union):
                    tile_polygons.append(tile_polygon)

    return tile_polygons


def annotations(conn: girder_client.GirderClient, 
                sample_id: str) -> dict:
    """Obtain annotations dictionary for a sample.

    This function strips some of the data from each annotation and organizes 
    it into a dictionary of lists, so if you need e.g. user data, consider
    calling `list(conn.get(f'annotaiton/item/{sample_id}'))` directly.

    Parameters
    ----------
    conn : girder_client.GirderClient
        An open, authenticated connection to the Digital Slide Archive
    sample_id : str
        The sample id in string format

    Returns
    -------
    annotation_elements : dict
        Dictionary of annotations where keys are class / group names and 
        values are lists of annotation objects.
    """
    
    annotation_response = conn.get(f'annotation/item/{sample_id}')
    
    annotation_elements = dict()
    
    for annotation_object in annotation_response:
        annotation = annotation_object['annotation']
        elements_list = annotation['elements']
        
        for e in elements_list:
            # Ensure that the object has a class assigned to it; if not, assign "default"
            if 'group' not in e.keys():
                e['group'] = 'default'

            if e['group'] in annotation_elements.keys():
                annotation_elements[e['group']].append(e)
            else:
                annotation_elements[e['group']] = [e]
    
    return annotation_elements

