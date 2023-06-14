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
    scale_slide_annotations
)
from histomicstk.saliency.tissue_detection import get_slide_thumbnail
import numpy as np
from sunycell import dsa
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util import Retry
import time
from typing import Any, Dict, Tuple, Optional, Union, Sequence, List, Callable


class DSAImage(dict):
    def __init__(
            self,
            conn: girder_client.GirderClient,
            collection_name: str = None,
            folder_name: str = None,
            image_name: str = None):
        
        self.conn = conn
        self.collection_name = collection_name
        self.folder_name = folder_name
        self.image_name = image_name
        self.folder_path = f'{collection_name}/{folder_name}'
    
    def __repr__(self):
        properties = []
        properties.extend([
            f'collection_name: {self.collection_name}',
            f'folder_name: {self.folder_name}',
            f'image_name: {self.image_name}',
            f'shape: {self.shape}',
            f'spatial_resolution: {self.resolution}',
            f'height: {self.height}',
            f'width: {self.width}'
        ])

        properties = '; '.join(properties)
        string = f'{self.__class__.__name__}({properties})'
        return string
    
    @property
    def collection_id(self) -> str:
        """Collection ID containing this image on the DSA instance."""
        return dsa.get_collection_id(conn=self.conn,
                                     collection_name=self.collection_name)
    
    @property
    def folder_id(self) -> str:
        """Folder ID of the image location on the DSA."""
        return dsa.get_folder_id(conn=self.conn,
                                 folder_path=self.folder_path)
    
    @property
    def sample_id(self) -> str:
        """Sample ID on the DSA instance."""
        return dsa.get_sample_id(conn=self.conn,
                                 sample_name=self.image_name,
                                 folder_path=self.folder_path)

    @property
    def metadata(self) -> dict:
        """Metadata dictionary returned from DSA."""
        return self._get_metadata()
    
    def _get_metadata(self) -> dict:
        """Retrieve metadata from the server.
        
        Metadata includes the following:
            levels: Number of magnification levels stored in the file
            magnification: Optical magnification of the slide
            mm_x: Spatial resolution in X, in millimeters per pixel
            mm_y:  Spatial resolution in Y, in millimeters per pixel
            sizeX: Width of the slide
            sizeY: Height of the slide
            tileHeight: Height of each tile in the TIFF
            tileWidth: Width of each tile in the TIFF
        """
        return dsa.image_metadata(self.conn, self.sample_id)
    
    @property
    def levels(self) -> int:
        """Number of levels in this image."""
        return self.metadata['levels']
        
    @property
    def resolution(self) -> float:
        """Spatial resolution in mpp (microns per pixel).
        
        Calculated as the average of the mm_x and mm_y properties.
        """
        return 1000 * (self.metadata['mm_x'] + self.metadata['mm_y']) / 2.0
    
    @property
    def height(self) -> int:
        """Image height, if 2D."""
        return self.metadata['sizeY']
    
    @property
    def width(self) -> int:
        """Image width, if 2D."""
        return self.metadata['sizeX']

    @property
    def shape(self) -> Tuple[int, int]:
        """Tensor shape as :math:`(W, H)`."""
        return tuple((self.width, self.height))
    

    def thumbnail(self) -> np.ndarray:
        return get_slide_thumbnail(self.conn, self._sample_id)

    
    def roi(self, bounds: dict, mpp: float = None) -> np.array:
        """Use HTK to pull an ROI from this image as a numpy array."""
        if mpp is None:
            return dsa.image_data(self.conn, self.sample_id, bounds_dict=bounds)
        else:
            scale_factor, appendStr = get_scale_factor_and_appendStr(self.conn,
                                                             self.sample_id,
                                                             MPP=float(mpp),
                                                             MAG=None)
            return dsa.image_data(self.conn, self.sample_id, bounds_dict=bounds, appendStr=appendStr)
    
    def annotations(self) -> list:
        """Obtain the annotations for the image, filtered by target_groups."""
        
        annotation_response = self.conn.get(f'annotation/item/{self.sample_id}')
        
        annotation_elements = dict()
        
        for annotation_object in annotation_response:
            annotation = annotation_object['annotation']
            elements_list = annotation['elements']
            
            for e in elements_list:
                if e['group'] in annotation_elements.keys():
                    annotation_elements[e['group']].append(e)
                else:
                    annotation_elements[e['group']] = [e]
        
        return annotation_elements


def dsa_connection(api_url: str, api_key: str) -> girder_client.GirderClient:
    """Connect to a DSA server.

    Parameters
    ----------
    api_url : string
        URL to the API endpoint.

    api_key : string
        API key for the DSA server.

    Returns
    -------
    gc : girder_client.GirderClient
        An authenticated Girder client object.
    """
    gc = girder_client.GirderClient(apiUrl=api_url)
    gc.authenticate(apiKey=api_key)
    return gc


def get_collection_id(collection_name: str,
                      conn: girder_client.GirderClient) -> str:
    """Given a connection, grab the id of the target collection."""
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
                  search_limit: int = 100) -> str:
    """Given a folder name and connection, return the folder ID number."""
    folder_id = None

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
    
    print(f'WARNING: Did not find folder path {folder_path} on the server.')
    return None


def get_sample_id(conn, sample_name, folder_path):
    """Given a connection, collection & folder combo, and sample name, return the sample ID number.

    We assume that there are no nested folders -- it goes collection /
    folder_list / item_list
    """
    id_list, item_list = ids_names_from_htk(conn, folder_path)

    for (id, item) in zip(id_list, item_list):
        if item == sample_name:
            return id
    print(f'Did not find sample {sample_name} in {folder_path}. Please recheck your access and spelling of the path.')
    return None


def image_metadata(conn, sample_id):
    """Return the image metadata given sample ID."""
    resp = conn.get(f'/item/{sample_id}/tiles')
    return resp


def ids_names_from_htk(conn, folder_path):
    """Get all item IDs and names in a folder."""
    
    folder_id = get_folder_id(conn, folder_path)
    
    item_list_htk = conn.listItem(folder_id)

    # Parse the retrieved item list
    item_ids = []
    item_names = []
    for item in item_list_htk:
        item_ids.append(item['_id'])
        item_names.append(item['name'])

    return item_ids, item_names


def slide_annotations(conn, slide_id, target_mpp, log=None, group_list=None):
    """Return a single slide's annotations.

    Accepts an optional annotation_list parameter indicating the groups to get.
    If not provided, grab everything.
    """
    # Case-insensitive group list!
    if group_list is not None:
        gl = [x.lower() for x in group_list]

    # Pull down the annotation objects
    try:
        annotations_resp = conn.get('/annotation/item/' + slide_id)
    except girder_client.HttpError:
        # The server couldn't find the object
        if log is not None:
            log.warning(f'Could not find an item on the server with id {slide_id}.')
        else:
            print(f'Could not find an item on the server with id {slide_id}.')
        return None, None, None
    except Exception:
        # Unclear why this failure happened
        if log is not None:
            log.warning(f'Caught exception {Exception} while getting annotations for {slide_id}.')
        else:
            print(f'Something went wrong getting annotations for {slide_id}.')
        return None, None, None

    # Do they exist? If not, return false
    if len(annotations_resp) == 0:
        if log is not None:
            log.warning(f'No annotations were found for {slide_id}.')
        else:
            print(f'No annotations were found for {slide_id}.')

    #    return None, None, None

    # Get the scale factor and string for this slide
    scale_factor, appendStr = get_scale_factor_and_appendStr(conn,
                                                             slide_id,
                                                             MPP=float(target_mpp),
                                                             MAG=None)

    # Get the info of the elements based on the now-scaled annotations
    element_infos = get_bboxes_from_slide_annotations(annotations_resp)
    if group_list is not None:
        element_infos = element_infos[element_infos['group'].isin(gl)]

    return element_infos, scale_factor, appendStr


def slide_elements(conn, item_id, target_mpp=None, group_list=None):
    """Retrieve a list of elements from the HTK annotation response.

    Each element in this list corresponds to a polygon.
    Optionally, ask for elements that belong to a specific group or list of groups.
    """
    # Pull down the annotation objects
    annotations_resp = conn.get('annotation/item/' + item_id)
    img_metadata = conn.get('item/' + item_id + '/tiles')
    # img_name = conn.get('item/' + item_id)['name']

    if len(annotations_resp) == 0:
        print('The annotation response had a length of Zero')
        return [], []

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


def image_data(conn, sample_id, bounds_dict, appendStr=None):
    """Return a numpy image defined by the connection, sample_id, and ROI."""
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
        img_roi = get_image_from_htk_response(resp)
    except Exception:
        print(f'{Exception}: Could not convert image response {resp} to image')
        return None

    return img_roi



