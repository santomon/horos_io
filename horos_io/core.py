import os
import re
from functools import partial
from typing import Callable, Union, Dict, Tuple, List
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from horos_io import _config
from horos_io._utils import __always_true, globSSF, get_ids

Path = Union[str, os.PathLike]


def _filter_by_contour_name(contour_name: str, elem: ET.Element) -> bool:
    """just keep calm... dont worry about indexing"""
    return elem[17].text == contour_name


def _get_name_from_template(path_to_sequence: Path, template: str) -> str:
    result = re.findall(template,
                        "_".join(globSSF("*", root_dir=path_to_sequence)))
    if len(result) <= 1:
        return result[0]
    else:
        raise ValueError(f"More than 1 match fore template {template}:\n\n result: {result}")


def _load_horos_contour(contour_path: Path, n_frames: int, n_slices: int,
                        filter_: Callable[[ET.Element], bool] = __always_true) -> np.ndarray:
    """
    to really understand this file format; getting elements should really be smarter than indexing with integers
    anyway;
    simply loads the contour from the contour path;
    Args:
        contour_path: path to contour .xml file
        n_frames: number of frames of the corresponding sequence
        n_slices: number of slices of the corresponding sequence; if 1-Dim; e.g. LAX, then 1 should be passed
        filter_: takes an element at per-Image level (can have multiple ROIs; and decides on some criterion if it should be taken);
                    CAVE: will overwrite contours if you dont filter for exactly one contour per image
    Returns: ndarray in the shape of original dicoms, where each entry is either 0, if no contour, or a list of pixels (tuples)

    """
    tree = ET.parse(contour_path)
    root = tree.getroot()

    result = np.zeros((n_frames, n_slices), dtype=object)

    for elem in root[0][1]:
        # image level
        image_index = int(elem[1].text)
        f = image_index // n_slices
        s = image_index % n_slices  # if longaxis, s will result in 0

        for roi in elem[5]:
            # multiple contours can exist for every image
            if filter_(roi):
                result[f, s] = [eval(point.text) for point in roi[23]]
    return result if result.shape[1] > 1 else result.reshape(result.shape[0])


def _load_omega_contour(contour_path: Path, n_frames: int, n_slices: int) -> Dict[str, np.ndarray]:
    """
    Args:
        contour_path:
        n_frames:
        n_slices:

    Returns:
    """
    # any contour should be named <Path>/omega_{slice_type}.xml
    omega = os.path.split(contour_path)[1].split(".")[0]
    return {contour_name: _load_horos_contour(contour_path, n_frames, n_slices,
                                              filter_=partial(_filter_by_contour_name, contour_name))
            for contour_name in getattr(_config, f"{omega}_names")}


def get_n_frames_from_seq_path(path_to_sequence: Path) -> int:
    """
    get the number of frames from the sequence;
    CAVE: asssumes, that there are only files of the form
        IM-XXXX-XXXX.dcm or IM-XXXX-XXXX-XXXX.dcm inside the folder;
        any other form of the data might silently return wrong results
    Args:
        path_to_sequence:
    Returns:
    """
    # thankfully, we only need to do one look ahead
    files = globSSF("*.dcm", root_dir=path_to_sequence)
    if len(files) == 0:
        raise ValueError(f"no .dcm files found in {path_to_sequence}; current dir is {os.getcwd()}")
    s = re.compile(r"(?<=IM-\d{4}-)\d{4}")
    frame_idxs = s.findall("_".join(files))
    return max([int(frame_idx) for frame_idx in frame_idxs])


def get_n_slices_from_seq_path(path_to_sequence: Path) -> int:
    """
    get the number of slices from the sequence;
    CAVE: asssumes, that there are only files of the form
        IM-XXXX-XXXX.dcm or IM-XXXX-XXXX-XXXX.dcm inside the folder;
        any other form of the data might silently return wrong results
    Args:
        path_to_sequence:
    Returns:

    """
    files = globSSF("*.dcm", root_dir=path_to_sequence)
    s = re.compile(r"IM-\d{4}-\d{4}.dcm")
    if s.match(files[0]):
        # we have long axis images,
        return 1
    else:
        s = re.compile(r"(?<=IM-\d{4}-\d{4}-)\d{4}(?=.dcm)")
        slice_idxs = s.findall("_".join(files))
        return max([int(slice_idx) for slice_idx in slice_idxs])


def load_horos_contour(path_to_contour: Path, sequence: Union[np.ndarray, tuple, Path]) -> Union[
    np.ndarray, Dict[str, np.ndarray]]:
    """
    Args:
        path_to_contour: path should be a contour .xml file
        sequence: sequence can be:
                            - a sequence loaded by load_cine_sequence
                            - a path to a sequence that could be loaded by load_cine_sequence
                            - a tuple (n_frames, n_slices)
                  this is needed in order to infer, what the image_index from the .xml mean
    Returns: an ndarray of either shape (n_frames,) or (n_frames, n_slices)
             depending if it was lax or short axis;
             the entries are 0, if no contour, an instance of List[Tuple[float, flaot]], if it exists,
             to get a mask of existing contours, simply do:
             result != 0

             if an omega contour; will instead return a dict[contournames, contour]
    """
    if len(sequence) == 2:
        n_frames, n_slices = sequence
    elif isinstance(sequence, np.ndarray):
        n_frames, n_slices, *rest = sequence.shape
    else:
        # has to be a path or sth; if not, this should automatically result in an error
        n_frames = get_n_frames_from_seq_path(sequence)
        n_slices = get_n_slices_from_seq_path(sequence)
    if "omega" not in path_to_contour:
        return _load_horos_contour(path_to_contour, n_frames, n_slices)
    else:
        return _load_omega_contour(path_to_contour, n_frames, n_slices)


def load_horos_contour_unstructured(contour_path: Path) -> List[Tuple[int, str, List[Tuple[float, float]]]]:
    """
    sometimes it might be easier to not rely on prior structural information to construct an ndarray
    this function simply loads all existing contours in a given .xml file

    to be seen, if there should be more returned
    Args:
        contour_path: path to the contour .xml file
    Returns:
        a list of all existing contours in a triple (ImageIndex, ROIName, Contour)
    """
    tree = ET.parse(contour_path)
    root = tree.getroot()

    result = []
    for elem in root[0][1]:
        # image level
        image_index = int(elem[1].text)
        for roi in elem[5]:
            contour_name = roi[17].text
            contour = [eval(point.text) for point in roi[23]]
            result.append((image_index, contour_name, contour))
    return result



def get_contour_info_by_type(root: Path, contour_type: str) -> pd.DataFrame:
    """
    given a Horos root and a contour type; returns a dataframe with information
    on given contour in that dataset

    (
    "ID": folder name, where contours reside,
     "contour_path": relative path to the contour file, starting from root of the data,
     "contour_type": one of config.contour_types,
     "location": joins the root dir with the sequence path
     "slice_type": one of config.slice_types,
    )
    Args:
        root: Horos dataset root
        contour_type: a string; will search for <contour_type>.xml in ID directories
    Returns:
    """
    result = pd.DataFrame()
    result["ID"] = get_ids(root)
    result["contour_type"] = contour_type

    result["contour_path"] = result["ID"].apply(
        lambda id_: os.path.join(id_, f"{contour_type}.xml")
        if os.path.isfile(os.path.join(root, id_, f"{contour_type}.xml"))
        else ""
    )
    result["contour_path"].replace("", pd.NA, inplace=True)

    result["location"] = result["contour_path"].apply(
        lambda path: os.path.normpath(os.path.join(root, path)) if pd.notna(path) else ""
    )
    result["location"].replace("", pd.NA, inplace=True)
    result = result[["ID", "contour_type", "contour_path", "location"]]
    return result


def existing_contours_within(group: pd.DataFrame) -> pd.DataFrame:
    """
    TODO: finish this
    takes a group from get_combined_info, that has been properly grouped, such that there is only one row left;
    loads contours and creates a new df, with additional entry for frame and slice, where contours exist;

    if a contour object with multiple contours, will raise an error if there is any discrepancy
    Args:
        group:

    Returns:
    """
    pass

