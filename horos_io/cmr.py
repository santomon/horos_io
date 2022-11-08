"""
module for working with horos images and contours. the dataformat is as follows:
the names for anonymous study and the sequences are inconsistent

ROOT |--- ID0
                 |---------   Anonymous_Study
                                    |------------ <LAX>
                                                    |----------- IM-0001-0001.dcm
                                                    |----------- IM-0001-0002.dcm
                                                    ...
                                    |------------ <SAX>
                                                    |----------- IM-0001-0001-0001.dcm
                                                    |----------- IM-0001-0001-0002.dcm
                                                    ...
                 |--- sax_lv_endo.xml
                 |--- sax_lv_epi.xml
                 ...
    |--- ID1
                 |--- ...
                 ...
    ...

not that it really matters;
for convenience, every location of contours and image seqs, will
    be stored in a .csv

CAVE: currently does not respect the fact, that we could have non-CINEs in our data
"""
import functools
import os
import pathlib
from typing import Union

import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

from horos_io import _config
from horos_io.core import get_n_frames_from_seq_path, \
    get_n_slices_from_seq_path, _get_name_from_template, get_contour_info_by_type
from ._utils import _to_str, get_seq_paths, get_study_date

Path = Union[os.PathLike, str]


def load_cine_sequence(path_to_sequence: Path) -> np.ndarray:
    """
    convenience function to let the algorithm decide, if it is lax or sax;

    a problem is, is that sequences can be repeated multiple times,
    the first code in IM-XXXX-0001.dcm or IM-XXXX-0001-0001.dcm denotes this;

    in our data we have only 1 sequence of however many repetitions there were
    Args:
        path_to_sequence:
    Returns:
    """
    p = path_to_sequence
    return load_lax_sequence(p) if get_n_slices_from_seq_path(p) == 1 else load_sax_sequence(p)


def load_sax_sequence(path_to_sequence: Path) -> np.ndarray:
    """

        for slices, they are sorted by the Y-component of the ImagePositionPatient Metadata of each slice;
        in theory, that value increases the more dorsal we go with respect to the patient.
        Ideally, the Apex should be the most ventral in the patient (inverted heart anatomy? scanning patient face down?)
    Args:
        path_to_sequence:
    Returns:
        a numpy array, where the contents of the array are pydicom.FileDataset of the images of shape
    (n_frames, n_slices)
    """
    n_frames = get_n_frames_from_seq_path(path_to_sequence)
    n_slices = get_n_slices_from_seq_path(path_to_sequence)

    result = np.array([[
        pydicom.dcmread(os.path.join(path_to_sequence,
                                     _get_name_from_template(path_to_sequence,
                                                             fr"IM-\d\d\d\d-00{_to_str(f + 1)}-00{_to_str(s + 1)}.dcm")))
        for s in range(n_slices)]
        for f in range(n_frames)])
    result = sort_SAX_by_y(result)
    return result


def load_lax_sequence(path_to_sequence: Path) -> np.ndarray:
    """
    returns a numpy array, where the contents of the array are pydicom.FileDataset of the images of shape
    (n_frames,)
    the indices correspond to the frame in the video
    Args:
        path_to_sequence:
    Returns:  np-array of the long axis images
    """
    n_frames = get_n_frames_from_seq_path(path_to_sequence)

    return np.array([
        pydicom.dcmread(os.path.join(path_to_sequence,
                                     _get_name_from_template(path_to_sequence, fr"IM-\d\d\d\d-00{_to_str(f + 1)}.dcm")))
        for f in range(n_frames)])


def _get_slice_type(path: Path) -> str:
    if "2ch" in str(path).lower():
        return "cine_2ch"
    elif "3ch" in str(path).lower():
        return "cine_3ch"
    elif "4ch" in str(path).lower():
        return "cine_4ch"
    elif "sax" in str(path).lower() or "_sa" in str(path).lower():
        return "cine_sa"
    else:
        raise ValueError(f"unknown slice type with path: {path}")


def sort_SAX_by_y(X: np.ndarray) -> np.ndarray:
    """
    Args:
        X: a 2D short axis stack;
        if dicom innolitics can be believed, we can assume that the Y-world axis increased the more dorsal
        you go with respect to the patient. therefore, in a SAX stack, the most apical slice should have the smallest y;
        we would be able to use this to sort the stack without relying on manually inputting the order;

        since we our SAX stack is of shape (n_frames, n_slices); need to transpose and retranspose
    """
    return np.array(
        sorted(X.T, key=lambda x: x[0].ImagePositionPatient[1])
    ).T


@functools.lru_cache()
def get_image_info(root: Path) -> pd.DataFrame:
    """
    checks the root for any Horos Exported Sequences and returns a DataFrame with the information;
    should be in core; but since we currently also save the slice_type; which is cmr specific;
    it lies here for now...

    columns:
    ("seq_path",  <relative path starting from the root of the data>
     "location",  joins the rootdir with seq path
     "ID",        <Impression_Cmr{ID}>
     "study_data", well, study date
     "slice_type", one of config.slice_types
     "n_frames",
     "n_slices",

    sequence can be loaded with load_cine_sequence
    Args:
        root: root of the data; contents should be directories of Studies
    Returns:
        a DataFrame with the above information
    """
    result = pd.DataFrame()

    result["seq_path"] = get_seq_paths(root)

    result["ID"] = result["seq_path"].apply(lambda loc: pathlib.Path(loc).parts[0])  # ID is name of the first folder
    result["study_date"] = result["ID"].apply(functools.partial(get_study_date, root=root))
    result["location"] = [os.path.normpath(os.path.join(root, seq_path))
                          for seq_path in get_seq_paths(root)]
    result["slice_type"] = result["location"].apply(_get_slice_type)
    result["n_frames"] = result["location"].apply(get_n_frames_from_seq_path)
    result["n_slices"] = result["location"].apply(get_n_slices_from_seq_path)

    # CAVE: why did we do this
    result = result[["ID", "study_date", "slice_type", "n_frames", "n_slices", "seq_path", "location"]]

    if result.shape[0] == 0:
        raise ValueError(f"could not find any images in {root}")

    return result.sort_values(by=["ID", "slice_type"])


@functools.lru_cache()
def get_contour_info(root: Path) -> pd.DataFrame:
    """
    dataframe columns:
    ("ID": ,
     "contour_path": relative path to the contour file, starting from root of the data,
     "contour_type": one of config.contour_types,
     "location": joins the root dir with the sequence path
     "slice_type": one of config.slice_types,
    )

    creates a csv file containing location of all contour files; or whether they exist at all.
    Args:
        root: root of the data; contents should be directories of Impression Studies
    Returns:
    """
    result = pd.concat([get_contour_info_by_type(root, contour_type)
                        for contour_type in tqdm(_config.contour_types)], ignore_index=True)
    result["slice_type"] = result["contour_type"].apply(lambda x: _get_slice_type(x))

    if result.shape[0] == 0:
        raise ValueError(f"could not find any contours in {root}")

    return result.sort_values(by=["ID", "slice_type", "contour_type"])


@functools.lru_cache()
def get_combined_info(root: Path) -> pd.DataFrame:
    """
    Args:
        root: Horos Dataset root
    Returns:
        DataFrame, which contains the combined info you get from get_contour_info and get_image_info
        by merging with ID and slice_type
    """
    image_info = get_image_info(root)
    contour_info = get_contour_info(root)
    return pd.merge(contour_info[contour_info["location"].notna()],
                    image_info, on=["ID", "slice_type"], suffixes=("_contour", "_images"))


if __name__ == '__main__':
    pass
