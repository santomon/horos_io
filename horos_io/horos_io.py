"""
module for working with horos images and contours. the dataformat is as follows:
the names for anonymous study and the sequences are inconsistent

ROOT |--- Impression0001
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
    |--- Impression0002
                 |--- ...
                 ...
    ...

not that it really matters;
for convenience, every location of contours and image seqs, will
    be stored in a .csv

useful function call:
combined_info = pd.merge(contour_info[contour_info["location"].notna()],
                        image_info, on=["ID", "slice_type"], suffixes=("_contour", "_images"))
DEFINITELY MERGE ON ["ID", "slice_type"] to get correct correspondence of images and contours

CAVE: short axis stacks have a value for "basal_first", which needs to be passed
to the reading function to get the correct order of images

CAVE: currently does not respect the fact, that we could have non-CINEs in our data
"""
import os
import pathlib
import re
import xml.etree.ElementTree as ET
from functools import partial
from typing import Union, Callable, Dict, Optional, NoReturn, List

import numpy as np
import pandas as pd
import pydicom

from . import config
from ._utils import __always_true, globSSF, _to_str

Path = Union[os.PathLike, str]


def load_sequence(path_to_sequence: Path, basal_first: bool) -> np.ndarray:
    """
    convenience function to let the algorithm decide, if it is lax or sax;
    will need to pass basal_first though, just in case...

    a problem is, is that sequences can be repeated multiple times,
    the first code in IM-XXXX-0001.dcm or IM-XXXX-0001-0001.dcm denotes this;

    in our data we have only 1 sequence of however many repetitions there were
    Args:
        path_to_sequence:
        basal_first:
    Returns:
    """
    p = path_to_sequence
    return load_lax_sequence(p) if _get_n_slices_from_seq_path(p) == 1 else load_sax_sequence(p, basal_first)


def load_lax_sequence(path_to_sequence: Path) -> np.ndarray:
    """
    returns a numpy array, where the contents of the array are pydicom.FileDataset of the images of shape
    (n_frames,)
    the indices correspond to the frame in the vidoe
    Args:
        path_to_sequence:
    Returns:  np-array of the long axis images
    """
    n_frames = _get_n_frames_from_seq_path(path_to_sequence)

    return np.array([
        pydicom.dcmread(os.path.join(path_to_sequence,
                                     _get_name_from_template(path_to_sequence, fr"IM-\d\d\d\d-00{_to_str(f + 1)}.dcm")))
        for f in range(n_frames)])


def load_sax_sequence(path_to_sequence: Path, basal_first: bool) -> np.ndarray:
    """
        returns a numpy array, where the contents of the array are pydicom.FileDataset of the images of shape
    (n_frames, n_slices)
    Args:
        path_to_sequence:
        basal_first: if True, will invert the slice order
    Returns:
    """
    n_frames = _get_n_frames_from_seq_path(path_to_sequence)
    n_slices = _get_n_slices_from_seq_path(path_to_sequence)
    ordering = -1 if basal_first else 1

    return np.array([[
        pydicom.dcmread(os.path.join(path_to_sequence,
                                     _get_name_from_template(path_to_sequence,
                                                             fr"IM-\d\d\d\d-00{_to_str(f + 1)}-00{_to_str(s + 1)}.dcm")))
        for s in range(n_slices)[::ordering]]
        for f in range(n_frames)])


def load_horos_contour(path_to_contour: Path, sequence: Union[np.ndarray, tuple, Path]) -> Union[
    np.ndarray, Dict[str, np.ndarray]]:
    """
    Args:
        path_to_contour: path should be a contour .xml file
        sequence: sequence can be:
                            - a sequence loaded by load_sequence
                            - a path to a sequence that could be loaded by load_sequence
                            - a tuple (n_frames, n_slices)
                  this is needed in order to infer, what the image_index from the .xml mean
    Returns: an ndarray of either shape (n_frames,) or (n_frames, n_slices)
             depending if it was lax or short axis;
             the entries are 0, if no contour, an instance of List[Tuple[float, flaot]], if it exists,
             to get a mask of existing contours, simply do:
             result != 0
    """
    if len(sequence) == 2:
        n_frames, n_slices = sequence
    elif isinstance(sequence, np.ndarray):
        n_frames, n_slices, *rest = sequence.shape
    else:
        # has to be a path or sth; if not, this should automatically result in an error
        n_frames = _get_n_frames_from_seq_path(sequence)
        n_slices = _get_n_slices_from_seq_path(sequence)
    if "omega" not in path_to_contour:
        return _load_horos_contour(path_to_contour, n_frames, n_slices)
    else:
        return load_omega_contour(path_to_contour, n_frames, n_slices)


def _make_image_info_csv(root: Path, basal_info_file: Optional[str] = None,
                         out: Optional[Path] = None) -> NoReturn:
    """
    one-time operation to update the information of the dataset;

    basal_info_file is but a csv document, that contains, if the sax stack has basal slice first
    in the sequence. is needed for having consistent ordering across sequences

    columns:
    ("seq_path",  <relative path starting from the root of the data>
     "location",  <relative path starting from the source root..., actually, just ignore this>
     "ID",        <Impression_Cmr{ID}>
     "slice_type", one of config.slice_types
     "n_frames",
     "n_slices",
     "basal_first", one of NA, True or False; only relevant for loading SAX in correct order)

    seq_path is relative to the root
    location is path relative to the source_root (ventricular-function/container/files)

    creates a csv file containing location of the sequence.
    sequence can be loaded with load_sequence
    Args:
        root: root of the data; contents should be directories of Impression Studies
        out: optional parameter to specify, what and where to save
    Returns:
    """

    if basal_info_file is None:
        basal_info_file = os.path.join(root, "basal_info.csv")

    basal_firsts = _load_basaL_first_as_list(basal_info_file)
    result = pd.DataFrame()

    result["seq_path"] = globSSF("*/*/*/", root_dir=root)
    result["ID"] = result["seq_path"].apply(lambda loc: pathlib.Path(loc).parts[0])  # ID is name of the first folder
    result["location"] = [os.path.normpath(os.path.join(root, seq_path))
                          for seq_path in globSSF("*/*/*/", root_dir=root)]
    result["slice_type"] = result["location"].apply(_get_slice_type)
    result["n_frames"] = result["location"].apply(_get_n_frames_from_seq_path)
    result["n_slices"] = result["location"].apply(_get_n_slices_from_seq_path)

    # true if sax and contained in basal_first
    # NaN if lax
    result["basal_first"] = result.apply(lambda row: re.findall(r"\d{4}", row["ID"])[0] in basal_firsts
    if row["slice_type"] == "cine_sa" else pd.NA, axis=1)

    if out is None:
        out = os.path.normpath(os.path.join(root, "image_info.csv"))
    result.sort_values(by="ID").to_csv(out)


def _make_contour_info_csv(root: Path, out: Optional[Path] = None) -> NoReturn:
    """
    one-time operation to update the information of the dataset;

    ("ID": <Impression_Cmr{ID}>,
     "contour_path": relative path to the contour file, starting from root of the data,
     "contour_type": one of config.contour_types,
     "location": relative path to the contour file, starting from source [IGNORE],
     "slice_type": one of config.slice_types,
     "c
    )

    creates a csv file containing location of all contour files; or whether they exist at all.
    Args:
        root: root of the data; contents should be directories of Impression Studies
    Returns:
    """

    result = pd.concat([_make_contour_info_csv_by_contour_type(root, contour_type)
                        for contour_type in config.contour_types], ignore_index=True)

    result["location"] = result["contour_path"].apply(
        lambda path: os.path.normpath(os.path.join(root, path)) if pd.notna(path) else ""
    )
    result["location"].replace("", pd.NA, inplace=True)
    if out is None:
        out = os.path.normpath(os.path.join(root, "contour_info.csv"))
    result.sort_values(by="ID").to_csv(out)


def _make_contour_info_csv_by_contour_type(root: Path, contour_type: str) -> pd.DataFrame:
    result = pd.DataFrame()
    result["ID"] = globSSF(f"Impression*", root_dir=root)
    result["contour_path"] = result["ID"].apply(
        lambda id_: os.path.join(id_, f"{contour_type}.xml")
        if os.path.isfile(os.path.join(root, id_, f"{contour_type}.xml"))
        else ""
    )
    result["slice_type"] = result["contour_path"].apply(lambda x: _get_slice_type(contour_type))

    result["contour_path"].replace("", pd.NA, inplace=True)
    result["contour_type"] = contour_type
    return result


def _get_n_slices_from_seq_path(path_to_sequence: Path) -> int:
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


def _get_n_frames_from_seq_path(path_to_sequence: Path) -> int:
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
    s = re.compile(r"(?<=IM-\d{4}-)\d{4}")
    frame_idxs = s.findall("_".join(files))
    return max([int(frame_idx) for frame_idx in frame_idxs])


def _get_slice_type(path: Path) -> str:
    if "2ch" in str(path).lower():
        return "cine_2ch"
    elif "3ch" in str(path).lower():
        return "cine_3ch"
    elif "4ch" in str(path).lower():
        return "cine_4ch"
    elif "sax" in str(path).lower():
        return "cine_sa"
    else:
        raise ValueError(f"unknown slice type with path: {path}")


def _load_basaL_first_as_list(basal_first_file: Path) -> List[str]:
    basal_df = pd.read_csv(basal_first_file, dtype=str)
    return list(basal_df["basal_first"])


def _get_name_from_template(path_to_sequence: Path, template: str) -> str:
    result = re.findall(template,
                        "_".join(globSSF("*", root_dir=path_to_sequence)))
    if len(result) <= 1:
        return result[0]
    else:
        raise ValueError(f"More than 1 match fore template {template}:\n\n {result}")


def _load_horos_contour(contour_path: Path, n_frames: int, n_slices: int,
                        filter_: Callable[[ET.Element], bool] = __always_true) -> np.ndarray:
    """
    there is no real inherent logic to this trash file format...
    i wouldnt bother trying to understand what all the indices mean...
    Args:
        contour_path:
        n_frames:
        n_slices:
        filter_: takes an element at per-Image level (can have multiple ROIs; and decides on some criterion if it should be taken

    Returns:

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


def _filter_by_contour_name(contour_name: str, elem: ET.Element) -> bool:
    """just keep calm... dont worry about indexing"""
    return elem[17].text == contour_name


def load_omega_contour(contour_path: Path, n_frames: int, n_slices: int) -> Dict[str, np.ndarray]:
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
            for contour_name in getattr(config, f"{omega}_names")}

if __name__ == '__main__':
    pass
