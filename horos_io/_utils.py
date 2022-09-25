import glob
import os.path
import pathlib
from typing import List, Dict, Union, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pydicom
from decorator import decorator
from pydicom.errors import InvalidDicomError
from scipy import interpolate

Path = Union[str, os.PathLike]


def __always_true(*args, **kwargs) -> bool: return True


@decorator
def handle_1D(f, *args, **kwargs):
    def _handle_1D(item):
        if isinstance(item, np.ndarray):
            if len(item.shape) < 2:
                return item[:, np.newaxis]
            else:
                return item
        elif isinstance(item, dict):
            return {k: _handle_1D(v) for k, v in item.items()}
        elif isinstance(item, tuple):
            return tuple([_handle_1D(v) for v in item])
        else:
            return item

    return f(*_handle_1D(args), **_handle_1D(kwargs))


def _to_str(i):
    if not 99 >= i >= 0 or not round(i) == i:
        raise ValueError(f"i should be 2 digits, but instead it is {i}")
    return str(i) if i >= 10 else f"0{i}"


def get_log(log: str, root: str):
    if log is None:
        log = os.path.join(root, "val_contour_log.csv")
    if os.path.isfile(log):
        df = pd.read_csv(log, index_col=0)
    else:
        df = pd.DataFrame(columns=["ID", "contour_type", "result", "by", "frame", "slice", "remark", "time_stamp"])
    return df


def _has_dicom(pth: Path) -> bool:
    for f in glob.glob(os.path.join(pth, "*")):
        try:
            pydicom.dcmread(f)
            return True
        except InvalidDicomError:
            continue
        except PermissionError:
            continue
    return False


def get_seq_paths(root: Path) -> List[str]:
    """
    checks for sequences, which are assumed to  be 3 levels into the root;

    Args:
        root: path to the Horos Dataset
    Returns:
        list of sequence names
    """
    return [pth for pth in globSSF("*/*/*/", root_dir=root) if _has_dicom(os.path.join(root, pth))]


def get_ids(root: Path):
    """
    gets a list of IDs; not as trivial, as there can be other folders, too;
    checks for sequences, which are assumed to  be 3 levels into the root;
    splits the first directory of it and returns non duplicates of it
    Args:
        root: path to the Horos Dataset
    Returns:
        list of IDs
    """
    seqs = get_seq_paths(root)
    result = [pathlib.Path(seq).parts[0] for seq in seqs]
    return sorted(list(set(result)))


def get_log_path(log: str, root: str):
    if log is None:
        log = os.path.join(root, "val_contour_log.csv")
    return log


def globSSF(path_name, /, root_dir, **kwargs) -> List[str]:
    """mimcks behaviour of future globs"""
    return [os.path.relpath(pth, root_dir) for pth in
            glob.glob(os.path.normpath(os.path.join(root_dir, path_name)), **kwargs)]


####
def mask_from_omega_contour(cines: np.ndarray, contours: Dict[str, np.ndarray], loc: Union[int, Tuple]) -> np.ndarray:
    # TODO: raises error when passing 1D sequences
    # simply havin the user pass the location as tuple or int should be much more flexible
    mask = np.zeros_like(cines[loc].pixel_array)
    for i, (name, c) in enumerate(contours.items()):
        tck, _ = interpolate.splprep([*zip(*c[loc])], s=0, per=True)
        xnew, ynew = interpolate.splev(np.linspace(0, 1, 500), tck, der=0)
        smooth_contour = np.array([(round(px), round(py)) for px, py in zip(xnew, ynew)])
        mask = cv2.drawContours(mask, [smooth_contour], 0, i + 1, - 1)
    return mask


