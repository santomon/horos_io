import glob
import os
import os.path
import pathlib
import sys
from typing import List, Dict, Union, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pydicom
from decorator import decorator
from pydicom.errors import InvalidDicomError
from scipy import interpolate

from horos_io import _config

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


def globSSF(path_name, /, root_dir=None, **kwargs) -> List[str]:
    """mimcks behaviour of future globs"""
    if root_dir is None:
        return glob.glob(path_name, **kwargs)

    return [os.path.relpath(pth, root_dir) for pth in
            glob.glob(os.path.normpath(os.path.join(root_dir, path_name)), **kwargs)]


####
def mask_from_omega_contour(cines: np.ndarray, contours: Dict[str, np.ndarray], loc: Union[int, Tuple]) -> np.ndarray:
    # TODO: raises error when passing 1D sequences
    # simply havin the user pass the location as tuple or int should be much more flexible
    mask = np.zeros_like(cines[loc].pixel_array)
    for i, (name, c) in enumerate(contours.items()):
        if name != "aroot":
            tck, _ = interpolate.splprep([*zip(*c[loc])], s=0, per=True)
            xnew, ynew = interpolate.splev(np.linspace(0, 1, 500), tck, der=0)
            smooth_contour = np.array([(round(px), round(py)) for px, py in zip(xnew, ynew)])
            mask = cv2.drawContours(mask, [smooth_contour], 0, i + 1, - 1)
        else:
            aroot_c = interpolate_aroot(c[loc])
            mask = cv2.drawContours(mask, [np.array(aroot_c, dtype=int)], 0, i + 1, -1)
    return mask


def interpolate_aroot(landmarks=List[tuple]) -> List[tuple]:
    """
    an aortic root selection of landmarks consists of exactly 6 points; starting with 2 distal points;
    and then going clockwise;

    interpolates the lateral segments, while using linspace for the horizontal sections;

    Args:
        landmarks:

    Returns:

    """
    if not len(landmarks) == 6:
        raise ValueError(f"aroot needs to have exactly 6 landmarks; instead there are {len(landmarks)}")

    def linspace_interpolation(p1, p2, n=10):
        result = np.linspace(0, 1, n)[:, np.newaxis] * (np.array(p2) - p1) + p1
        return result

    def spline_interpolation(seq: List[tuple], n=10) -> List[tuple]:
        tck, _ = interpolate.splprep([*zip(*seq)], s=0, per=True)
        xnew, ynew = interpolate.splev(np.linspace(0, 1, n), tck, der=0)
        return [*zip(xnew, ynew)]

    dist = linspace_interpolation(landmarks[0], landmarks[1])
    inner = spline_interpolation(landmarks[1:4])
    prox = linspace_interpolation(landmarks[3], landmarks[4])
    outer = spline_interpolation(landmarks[-2:] + [landmarks[0]])
    return dist + inner + prox + outer


def masks2spline(masks: np.ndarray):
    """converts an integer mask into a list of contours"""
    return [masks == i for i in np.unique(masks)]


def mask2spline(mask, ):
    """
    Convert a binary mask to a spline.

    Args:
        mask (ndarray): Binary mask.
        epsilon (float): The epsilon value for RDP algorithm.

    Returns:
        x (ndarray): x coordinates of the spline.
        y (ndarray): y coordinates of the spline.
    """
    # Prepare the mask
    _, thrs = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU)

    # Find the contour
    contours, _ = cv2.findContours(thrs, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = sorted_contours[0]
    largest_contour = [(p[0], p[1]) for p in largest_contour]

    return largest_contour


def get_tag(tag: Tuple[str, str], ID: str, root: str) -> str:
    dicom_paths = globSSF(f"{os.path.join(root, ID, '*/*/*.dcm')}")
    return pydicom.dcmread(dicom_paths[0])[tag].value


def get_study_date(ID: str, root: str) -> str:
    return get_tag(_config.study_date_tag, ID, root)
