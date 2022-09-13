import os
from typing import List

import numpy as np
import pandas as pd
import pydicom

from horos_io import load_lax_sequence
from horos_io._utils import _to_str
from horos_io.core import get_n_slices_from_seq_path, get_n_frames_from_seq_path, _get_name_from_template

from horos_io.cmr import Path


def _load_basaL_first_as_list(basal_first_file: Path) -> List[str]:
    basal_df = pd.read_csv(basal_first_file, dtype=str)
    return list(basal_df["basal_first"])


def load_sequence(path_to_sequence: Path, basal_first: bool) -> np.ndarray:
    """
    Legacy variant, relying on user input for ordering

    convenience function to let the algorithm decide, if it is lax or sax;
    will need to pass basal_first though, just in case...
    even though

    a problem is, is that sequences can be repeated multiple times,
    the first code in IM-XXXX-0001.dcm or IM-XXXX-0001-0001.dcm denotes this;

    in our data we have only 1 sequence of however many repetitions there were
    Args:
        path_to_sequence:
        basal_first: pass a dummy boolean if you load a LAX sequence; otherwise for SAX a boolean should be passed,
        that denotes, whether in the SAX sequence lower slice number means more basal; will then reorder, such that
        Apex is first
    Returns:
    """
    p = path_to_sequence
    return load_lax_sequence(p) if get_n_slices_from_seq_path(p) == 1 else load_sax_sequence(p, basal_first)


def load_sax_sequence(path_to_sequence: Path, basal_first: bool) -> np.ndarray:
    """
        returns a numpy array, where the contents of the array are pydicom.FileDataset of the images of shape
    (n_frames, n_slices)
    Args:
        path_to_sequence:
        basal_first: if True, will invert the slice order
    Returns:
    """
    n_frames = get_n_frames_from_seq_path(path_to_sequence)
    n_slices = get_n_slices_from_seq_path(path_to_sequence)
    ordering = -1 if basal_first else 1

    return np.array([[
        pydicom.dcmread(os.path.join(path_to_sequence,
                                     _get_name_from_template(path_to_sequence,
                                                             fr"IM-\d\d\d\d-00{_to_str(f + 1)}-00{_to_str(s + 1)}.dcm")))
        for s in range(n_slices)[::ordering]]
        for f in range(n_frames)])
