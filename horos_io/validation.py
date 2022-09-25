import os
import time
from datetime import datetime
from functools import reduce
from typing import Optional, Union, Dict

import numpy as np
import pandas as pd

import horos_io._legacy
import horos_io.core
from horos_io.cmr import Path
from horos_io._config import time_format


def last_validation_was_successful(log: pd.DataFrame, conf_field="result", **criteria):
    """
    Args:
        conf_field: name of the field in log_, from which to draw the confirmation
        **criteria: k: column name in log_; v: specific entry to filter by; either a specific value or a callable that is the applied to the respective series
        log:

    Returns:
        bool: True, if the last entry of this combination in log_ was successful
    """
    eligible = reduce(lambda x, y: x & y,
                      [log[k].apply(v) if callable(v) else log[k] == v for k, v in criteria.items()])
    in_question = log[eligible]
    if in_question.shape[0] == 0:
        return False
    else:
        # take the latest
        latest_idx = in_question["time_stamp"].apply(lambda st: datetime.strptime(st, time_format)).idxmax()
        return in_question.loc[latest_idx, conf_field]


def last_validation(log: pd.DataFrame, **criteria) -> pd.Series:
    """
    Args:
        **criteria: k: column name in log_; v: specific entry to filter by; either a specific value or a callable that is the applied to the respective series
        log:

    Returns:
        bool: True, if the last entry of this combination in log_ was successful
    """
    eligible = reduce(lambda x, y: x & y,
                      [log[k].apply(v) if callable(v) else log[k] == v for k, v in criteria.items()])
    in_question = log[eligible]
    if in_question.shape[0] == 0:
        return None
    else:
        # take the latest
        latest_idx = in_question["time_stamp"].apply(lambda st: datetime.strptime(st, time_format)).idxmax()
        return in_question.loc[latest_idx,]


def visually_confirm_omega_iter(combined_info: pd.DataFrame):
    """
    CAVE: should only be used for parametrization for the following test
    TODO: as more contour types roll out; we should dynamically parametrize this from passed arguments
    TODO: turn this into a fixture
    Args:
        combined_info:
    Returns:
    """
    combined_info = combined_info[combined_info["contour_type"].apply(lambda cont: "omega" in cont)]
    for i, row in combined_info.iterrows():
        cines = horos_io.load_cine_sequence(row["location_images"])
        contours = horos_io.core.load_horos_contour(row["location_contour"], row["location_images"])
        mt_contour = time.localtime(os.stat(row["location_contour"]).st_mtime)

        if len(cines.shape) == 1:
            cines = cines[:, np.newaxis]
            contours = {cn: contour[:, np.newaxis] for cn, contour in contours.items()}
        for f, s in zip(*np.where(list(contours.values())[0] != 0)):
            yield row["ID"], row["contour_type"], f, s, cines, contours, mt_contour


def changed_since_last_validation(mt_contour, log_row: pd.Series) -> bool:
    """TODO: check, if this method even works correctly"""
    if log_row is None:
        return True
    last_val = time.strptime(log_row["time_stamp"], time_format)
    return mt_contour > last_val


def write_log(log_: Path, **kwargs):
    if log_ is None:
        log_ = os.path.join(kwargs["root"], "val_contour_log.csv")
    if os.path.isfile(log_):
        df = pd.read_csv(log_, index_col=0)
    else:
        if not os.path.isdir(os.path.split(log_)[0]):
            os.makedirs(os.path.split(log_)[0])
        df = pd.DataFrame(columns=["ID", "contour_type", "result", "by", "frame", "slice", "remark", "time_stamp"])
    df = df.append(pd.Series({
        "ID": kwargs["ID"],
        "contour_type": kwargs["omega"],
        "result": kwargs["ok"],
        "by": kwargs["by"],
        "frame": kwargs["f"],
        "slice": kwargs["s"],
        "remark": kwargs["remark"],
        "time_stamp": datetime.now().strftime(time_format)
    }), ignore_index=True)
    df.to_csv(log_)


def all_more_than_3_points(contour: np.ndarray) -> bool:
    """any contour should have more than 3 points for meaningful spline interpolation; (even though we only need 3 theoretically)"""
    return all([len(C) > 3 for C in contour.flatten()[np.where(contour.flatten() != 0)]])


def contour_is_valid(contour: Union[Dict[str, np.ndarray], np.ndarray], n: Optional[int] = None) -> bool:
    """this method kinda sucks bc it doesnt tell us which and where"""
    if isinstance(contour, dict):
        return reduce(lambda x, y: x and y, [contour_is_valid(value, n) for value in contour.values()])
    else:
        at_least_n = (contour != 0).sum() >= n if n is not None else True
        return at_least_n and all_more_than_3_points(contour)
