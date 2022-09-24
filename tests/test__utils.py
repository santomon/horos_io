import os.path

import numpy as np
import pytest

import horos_io._utils as _utils
from horos_io import load_cine_sequence, _config


@pytest.mark.parametrize("f", ["cmr.py", "cli.py", "_utils.py"])
def test_glob_ssf(f):
    paths = _utils.globSSF("*", root_dir="horos_io")
    assert f in paths


@pytest.mark.parametrize("i, i_expected", [(9, "09"), pytest.param(-1, ...),
                                           pytest.param(100, ...)])
def test__to_str(i, i_expected):
    if i_expected is not ...:
        assert _utils._to_str(i) == i_expected
    else:
        with pytest.raises(ValueError):
            _utils._to_str(i)


def test__has_dicom(horos_test_seq_path):
    assert _utils._has_dicom(horos_test_seq_path)
    assert not _utils._has_dicom(".")


def test_get_ids(horos_test_root):
    for ID, target_ID in zip(_utils.get_ids(horos_test_root), ["Impression_Cmr0064", "Impression_Cmr0067"]):
        assert ID == target_ID


def test_mask_from_omega_contour(horos_test_seq_path):
    """passing a mix of LAX and SAX sequences and testing, if shape is of loaded contours 1D vs 2D is respected"""
    # for p in horos_test_seq_path:
    p = horos_test_seq_path
    cine = load_cine_sequence(p)

    def random_contours():
        result = np.zeros_like(cine, dtype=object)
        Q = result.flatten()
        Q[0] = [(np.random.random(), np.random.random()) for _ in range(np.random.randint(5, 20))]
        result = Q.reshape(result.shape)
        return result

    omega = {cname: random_contours() for cname in _config.omega_4ch_names}

    mask = _utils.mask_from_omega_contour(cine, omega, (0, 0))

    assert mask.shape == cine.flatten()[0].pixel_array.shape

