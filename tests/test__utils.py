import os.path

import pytest

import horos_io._utils as _utils


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


def test__has_dicom(horos_test_seq_paths):
    for horos_test_seq_path in horos_test_seq_paths:
        assert _utils._has_dicom(horos_test_seq_path)
    assert not _utils._has_dicom(".")



def test_get_ids(horos_test_root):
    for ID, target_ID in zip(_utils.get_ids(horos_test_root), ["Impression_Cmr0064", "Impression_Cmr0067"]):
        assert ID == target_ID
