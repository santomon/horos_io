import pytest

import horos_io._utils as _utils


@pytest.mark.parametrize("f", ["horos_io.py", "cli.py", "_utils.py"])
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
