import pytest

from horos_io import validation


@pytest.mark.parametrize("crit, target", [({"ID": "Impression_Cmr0010", "frame": 10, "slice": 0, "contour_type": "omega_4ch"}, True),
                                          ({"ID": "Impression_Cmr0210", "frame": 10, "slice": 0, "contour_type": "omega_4ch"}, False)])
def test_last_validation_was_successful2(log_dummy, crit, target):
    assert validation.last_validation_was_successful(log_dummy, conf_field="result", **crit) == target

