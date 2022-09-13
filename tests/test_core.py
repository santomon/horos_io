from horos_io import core


def test_load_horos_contour_unstructured():
    """might need to extend this"""
    pth = "tests/omega_4ch.xml"
    result = core.load_horos_contour_unstructured(pth)
    assert len(result) == 10
