import os.path

import pandas as pd
import pytest


@pytest.fixture()
# if testing horos; with our data, change this path to where your data is located
def horos_test_root():
    return os.path.normpath("./tests/horos_dummy")

SEQ = ['Impression_Cmr0064/Anonymous_Study - 0/CINE_segmented_LAX_2Ch_7/',
                        'Impression_Cmr0064/Anonymous_Study - 0/CINE_segmented_LAX_3Ch_5/',
                        'Impression_Cmr0064/Anonymous_Study - 0/CINE_segmented_LAX_4Ch_6/',
                        'Impression_Cmr0064/Anonymous_Study - 0/CINE_segmented_SAX_28/',
                        'Impression_Cmr0067/Anonymous_Study - 0/CINE_segmented_LAX_2Ch_7/',
                        'Impression_Cmr0067/Anonymous_Study - 0/CINE_segmented_LAX_3Ch_9/',
                        'Impression_Cmr0067/Anonymous_Study - 0/CINE_segmented_LAX_4Ch_10/',
                        'Impression_Cmr0067/Anonymous_Study - 0/CINE_segmented_SAX_53/']
N_SLICES = [1, 1, 1, 13, 1, 1, 1, 13]

@pytest.fixture(params=SEQ)
def horos_test_seq_path(horos_test_root, request):
    return os.path.normpath(os.path.join(horos_test_root, request.param))


@pytest.fixture(params=N_SLICES)
def horos_test_seq_n_slices(request):
    return request.param


@pytest.fixture()
def horos_basal_first_file(horos_test_root):
    return os.path.join(horos_test_root, "basal_info.csv")


@pytest.fixture()
def horos_contour_info_path(horos_test_root):
    return os.path.join(horos_test_root, "target_contour_info.csv")


@pytest.fixture()
def horos_image_info_path(horos_test_root):
    return os.path.join(horos_test_root, "target_image_info.csv")


@pytest.fixture()
def log_dummy():
    return pd.read_csv("./tests/visual_confirmation.csv", index_col=0)

# def pytest_generate_tests(metafunc: pytest.Metafunc):
#
#     if "horos_test_seq_path" in metafunc.fixturenames and "horos_test_seq_n_slices" in metafunc.fixturenames:
#         metafunc.parametrize("horos_test_seq_path, horos_test_seq_n_slices", zip(SEQ, N_SLICES))


def pytest_addoption(parser):
    parser.addoption("--pth")
