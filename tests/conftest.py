import os.path

import pandas as pd
import pytest


@pytest.fixture()
# if testing horos; with our data, change this path to where your data is located
def horos_test_root():
    return os.path.normpath("./tests/horos_dummy")


@pytest.fixture()
def horos_test_seq_paths(horos_test_root):
    seq_paths = ['Impression_Cmr0064\\Anonymous_Study - 0\\CINE_segmented_LAX_2Ch_7\\',
                 'Impression_Cmr0064\\Anonymous_Study - 0\\CINE_segmented_LAX_3Ch_5\\',
                 'Impression_Cmr0064\\Anonymous_Study - 0\\CINE_segmented_LAX_4Ch_6\\',
                 'Impression_Cmr0064\\Anonymous_Study - 0\\CINE_segmented_SAX_28\\',
                 'Impression_Cmr0067\\Anonymous_Study - 0\\CINE_segmented_LAX_2Ch_7\\',
                 'Impression_Cmr0067\\Anonymous_Study - 0\\CINE_segmented_LAX_3Ch_9\\',
                 'Impression_Cmr0067\\Anonymous_Study - 0\\CINE_segmented_LAX_4Ch_10\\',
                 'Impression_Cmr0067\\Anonymous_Study - 0\\CINE_segmented_SAX_53\\']

    return [os.path.normpath(os.path.join(horos_test_root, seq_path)) for seq_path in seq_paths]


@pytest.fixture()
def horos_test_seq_n_slices():
    return [1, 1, 1, 13, 1, 1, 1, 13]


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


def pytest_addoption(parser):
    parser.addoption("--pth")
