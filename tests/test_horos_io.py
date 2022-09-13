#!/usr/bin/env python
"""
Tests for `horos_io` package.
TODO: move away from legacy loading of contours n shit in tests

contains tests for general functionality of loading contours, images and shit exported from our HOROS Mac;
for tests regarding actual integrity of a horos dataset, refer to test_horos_dataset.py
"""
# pylint: disable=redefined-outer-name
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import pydicom
import pytest
from matplotlib import pyplot as plt

import horos_io._legacy
import horos_io.core
from horos_io import cli, load_sax_sequence
from horos_io import cmr
from horos_io._utils import globSSF


def test__make_contour_info_csv_pipeline(horos_test_root: str):
    cli.make_contour_info_csv(horos_test_root,
                                       out=os.path.join(horos_test_root, "test_contour_info.csv"))
    assert True


def test__make_image_info_csv_pipeline(horos_test_root: str):
    cli.make_image_info_csv(horos_test_root,
                                     out=os.path.join(horos_test_root, "test_image_info.csv"))
    assert True


def test_get_n_frames_from_seq_path(horos_test_root):
    for seq_path in globSSF("*/*/*/", root_dir=horos_test_root):
        seq_root = os.path.join(horos_test_root, seq_path)
        n_frames = horos_io.core.get_n_frames_from_seq_path(seq_root)
        assert n_frames == 25, f"n_frames should be 25, but is {n_frames}; failed at {seq_root}"


def test_get_n_slices_from_seq_path(horos_test_seq_paths, horos_test_seq_n_slices):
    for seq_path, n_slices in zip(horos_test_seq_paths, horos_test_seq_n_slices):
        assert horos_io.core.get_n_slices_from_seq_path(seq_path) == n_slices


def test_load_lax_sequence_pipeline(horos_test_seq_paths):
    lax_paths = [seq_path for seq_path in horos_test_seq_paths
                 if ("2ch" in seq_path.lower() or "3ch" in seq_path.lower() or "4ch" in seq_path.lower())]
    assert len(lax_paths) > 0

    for lax_path in lax_paths:
        sample = horos_io.load_lax_sequence(lax_path)
        assert sample.shape == (25,)


def test_load_sax_sequence_pipeline(horos_test_seq_paths, horos_basal_first_file):
    sax_paths = [seq_path for seq_path in horos_test_seq_paths if "sax" in seq_path.lower()]
    assert len(sax_paths) > 0

    basal_firsts = horos_io._legacy._load_basaL_first_as_list(horos_basal_first_file)
    for sax_path in sax_paths:
        sample = horos_io._legacy.load_sax_sequence(sax_path, re.findall(r"\d{4}", sax_path)[0] in basal_firsts)
        assert sample.shape == (horos_io.core.get_n_frames_from_seq_path(sax_path),
                                horos_io.core.get_n_slices_from_seq_path(sax_path))


def test_load_sequence_pipeline(horos_image_info_path):
    image_info = pd.read_csv(horos_image_info_path, index_col=0)

    for i, row in image_info.iterrows():
        sample = horos_io._legacy.load_sequence(row["location"], row["basal_first"])
        for entry in sample.flatten():
            assert isinstance(entry, pydicom.FileDataset)


def test_load_horos_contour(horos_image_info_path, horos_contour_info_path):
    contour_info = pd.read_csv(horos_contour_info_path, index_col=0)
    image_info = pd.read_csv(horos_image_info_path, index_col=0)
    combined = pd.merge(contour_info[contour_info["location"].notna()],
                        image_info, on=["ID", "slice_type"], suffixes=("_contour", "_images"))

    for i, row in combined.iterrows():
        result: np.ndarray = horos_io.core.load_horos_contour(row["location_contour"], row["location_images"])
        assert (result != 0).any()


@pytest.mark.skipif(not ("visual" in sys.argv),
                    reason="should only be run, if we want to manually sanitiy check the ordering")
@pytest.mark.visual
def test_visually_confirm_ordering(horos_image_info_path, horos_contour_info_path):
    """
    fix the skipif shenanigans
    Args:
        horos_image_info_path:
        horos_contour_info_path:

    Returns:
    """
    contour_info = pd.read_csv(horos_contour_info_path, index_col=0)
    image_info = pd.read_csv(horos_image_info_path, index_col=0)
    image_info = image_info[image_info["slice_type"] == "cine_sa"]
    combined = pd.merge(contour_info[contour_info["location"].notna()],
                        image_info, on=["ID", "slice_type"], suffixes=("_contour", "_images"))

    for i, row in combined.iterrows():
        cines = horos_io._legacy.load_sequence(row["location_images"], row["basal_first"])
        contours = horos_io.core.load_horos_contour(row["location_contour"], row["location_images"])

        mid_minus3 = cines.shape[1] // 2 - 3

        plt.imshow(cines[0, mid_minus3].pixel_array)
        plt.scatter(*zip(*contours[0, mid_minus3]))
        plt.show()
    assert True


def test__load_omega_4ch_contour():
    pth = "tests/omega_4ch.xml"
    result = horos_io.core.load_horos_contour(pth, (25, 1))

    for contour_name, contours in result.items():
        locs = np.array((0, 10))  # ED in frame 0 and ES in frame 10
        target = np.zeros_like(contours, dtype=bool)
        target[locs] = True
        assert all((contours != 0) == target)


@pytest.mark.parametrize("sax_path, basal_first",
                         [('Impression_Cmr0064\\Anonymous_Study - 0\\CINE_segmented_SAX_28\\', True),
                          ('Impression_Cmr0067\\Anonymous_Study - 0\\CINE_segmented_SAX_53\\', False)])
def test_sort_sax_by_y(sax_path, horos_test_root, basal_first):
    """
    test to see if we can load SAX Stack in right order without relying on manual confirmation
    """
    pth = os.path.normpath(os.path.join(horos_test_root, sax_path))
    sax_manual = load_sax_sequence(pth, basal_first=basal_first)
    sax_by_y = horos_io.sort_SAX_by_y(load_sax_sequence(pth, basal_first=False))
    for by_y, manual in zip(sax_by_y.flatten(), sax_manual.flatten()):
        assert (by_y.pixel_array == manual.pixel_array).all()




def sax_method_iter():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth")
    args, _ = parser.parse_known_args()
    if args.pth is not None:
        image_info = pd.read_csv(os.path.join(args.pth, "image_info.csv"), index_col=0)
        for i, row in image_info[image_info["slice_type"] == "cine_sa"].iterrows():
            full_seq_path = os.path.normpath(os.path.join(args.pth, row["seq_path"]))
            yield full_seq_path, row
    else:
        yield None, None # dummy values

@pytest.mark.skipif(not "--pth" in sys.argv, reason="this test is only to be run when specifying a path to a dataset")
@pytest.mark.parametrize("full_seq_path, row", sax_method_iter())
def test_sort_sax_by_y_hard(full_seq_path, row):
    """yaaay"""
    sax_manual = horos_io._legacy.load_sax_sequence(full_seq_path, row["basal_first"])
    sax_by_y = horos_io.sort_SAX_by_y(load_sax_sequence(full_seq_path, False))
    for s_manual, s_by_y in zip(sax_manual.flatten(), sax_by_y.flatten()):
        assert (s_manual.pixel_array == s_by_y.pixel_array).all()
