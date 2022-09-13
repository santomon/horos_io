"""
Console script for horos io;
will incorporate functionality for running visual sanity checks on contours overlayed on top of the data;

generating .csv files for data
"""
import os
import sys
import typing
from typing import Optional, NoReturn

import click
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from horos_io import load_horos_contour
from horos_io._utils import globSSF, mask_from_omega_contour, get_log, get_log_path
from horos_io.cmr import Path, get_image_info, get_contour_info, get_combined_info
from horos_io.ui import shitty_manual_confirm
from horos_io.validation import visually_confirm_omega_iter, write_log, last_validation_was_successful, contour_is_valid


@click.group()
def main():
    """Console script for horos_io."""
    return 0


@click.command("image_info",
               help="saves information about the horos dataset as csv, such that it can be later used for efficient"
                    "loading of the data")
@click.option("--root", default=".", help="directory to the Horos Dataset")
@click.option("--silent", default=False, is_flag=True, help="if set, will not print out the result")
@click.option("--out", default=None,
              help="how the image_info,csv should be saved; if not specified, will save an image_info.csv in root")
def make_image_info_csv(root: Path,
                        out: Optional[Path],
                        silent: bool) -> NoReturn:
    """
    one-time operation to update the information of the dataset;
    columns:
    ("seq_path",  <relative path starting from the root of the data>
     "location",  joins the rootdir with seq path
     "ID",        <Impression_Cmr{ID}>
     "slice_type", one of config.slice_types
     "n_frames",
     "n_slices",

    seq_path is relative to the root
    location is joint path of root and seq_path

    creates a csv file containing location of the sequence.
    sequence can be loaded with load_cine_sequence
    Args:
        silent: if silent, will not print out the result to console
        root: root of the data; contents should be directories of Impression Studies
        out: optional parameter to specify, what and where to save; if None, will save image_info.csv inside root
    Returns:
    """
    click.echo(f"retrieving image info from {root}")
    result = get_image_info(root)
    if not silent:
        click.echo("Findings:\n\n")
        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            click.echo(result)
    if out is None:
        out = os.path.normpath(os.path.join(root, "image_info.csv"))
    click.echo(f"saving result to {out}")
    result.to_csv(out)


@click.command("contour_info",
               help="saves information about the horos dataset as csv, such that it can be later used for efficient"
                    "loading of the data")
@click.option("--root", default=".", help="directory to the Horos Dataset")
@click.option("--silent", default=False, is_flag=True, help="if set, will not print out the result")
@click.option("--out", default=None,
              help="how the image_info,csv should be saved; if not specified, will save an image_info.csv in root")
def make_contour_info_csv(root: Path, out: Optional[Path], silent: bool) -> NoReturn:
    """
    one-time operation to update the information of the dataset;

    ("ID": <Impression_Cmr{ID}>,
     "contour_path": relative path to the contour file, starting from root of the data,
     "contour_type": one of config.contour_types,
     "location": joins the root dir with the sequence path
     "slice_type": one of config.slice_types,
    )

    creates a csv file containing location of all contour files; or whether they exist at all.
    Args:
        root: root of the data; contents should be directories of Impression Studies
    Returns:
    """
    root = os.path.normpath(root)
    click.echo(f"creating contour info for {root}")
    result = get_contour_info(root)
    if not silent:
        click.echo("Findings:\n\n")
        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            click.echo(result)

    if out is None:
        out = os.path.normpath(os.path.join(root, "contour_info.csv"))
    click.echo(f"saving contour info to: {out}")
    result.to_csv(out)


@click.command()
@click.option("--by", required=True, help="the name that will be recorded in the log_")
@click.option("--root", default=".", help="source root of the Horos dataset")
@click.option("--only_unconfirmed", is_flag=True, default=False,
              help="will only prompt you cases, that have not been confirmed by the user yet (registered in log_)")
@click.option("--log", default=None,
              help="if None, will save or update a val_contour_log.csv in root, else checks your specified log_ instead")
def validate(by: str, root: Path, only_unconfirmed: bool, log: Path):
    """
    visually confirm literally every existing omega contour... (might need to change with params to return
    an adequate iterator)

    will also save a dataframe / update a dataframe in <root>/val_contour_log.csv,
    that logs review result with timestamp
    """
    click.echo(f"starting validation process in root: {root}")

    for ID, omega, f, s, cines, contours in tqdm(visually_confirm_omega_iter(combined_info=get_combined_info(root))):
        if only_unconfirmed and last_validation_was_successful(get_log(log, root), ID=ID, frame=f, slice=s,
                                                               contour_type=omega):  # CAVE: hard coded
            continue
        fig, ax = plt.subplots()
        ax.imshow(cines[f, s].pixel_array)
        ax.imshow(mask_from_omega_contour(cines, contours, f, s), cmap="jet", alpha=0.5)
        ax.figure.suptitle(f"{ID}: {omega} f: {f}, s: {s}, ")
        ok, remark = shitty_manual_confirm(ax)
        write_log(log, **locals())
    click.echo("batch finished validation!")


@click.command("rename", help="renames all contours with .xml and .rois_series attached to it")
@click.argument("old", nargs=1)
@click.argument("target", nargs=1)
@click.option("--root", help="Data root")
def rename_contour_files(root: str, old: str, target: str) -> typing.NoReturn:
    """
    renames .xml and .roi_series files by dictionary, that lie in the impression folders:
    root |--- Impression_CmrXXXX
                    |---- lv_epi.xml
                    |---- lv_epi.roi_series
    Args:
        root:
    Returns:
    """
    files = globSSF(f"*/{old}.xml", root_dir=root) + globSSF(f"*/{old}.rois_series", root_dir=root)
    for f in files:
        path, f_name = os.path.split(f)
        f_name, f_ending = f_name.split(".")
        src = os.path.normpath(os.path.join(root, path, f"{old}.{f_ending}"))
        dst = os.path.normpath(os.path.join(root, path, f"{target}.{f_ending}"))
        if os.path.isfile(dst):
            raise FileExistsError(f"{dst} for {src} already exists")
        os.rename(src, dst)
        click.echo(f"renamed {src} to {dst}")
    click.echo("finished")


@click.group("check", help="given a root and a log file, checks if the user (--by) has seen all the samples")
def check():
    pass


@click.option("--by", required=True, help="the name to check for in the logs")
@click.option("--root", default=".", help="source root to the Horos dataset")
@click.option("--log", default=None,
              help="if None, will check for val_contour_log.csv in root, else checks your specified log instead")
@click.command("failed", help="check to log for which contours did not and in validation by the user")
def check_failed(by: str, root: str, log: str):
    click.echo(
        f"checking for failed validations by {by} for dataset in {root}, using logfile {get_log_path(log, root)}")
    log_df = get_log(log, root)
    failed = log_df.apply(lambda row: not last_validation_was_successful(log_df,
                                                                         by=row["by"],
                                                                         ID=row["ID"],
                                                                         frame=row["frame"],
                                                                         slice=row["slice"]), axis=1)
    result = log_df[(log_df["by"] == by) & failed]
    result = result[["by", "ID", "frame", "slice"]].drop_duplicates()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        click.echo(result)
        click.echo("----------------")
        click.echo(f"These samples did not end with confirmation by user {by}")


@click.command("unobserved",
               help="looks at the data and the logs to determine, which contours by which ID still needs to,"
                    "be evalutated by a given user;")
@click.option("--root", default=".", help="source root to the Horos dataset")
@click.option("--log", default=None,
              help="if None, will check for val_contour_log.csv in root, else checks your specified log instead")
@click.option("--by", required=True, help="the name to check for in the logs")
def check_unobserved(by: str, root: str, log: str):
    click.echo(
        f"checking, which contours have not been seen by user {by} yet in datasat at {root}, using log {get_log_path(log, root)}")
    log_df = get_log(log, root)
    combined_info_ = get_combined_info(root)
    # combined_info_["ID", "contour_type"]
    click.echo("Under construction")
    return

    # for g_n, group in combined_info_.groupby(["ID", "slice_type", "contour_type", "location_contour", "location_images"]):
    #     click.echo(group)
    #
    # # TODO: open the contour to look for which frame / slice needs to be seen
    # # TODO: filter by omega


@click.command("existing", help="look at exsiting contours in root, given a number of contour types; will check"
                                "for existing <contour_type>.xml files that are in child directories of root")
@click.argument("contour_types", nargs=-1)
@click.option("--n", default=None, type=int,
              help="If passed, will check if a contour file has at least that many contours")
@click.option("--root", default=".", help="source root to the Horos dataset")
def check_existing_contours(contour_types: typing.List[str], root: str, n: Optional[int]):
    """currently terrible and unhelpful vaidation process... honestly using pytest was pretty king"""
    click.echo(f"checking contour types {contour_types} in root: {root}...")
    combined_info = get_combined_info(root)

    click.echo("the following contours need to be rechecked: ")
    # TODO: improve logging...
    # really, pytest for this kinda stuff wasnt half bad
    for contour_type in contour_types:
        eligible = combined_info[
            (combined_info["contour_type"] == contour_type) & (combined_info["location_contour"].notna())]
        eligible.loc[:, "valid"] = eligible.apply(
            lambda row: contour_is_valid(load_horos_contour(row["location_contour"], row["location_images"]), n),
            axis=1)
        click.echo(eligible[eligible["valid"] != True][["ID", "contour_path", "contour_type"]])

    click.echo("finished checking")


main.add_command(make_image_info_csv)
main.add_command(make_contour_info_csv)
main.add_command(rename_contour_files)
main.add_command(validate)
main.add_command(check)
check.add_command(check_failed)
check.add_command(check_unobserved)
check.add_command(check_existing_contours)

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
