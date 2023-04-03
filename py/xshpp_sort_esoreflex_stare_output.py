import sys
import logging
import shutil
import argparse

from pathlib import Path
from datetime import datetime
from astropy.io import fits


log = logging.getLogger(__name__)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filepath",
        type=str,
        help="Path to the ROOT_DATA_DIR configured in ESOREFLEX",
    )

    args = parser.parse_args()

    if not args.filepath:
        print("You need to supply a filepath. Stopping execution")
        exit(1)

    sort_esoreflex_outputs(args.filepath)


def sort_reduction(cdir, DIRS):
    reduc_time = cdir.stem.replace("T", " ") + " UTC"
    log.info(f"Sorting files for dataset reduced at {reduc_time}")

    # Datasets
    datasets = [f for f in cdir.iterdir() if f.stem.startswith("XSHOO")]
    datasets.sort()
    log.info(f"Found {len(datasets)} datasets")
    if len(datasets) == 0:
        log.info("No datasets found, nothing to do.")
        exit()

    # Temp directories
    tmp_dirs = [
        f
        for f in (
            DIRS["TMP_PRODUCTS_DIR"] / "xshooter/xsh_scired_slit_stare_1"
        ).iterdir()
        if not f.stem.startswith(".")
    ]
    tmp_dirs.sort()
    # log.info(f"Found {len(tmp_dirs)} directories for temporary files")

    # Make sure same number of temp dirs than datasets
    # if len(datasets) != len(tmp_dirs):
    #     raise ValueError(
    #         "Number of datasets and directories for temporary files is not the same, cannot sort files."
    #     )

    # check if no temp files created before the dataset reduction
    # t_reducstr = cdir.stem.replace('T', ' ')
    # dt_reduc = datetime.strptime(t_reducstr, "%Y-%m-%d %H:%M:%S")
    # for d in tmp_dirs:
    #     dtimestr = d.stem.replace('T', ' ')
    #     dtime = datetime.strptime(dtimestr, "%Y-%m-%d %H:%M:%S")
    #     if dtime < dt_reduc:
    #         raise ValueError("Some temporary files were created before the dataset reduction started. "
    #                          "I cannot sort files as some may be due to old reductions.")

    # Associate each temp dir to a dataset
    data = {
        "UVB": {"data": [], "tmp": []},
        "VIS": {"data": [], "tmp": []},
        "NIR": {"data": [], "tmp": []},
    }
    for i, dset in enumerate(datasets):
        log.debug(f"Searching dataset {dset.stem}")
        arm = fits.getheader([f for f in dset.glob("*_SCI_SLIT_FLUX_MERGE2D*")][0])[
            "HIERARCH ESO SEQ ARM"
        ]
        dset_dtobs = fits.getheader(
            [f for f in dset.glob("*_SCI_SLIT_FLUX_MERGE2D*")][0]
        )["DATE-OBS"]

        log.debug(f"-> {arm} arm, observed at {dset_dtobs}")
        data[arm]["data"].append(dset)
        found_tmp = False
        for dtmp in tmp_dirs:
            tmp_dtobs = fits.getheader([f for f in dtmp.iterdir() if not f.stem.startswith(".")][0])["DATE-OBS"]
            if dset_dtobs == tmp_dtobs:
                found_tmp = True
                data[arm]["tmp"].append(dtmp)
                log.debug(
                    f"Found {dtmp.stem} temporary directory data associated with dataset {dset.stem}"
                )
                break
        if not found_tmp:
            raise ValueError(
                "Did not find any temporary directory data matching the DATE-OBS"
                f"of dataset {dset.stem}"
            )

    log.info(
        "Found the following individual datasets:\n"
        f"{len(data['UVB']['data'])} UVB\n"
        f"{len(data['VIS']['data'])} VIS\n"
        f"{len(data['NIR']['data'])} NIR\n"
    )

    # Move the datasets into the correct directory structure in order to use
    # the post-processing scripts
    for arm in ["UVB", "VIS", "NIR"]:
        if len(data[arm]["data"]) == 0:
            log.info(f"No {arm} datasets to copy")
            continue

        log.info(f"Moving data for {arm} arm")

        # Root directory for each arm
        d = DIRS["ROOT_DATA_DIR"] / arm
        d.mkdir(exist_ok=True)

        for i, sd in enumerate(data[arm]["data"]):
            tmp = data[arm]["tmp"][i]
            log.debug(
                "Copying:\n"
                f"{sd}\n"
                "to\n"
                f"{d/sd.stem}\n"
                "and\n"
                f"{tmp}\n"
                "to\n"
                f"{d/sd.stem/tmp.stem}"
            )
            shutil.copytree(sd, d / sd.stem, dirs_exist_ok=True)
            shutil.copytree(tmp, d / sd.stem / tmp.stem, dirs_exist_ok=True)
    log.info(f"Finished sorting files succesfully for reduction {cdir}")


def sort_esoreflex_outputs(ROOT_DATA_DIR):
    ROOT_DATA_DIR = Path(ROOT_DATA_DIR)
    DIRS = {
        "ROOT_DATA_DIR": ROOT_DATA_DIR,
        "END_PRODUCTS_DIR": ROOT_DATA_DIR / "reflex_end_products",
        "TMP_PRODUCTS_DIR": ROOT_DATA_DIR / "reflex_tmp_products",
    }

    # Make sure directories exist
    for k, d in DIRS.items():
        if not d.exists():
            raise ValueError(
                f"Directory {d} does not exist. "
                "Did you change the ESOREFLEX default directory structure?"
            )

    # Look for reduction folders, exclude folders that start with '.'
    # This is to avoid including folders such as '.DS_Store' on macOS
    # or other hidden folders
    reduc_dirs = [
        f for f in DIRS["END_PRODUCTS_DIR"].iterdir() if not f.stem.startswith(".")
    ]
    if len(reduc_dirs) > 1:
        log.warning(
            "Found more than one reduction, will try to iterate through them"
            " but may fail. If there are old reductions in your"
            " reflex_end_products directory, try to clean them up by either"
            " deleting them or moving then to another directory."
        )
        for cdir in reduc_dirs:
            sort_reduction(cdir, DIRS)
    elif len(reduc_dirs) == 0:
        log.info("No datasets found, nothing to do.")
        exit()
    else:
        cdir = reduc_dirs[0]
        sort_reduction(cdir, DIRS)


if __name__ == "__main__":
    main()
