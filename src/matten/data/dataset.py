import os
import shutil
from pathlib import Path
from typing import Callable, List, Optional, Union

import requests
import torch
import tqdm
from loguru import logger
from torch_geometric.data import InMemoryDataset as PyGInMemoryDataset
from torch_geometric.data import extract_bz2, extract_gz, extract_tar, extract_zip

from matten.data.data import DataPoint
from matten.utils import to_list, to_path


class InMemoryDataset(PyGInMemoryDataset):
    """
    Base in memory dataset.

    Subclass must implement:
      - get_data()

    Args:
        filenames: filenames of the dataset. If a full path (e.g. /home/user/...),
            this is used directly to get the file. If not a full path, but only the
            file name (e.g. test_data.json), then will try to look for the files at
            <root>/filenames. Note, the latter approach maybe useful when you data
            preprocessing step is time-consuming, in which case you can process it
            once and reuse it.
            If the files do not exist, will try to download from `url`.
        root: root to store find the raw data files or to store the processed
            datasets for later use.
        processed_dirname: the files give by `filenames` will be processed and saved
            to disk for faster loading later on. `processed_dirname` gives the
            directory name to store the processed files. This should be a string not
            a full path, and the processed files will be stored in
            `<root>/<processed_dirname>`
        url: path to download the dataset if not present.
        reuse: whether to reuse the processed file in `processed_dirname` if found.
        dataset_statistics_fn: a callable to compute dataset statistics. The
            callable is expected to take a list of `DataPoint` as input and probably
            and return a dict of dataset statistics.
    """

    # TODO, better to move the logic of using url to download to datamodule

    def __init__(
        self,
        filenames: List[str],
        *,
        root: Union[str, Path] = ".",
        processed_dirname: str = "processed",
        url: Optional[str] = None,
        dataset_statistics_fn: Callable = None,
        pre_transform: Callable = None,
        reuse: bool = True,
    ):
        self.filenames = to_list(filenames)
        self.root = root
        self.processed_dirname = processed_dirname
        self.url = url
        self.dataset_statistics_fn = dataset_statistics_fn

        # !!! don't delete this block.
        # Otherwise, the inherent children class will ignore the download function here
        class_type = type(self)
        if class_type != InMemoryDataset:
            # if "download" not in self.__class__.__dict__:
            #     class_type.download = InMemoryDataset.download
            if "process" not in self.__class__.__dict__:
                class_type.process = InMemoryDataset.process

        # whether to reuse?
        if not reuse:
            for f in self.processed_paths:
                p = to_path(f)
                if p.exists():
                    p.unlink()
                    logger.info(f"`reuse=False`, deleting preprocessed data file: {p}")

            # delete dataset statistics file
            if self.dataset_statistics_fn:
                p = self.dataset_statistics_path
                if p.exists():
                    p.unlink()
                    logger.info(
                        "`reuse=False`, deleting preprocessed dataset statistics "
                        f"file: {p}"
                    )
        else:
            if files_exist(self.processed_paths):
                logger.info(
                    f"Found existing processed data files: {self.processed_paths}. "
                    f"Will reuse them. To disable reuse, set `reuse=False` of "
                    "DataModule."
                )

            # copy dataset statistics to cwd if existing
            # note, should copy even dataset_statistics_f is not needed -- for
            # the case a later experiment want to use the dataset statistics
            p = self.dataset_statistics_path
            if p.exists():
                shutil.copy2(p, Path.cwd())
                logger.info(
                    f"Found existing dataset statistics in {p}. Copying it "
                    f"to {Path.cwd()}."
                )

        super().__init__(root=root, pre_transform=pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    def get_data(self) -> List[DataPoint]:
        """
        Process the (downloaded) files to get a list of data points.

        In this function, the files in ``self.raw_file_names`` (i.e. `<root>/<filenames>`
        should be processed to generate a list of ``DataPoint`` object.
        """
        raise NotImplementedError

    def process(self):
        data_list = self.get_data()

        #
        # process dataset statistics file
        #
        if self.dataset_statistics_fn is not None:
            statistics = self.dataset_statistics_fn(data_list)
            logger.info(f"dataset_statistics: {statistics}")

            # save statistics to disk
            # Save two copies, one in processed_dir, together with other processed
            # files, the other in CWD, mainly to be loaded by model.
            p = self.dataset_statistics_path
            torch.save(statistics, p)
            shutil.copy2(p, Path.cwd())
            logger.info(
                f"Dataset statistics saved to: {p} and "
                f"{Path.cwd().joinpath(self.dataset_statistics_path.name)}"
            )

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])

        logger.info(f"Processed data files saved as {self.processed_paths}.")

    # def download(self):
    #     # from torch_geometric.data import download_url
    #     # download_url(self.url, self.raw_dir)
    #     ## download_url does not work for some url, e.g. matbench ones
    #
    #     logger.info(
    #         f"Did not find files {self.raw_file_names} in {self.raw_dir}. Now try to "
    #         f"download from {self.url}."
    #     )
    #     try:
    #         filepath = _fetch_external_dataset(self.url, self.raw_dir)
    #         _extract_file(filepath, self.raw_dir)
    #     except Exception:
    #         raise RuntimeError(f"Failed download and extract file from {self.url}.")

    @property
    def raw_file_names(self):
        """
        Names of the raw files, not path.
        """
        return self.filenames

    @property
    def processed_file_names(self):
        """
        Names of the processed files, not path.

        Original filenames with _data.pt appended.
        """
        return [to_path(f).stem + "_data.pt" for f in self.filenames]

    @property
    def raw_dir(self) -> str:
        """
        Raw directory to find the files, i.e. self.root.
        """
        return str(to_path(self.root))

    @property
    def processed_dir(self) -> str:
        """
        Directory to store the processed files, i.e.
        <self.root>/<self.processed_dirname>
        """
        return str(to_path(self.root).joinpath(self.processed_dirname))

    @property
    def dataset_statistics_path(self) -> Path:
        """
        Path to the dataset statistics file
        """
        return to_path(self.processed_dir).joinpath("dataset_statistics.pt")


# this is copied from matmainer.dataset.utils with slight modifications
def _fetch_external_dataset(url, save_dir):
    """
    Downloads file from a given url
    Args:
        url (str): string of where to get external dataset
        save_dir (str): directory where to save the file to be retrieved

    Returns:
        path to the downloaded file
    """

    filename = url.rpartition("/")[2].split("?")[0]
    filepath = to_path(save_dir).joinpath(filename)

    # Fetch data from given url
    logger.info(f"Fetching {filename} from {url} to {filepath}")

    r = requests.get(url, stream=True)

    pbar = tqdm.tqdm(
        desc=f"Fetching {url} in MB",
        position=0,
        leave=True,
        ascii=True,
        total=len(r.content),
        unit="MB",
        unit_scale=1e-6,
    )
    chunk_size = 2048
    with open(filepath, "wb") as file_out:
        for chunk in r.iter_content(chunk_size=chunk_size):
            pbar.update(chunk_size)
            file_out.write(chunk)

    r.close()

    return filepath


def _extract_file(filepath, folder):
    """
    Decompress a file.

    Support file types include: .gz .tar.gz, .tgz, .bz2, .zip

    Args:
        filepath: path to the file
        folder: directory to store the extracted file
    """
    filepath = str(to_path(filepath))

    filepath_split = filepath.split(".")
    ext1 = filepath_split[-1].lower()
    ext2 = filepath_split[-2].lower()

    if ext1 == "gz":
        if ext2 == "tar":
            extract_tar(filepath, folder)
        else:
            extract_gz(filepath, folder)
    elif ext1 == "tgz":
        extract_tar(filepath, folder)
    elif ext1 == "bz2":
        extract_bz2(filepath, folder)
    elif ext1 == "zip":
        extract_zip(filepath, folder)
    else:
        # raise ValueError(
        #     f"Cannot extract file {filename}. Unsupported compression format: {ext1}."
        # )

        print(f"No decompression performed for file: {filepath}")


def files_exist(files: List[str]) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([os.path.exists(f) for f in files])
