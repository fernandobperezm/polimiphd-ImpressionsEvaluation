""" MINDSmallReader.py
This module reads the small version of the Microsoft News Dataset (MIND Small).

Notes
-----
Columns of the dataset
    Impression ID
        MIND Dataset identifier for the impression. THEY DO NOT COME IN ORDER, i.e., a higher impression ID
        does not mean that the impression occurred later. See user "U1000" where impression "86767" comes first than
        impression "46640", so do not rely on impression reproducibility.pyid to partition the dataset. Also, partition id is shuffled
        across users, so two consecutive impression ids might refer to different users.
    History
        A column telling the previous interactions_exploded of users before the collection of data, i.e.,
        interactions_exploded that happened before the dataset was created and before the collection window for the
        dataset. There is no way to match any interaction here with a timestamp or impression. The paper mentions the
        history is sorted by order of interactions_exploded, i.e., the first element is the first click of the user,
        the second element is the second, and so on.
    Impressions
        A column telling the impressions and possible interactions_exploded that users had with that impression. A user
        may have more than one interaction for a given impression. It is stored as a :class:`str` and with the following
        format "NXXXX-Y" where XXXX are digits and represent the item id. Y is 0 or 1 and tells if the user interacted
        or not with the item NXXXX in the impression.
"""
import os
import re
import urllib.request
import zipfile
from enum import Enum
from typing import Optional

import attr
import numpy as np
import pandas as pd
import scipy.sparse as sp
from recsys_framework.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs
from recsys_framework.Utils.conf_logging import get_logger
from recsys_framework.Utils.decorators import timeit
from tqdm import tqdm

from data_splitter import remove_duplicates_in_interactions, remove_users_without_min_number_of_interactions, \
    split_sequential_train_test_by_column_threshold, T_KEEP, filter_impressions_by_interactions_index, \
    split_sequential_train_test_by_num_records_on_test
from mixins import BinaryImplicitDataset, BaseDataReader
from utils import typed_cache

tqdm.pandas()

logger = get_logger(
    logger_name=__file__,
)


def docstring_for_df(
    df: pd.DataFrame
) -> str:
    """

    References
    ----------
    .. [1] https://stackoverflow.com/a/61041468/13385583
    """
    docstring = 'Index:\n'
    docstring += f'    {df.index}\n'
    docstring += 'Columns:\n'
    for col in df.columns:
        docstring += f'    Name: {df[col].name}, dtype={df[col].dtype}, nullable: {df[col].hasnans}\n'

    return docstring


@timeit
def compare_impressions(
    impression_ids_1: np.ndarray,
    impression_ids_2: np.ndarray,
    impressions_1: np.ndarray,
    impressions_2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    assert impression_ids_1.shape == impression_ids_2.shape
    assert impressions_1.shape == impressions_2.shape
    assert impression_ids_1.shape[0] == impressions_1.shape[0]

    ids_equal = np.equal(impression_ids_1, impression_ids_2)
    arrays_equal = np.array(
        [
            np.array_equal(imp1, imp2)
            for imp1, imp2 in zip(impressions_1, impressions_2)
        ]
    )

    return ids_equal, arrays_equal


def convert_impressions_str_to_array(
    impressions: Optional[str],
) -> list[str]:
    """Extract interacted items from impressions.

    This function expects `impressions` to be a string as follows:
    - if None or NA, then this method returns [].
    - else, "NXXXX-Y NZZZZZ-Y NWWW-Y", i.e., a white-space separated string,
      where each item begins with an N followed by several numbers (the item id),
      then a dash character (-), then either 0 or 1. 1 Means the user interacted
      with this item, 0 otherwise.

    Notes
    -----
    Do not try to numba.jit decorate this function, given that numba does not have optimizations
    for the :py:mod:`re` module. Trying to decorate this function will cause it to fail on runtime.

    Returns
    -------
    list[str]
        a list containing the interacted item ids, if any, in the format "NXXXX". If the `impressions` string
        is None, empty, or non-interacted impression, then an empty list is returned.
        Else, a list containing the ids is returned.
    """
    if impressions is None or pd.isna(impressions) or impressions == "":
        return []

    return impressions.replace(
        "-0", ""
    ).replace(
        "-1", ""
    ).split(" ")


def extract_item_positions_in_impressions(
    impressions: Optional[str],
) -> list[int]:
    """Extract interacted items from impressions.

    This function expects `impressions` to be a string as follows:
    - if None or NA, then this method returns [].
    - else, "NXXXX-Y NZZZZZ-Y NWWW-Y", i.e., a white-space separated string,
      where each item begins with an N followed by several numbers (the item id),
      then a dash character (-), then either 0 or 1. 1 Means the user interacted
      with this item, 0 otherwise.

    Notes
    -----
    Do not try to numba.jit decorate this function, given that numba does not have optimizations
    for the :py:mod:`re` module. Trying to decorate this function will cause it to fail on runtime.

    Returns
    -------
    list[str]
        a list containing the interacted item ids, if any, in the format "NXXXX". If the `impressions` string
        is None, empty, or non-interacted impression, then an empty list is returned.
        Else, a list containing the ids is returned.
    """
    if impressions is None or pd.isna(impressions) or impressions == "":
        return []

    impressions_list = impressions.split(" ")

    return [
        pos
        for pos, item in enumerate(impressions_list)
        if item.endswith("-1")
    ]


def extract_interacted_item_in_impressions(
    impressions: Optional[str],
) -> list[str]:
    """Extract interacted items from impressions.

    This function expects `impressions` to be a string as follows:
    - if None or NA, then this method returns [].
    - else, "NXXXX-Y NZZZZZ-Y NWWW-Y", i.e., a white-space separated string,
      where each item begins with an N followed by several numbers (the item id),
      then a dash character (-), then either 0 or 1. 1 Means the user interacted
      with this item, 0 otherwise.

    Notes
    -----
    Do not try to numba.jit decorate this function, given that numba does not have optimizations
    for the :py:mod:`re` module. Trying to decorate this function will cause it to fail on runtime.

    Returns
    -------
    list[str]
        a list containing the interacted item ids, if any, in the format "NXXXX". If the `impressions` string
        is None, empty, or non-interacted impression, then an empty list is returned.
        Else, a list containing the ids is returned.
    """
    if impressions is None or pd.isna(impressions) or impressions == "":
        return []

    item_ids = re.findall(
        pattern=r"(N\d+)-1",
        string=impressions,
    )
    return item_ids


def download_remote_file(
    url: str,
    destination_filename: str,
    force_download=False,
):
    """
    Download a URL to a temporary file.
    See: https://docs.microsoft.com/en-us/azure/open-datasets/dataset-microsoft-news?tabs=azureml-opendatasets#functions
    """

    if (
        not force_download
        and os.path.isfile(destination_filename)
    ):
        logger.info(
            f'Bypassing download of already-downloaded file {os.path.basename(url)}'
        )

        return destination_filename

    logger.info(
        f'Downloading file {os.path.basename(url)} to {destination_filename}'
    )

    urllib.request.urlretrieve(
        url=url,
        filename=destination_filename,
        reporthook=None,
    )
    assert os.path.isfile(destination_filename)

    logger.info(
        f'Downloaded file {os.path.basename(url)} to {destination_filename}, {os.path.getsize(destination_filename)} bytes.'
    )
    return destination_filename


class MINDVariant(Enum):
    SMALL = "SMALL"
    LARGE = "LARGE"


@attr.frozen
class MINDSmallConfig:
    data_folder = os.path.join(
        ".", "data", "MIND-SMALL",
    )
    variant = MINDVariant.SMALL

    num_train_data_points = 156_965
    num_validation_data_points = 73_152
    num_test_data_points = 0

    base_url = 'https://mind201910small.blob.core.windows.net/release'

    train_remote_filename = 'MINDsmall_train.zip'
    validation_remote_filename = 'MINDsmall_dev.zip'
    test_remote_filename = ''

    parquet_engine = "pyarrow"
    parquet_use_nullable_dtypes = True

    min_number_of_interactions = 3
    binarize_impressions = True
    keep_duplicates: T_KEEP = "first"

    # The SMALL variant of the dataset has no test set, therefore it must always be false.
    use_test_set = attr.field(
        default=False,
        validator=[
            attr.validators.instance_of(bool),
            lambda _, att, val: not val
        ]
    )


@attr.frozen
class MINDLargeConfig(MINDSmallConfig):
    data_folder = os.path.join(
        ".", "data", "MIND-LARGE",
    )
    variant = MINDVariant.LARGE

    num_train_data_points = 2_232_748
    num_validation_data_points = 376_471
    num_test_data_points = 2_370_727

    train_remote_filename = 'MINDlarge_train.zip'
    validation_remote_filename = 'MINDlarge_dev.zip'
    test_remote_filename = 'MINDlarge_test.zip'

    # The LARGE variant can use the test set.
    use_test_set = attr.field(
        default=False,
        validator=[
            attr.validators.instance_of(bool),
        ]
    )


class MINDRawData:
    """Class that reads the 'raw' MIND-SMALL data from disk.

    We refer to raw data as the original representation of the data,
    without cleaning or much processing. The dataset originally has XYZ-fernando-debugger data points.
    """

    def __init__(
        self,
        config: MINDSmallConfig,
    ):
        self._config = config

        self._original_dataset_root_folder = os.path.join(
            self._config.data_folder, "original",
        )

        self._original_dataset_train_folder = os.path.join(
            self._original_dataset_root_folder, "train",
        )
        self._original_dataset_train_file = os.path.join(
            self._original_dataset_train_folder, "behaviors.tsv",
        )

        self._original_dataset_validation_folder = os.path.join(
            self._original_dataset_root_folder, "dev",
        )
        self._original_dataset_validation_file = os.path.join(
            self._original_dataset_validation_folder, "behaviors.tsv",
        )

        self._original_dataset_test_folder = os.path.join(
            self._original_dataset_root_folder, "test",
        )
        self._original_dataset_test_file = os.path.join(
            self._original_dataset_test_folder, "behaviors.tsv",
        )

        self.num_train_data_points = self._config.num_train_data_points
        self.num_validation_data_points = self._config.num_validation_data_points
        self.num_test_data_points = self._config.num_test_data_points

        self._pandas_read_csv_kwargs = dict(
            header=None,
            names=['impression_id', 'user_id', 'str_timestamp', 'str_history', 'str_impressions'],
            dtype={
                "impression_id": np.int32,
                "user_id": pd.StringDtype(),
                "str_timestamp": pd.StringDtype(),
                "str_history": pd.StringDtype(),  # either 'object' or pd.StringDtype() produce the same column.
                "str_impressions": pd.StringDtype(),  # either 'object' or pd.StringDtype() produce the same column.
            },
            sep="\t",
        )

        self._pandas_to_datetime_kwargs = dict(
            errors="raise",
            dayfirst=False,  # The data is in MM/DD, so tell pandas to read dates in that way.
            yearfirst=False,
            utc=None,  # No information about dates being UTC or not.
            format="%m/%d/%Y %I:%M:%S %p",  # Format: MM/dd/YYYY HH:mm:ss AM or MM/dd/YYYY HH:mm:ss PM
            exact=True,
            infer_datetime_format=False,  # Pandas get rids of the AM/PM if it tries to infer the format.
        )

    @property  # type: ignore
    @typed_cache
    def train(self) -> pd.DataFrame:
        """ Interactions Dask Dataframe.

        The columns of the dataframe are:

        user_id : np.int32
        time_step : np.int32
        item_id : np.int32
            A value of 1 indicates no interactions_exploded with any item in the list. A value of 0 indicates error.
        """
        if not os.path.exists(self._original_dataset_train_file):
            self._download_dataset()

        # The behaviors.tsv file contains the impression logs and users' news click histories.
        # It has 5 columns divided by the tab symbol:
        # - impression_id: The ID of an impression.
        # - user_id: The anonymous ID of a user.
        # - str_timestamp: The impression time with format "MM/DD/YYYY HH:MM:SS AM/PM". No information about the user's
        #   timezone, the system's timezone, or if it is UTC.
        # - str_history: The news click history (ID list of clicked news) of this user before this impression. Seems
        #   that these values are not updated across impressions.
        # - str_impressions: List of news displayed in this impression and user's click behaviors on them (1 for click
        # and 0 for non-click).
        df_train: pd.DataFrame = pd.read_csv(
            filepath_or_buffer=self._original_dataset_train_file,
            **self._pandas_read_csv_kwargs,
        )
        df_train["timestamp"] = pd.to_datetime(
            arg=df_train.str_timestamp,
            **self._pandas_to_datetime_kwargs
        )
        df_train["item_id"] = df_train.str_impressions.progress_map(
            extract_interacted_item_in_impressions,
            na_action=None,  # Our function handles NA, so better to send them to the function.
        )
        df_train["impressions"] = df_train.str_impressions.progress_map(
            convert_impressions_str_to_array,
            na_action=None,  # Our function handles NA, so better to send them to the function.
        )

        assert df_train.shape[0] == self.num_train_data_points

        return df_train

    @property  # type: ignore
    @typed_cache
    def validation(self) -> pd.DataFrame:
        """ Interactions Dask Dataframe.

        The columns of the dataframe are:

        user_id : np.int32
        time_step : np.int32
        item_id : np.int32
            A value of 1 indicates no interactions_exploded with any item in the list. A value of 0 indicates error.
        """
        if not os.path.exists(self._original_dataset_validation_file):
            self._download_dataset()

        # The behaviors.tsv file contains the impression logs and users' news click histories.
        # It has 5 columns divided by the tab symbol:
        # - impression_id: The ID of an impression.
        # - user_id: The anonymous ID of a user.
        # - str_timestamp: The impression time with format "MM/DD/YYYY HH:MM:SS AM/PM". No information about the user's
        #   timezone, the system's timezone, or if it is UTC.
        # - str_history: The news click history (ID list of clicked news) of this user before this impression. Seems
        #   that these values are not updated across impressions.
        # - str_impressions: List of news displayed in this impression and user's click behaviors on them (1 for click
        # and 0 for non-click).
        df_validation: pd.DataFrame = pd.read_csv(
            filepath_or_buffer=self._original_dataset_validation_file,
            **self._pandas_read_csv_kwargs
        )
        df_validation["timestamp"] = pd.to_datetime(
            arg=df_validation.str_timestamp,
            **self._pandas_to_datetime_kwargs
        )
        df_validation["item_id"] = df_validation["str_impressions"].progress_map(
            extract_interacted_item_in_impressions,
            na_action=None,  # Our function handles NA, so better to send them to the function.
        )
        df_validation["impressions"] = df_validation["str_impressions"].progress_map(
            convert_impressions_str_to_array,
            na_action=None,  # Our function handles NA, so better to send them to the function.
        )

        assert df_validation.shape[0] == self.num_validation_data_points

        return df_validation.copy()

    @property  # type: ignore
    @typed_cache
    def test(self) -> pd.DataFrame:
        """ Interactions Dask Dataframe.

        The columns of the dataframe are:

        user_id : np.int32
        time_step : np.int32
        item_id : np.int32
            A value of 1 indicates no interactions_exploded with any item in the list. A value of 0 indicates error.
        """
        if not os.path.exists(self._original_dataset_test_file):
            self._download_dataset()

        if self._config.variant == MINDVariant.SMALL:
            # The SMALL variant of MIND does not have a test set. We return an empty dataframe when asked for it.
            return pd.DataFrame(
                data=None,
                columns=self._pandas_read_csv_kwargs["names"],
            ).astype(
                dtype=self._pandas_read_csv_kwargs["dtype"],
            )

        # The behaviors.tsv file contains the impression logs and users' news click histories.
        # It has 5 columns divided by the tab symbol:
        # - impression_id: The ID of an impression.
        # - user_id: The anonymous ID of a user.
        # - str_timestamp: The impression time with format "MM/DD/YYYY HH:MM:SS AM/PM". No information about the user's
        #   timezone, the system's timezone, or if it is UTC.
        # - str_history: The news click history (ID list of clicked news) of this user before this impression. Seems
        #   that these values are not updated across impressions.
        # - str_impressions: List of news displayed in this impression and user's click behaviors on them.
        #   Differently to the other splits (train, dev), impressions here DO NOT ENCODE IF THE USER CLICKED OR NOT
        #   IN THE IMPRESSION.
        df_test: pd.DataFrame = pd.read_csv(
            filepath_or_buffer=self._original_dataset_test_file,
            **self._pandas_read_csv_kwargs
        )
        df_test["timestamp"] = pd.to_datetime(
            arg=df_test.str_timestamp,
            **self._pandas_to_datetime_kwargs
        )
        df_test["item_id"] = df_test["str_impressions"].progress_map(
            extract_interacted_item_in_impressions,
            na_action=None,  # Our function handles NA, so better to send them to the function.
        )
        df_test["impressions"] = df_test["str_impressions"].progress_map(
            convert_impressions_str_to_array,
            na_action=None,  # Our function handles NA, so better to send them to the function.
        )

        assert df_test.shape[0] == self.num_test_data_points

        return df_test.copy()

    def _download_dataset(self) -> None:
        """ Private function to download the dataset into the project.

        See https://docs.microsoft.com/en-us/azure/open-datasets/dataset-microsoft-news?tabs=azureml-opendatasets
        for instructions to download the dataset. We adapted the functions from there.
        """

        os.makedirs(
            name=self._original_dataset_root_folder,
            exist_ok=True,
        )
        os.makedirs(
            name=self._original_dataset_train_folder,
            exist_ok=True,
        )
        os.makedirs(
            name=self._original_dataset_validation_folder,
            exist_ok=True,
        )
        os.makedirs(
            name=self._original_dataset_test_folder,
            exist_ok=True,
        )

        necessary_file_exist = (
            self._config.variant == MINDVariant.SMALL
            and os.path.exists(self._original_dataset_train_file)
            and os.path.exists(self._original_dataset_validation_file)
        ) or (
            self._config.variant == MINDVariant.LARGE
            and os.path.exists(self._original_dataset_train_file)
            and os.path.exists(self._original_dataset_validation_file)
            and os.path.exists(self._original_dataset_test_file)
        )

        if not necessary_file_exist:
            logger.info(
                "Downloading dataset from the original source."
            )

            # The dataset is split into training and validation set, each with a large and small version.
            # The format of the four files are the same.
            # For demonstration purpose, we will use small version validation set only.
            base_url = self._config.base_url

            remote_filenames = [
                self._config.train_remote_filename,
                self._config.validation_remote_filename,
                self._config.test_remote_filename,
            ]

            local_folders = [
                self._original_dataset_train_folder,
                self._original_dataset_validation_folder,
                self._original_dataset_test_folder,
            ]

            for remote_filename, local_folder in zip(remote_filenames, local_folders):
                if remote_filename == "":
                    continue

                local_zip_path = download_remote_file(
                    url=f"{base_url}/{remote_filename}",
                    destination_filename=os.path.join(
                        self._original_dataset_root_folder, remote_filename,
                    )
                )

                logger.info(
                    f"Extracting {local_zip_path} into {self._original_dataset_train_folder}"
                )
                with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(
                        local_folder,
                    )

                logger.info(
                    f"Deleting temporal ZIP file: {local_zip_path}"
                )
                os.remove(local_zip_path)


class PandasMINDRawData:
    """A class that reads the MINDSmall data using Pandas Dataframes."""

    def __init__(
        self,
        config: MINDSmallConfig,
    ):
        self.config = config
        self.raw_data_loader = MINDRawData(
            config=config
        )

        self._dataset_folder = os.path.join(
            self.config.data_folder, "pandas", "original",
        )
        self.file_interactions = os.path.join(
            self._dataset_folder, "interactions.parquet"
        )
        self.file_interactions_exploded = os.path.join(
            self._dataset_folder, "interactions_exploded.parquet"
        )
        self.file_impressions = os.path.join(
            self._dataset_folder, "impressions.parquet"
        )
        self.file_impressions_exploded = os.path.join(
            self._dataset_folder, "impressions_exploded.parquet"
        )
        self.file_impressions_metadata = os.path.join(
            self._dataset_folder, "impressions_metadata.parquet"
        )

        os.makedirs(
            name=self._dataset_folder,
            exist_ok=True,
        )

    @property  # type: ignore
    @typed_cache
    def _df_train_and_validation(self) -> pd.DataFrame:
        """

        Notes
        -----
        This reader may or not to load the `test` split from MIND-LARGE given that its impressions do not have
        clicked/not-clicked information, i.e., we have impressions but not interactions for users.
        """

        df_train = self.raw_data_loader.train.copy(
            deep=True
        )
        df_validation = self.raw_data_loader.validation.copy(
            deep=True
        )
        df_test = pd.DataFrame(
            columns=df_train.columns
        )

        # We have to update the impression_id values, as they are a RangeIndex for each dataset (i.e., they go from 1
        # to num_records). If we join both datasets, we must have a way to discern the impressions. We do this by
        # shifting the values of the "validation" impressions to higher values (specifically, they begin just after
        # the last record in the interactions_exploded).
        df_validation["impression_id"] += self.config.num_train_data_points

        # Only recreate the impression_id on the LARGE variant of the dataset, as the small one does not have
        # test dataset.
        if self.config.use_test_set:
            df_test = self.raw_data_loader.test.copy(
                deep=True,
            )
            if self.config.variant == MINDVariant.LARGE:
                df_test["impression_id"] += (
                    self.config.num_train_data_points
                    + self.config.num_validation_data_points
                )

        return pd.concat(
            objs=[
                df_train,
                df_validation,
                df_test,
            ],
            axis="index",
            ignore_index=True,  # Re-create the index.
            sort=False,
        )

    @property  # type: ignore
    @typed_cache
    def interactions(self) -> pd.DataFrame:
        """ Interactions Dask Dataframe.

        The columns of the dataframe are:

        user_id : np.int32
        time_step : np.int32
        item_id : np.int32
            A value of 1 indicates no interactions_exploded with any item in the list. A value of 0 indicates error.
        """
        if not os.path.exists(self.file_interactions):
            self._interactions_to_pandas()

        return pd.read_parquet(
            path=self.file_interactions,
            engine=self.config.parquet_engine,
            use_nullable_dtypes=self.config.parquet_use_nullable_dtypes,
        )

    @property  # type: ignore
    @typed_cache
    def interactions_exploded(self) -> pd.DataFrame:
        """ Interactions Dask Dataframe.

        The columns of the dataframe are:

        user_id : np.int32
        time_step : np.int32
        item_id : np.int32
            A value of 1 indicates no interactions_exploded with any item in the list. A value of 0 indicates error.
        """
        if not os.path.exists(self.file_interactions_exploded):
            self._interactions_exploded_to_pandas()

        return pd.read_parquet(
            path=self.file_interactions_exploded,
            engine=self.config.parquet_engine,
            use_nullable_dtypes=self.config.parquet_use_nullable_dtypes,
        )

    @property  # type: ignore
    @typed_cache
    def impressions(self) -> pd.DataFrame:
        if not os.path.exists(self.file_impressions):
            self._impressions_to_pandas()

        return pd.read_parquet(
            path=self.file_impressions,
            engine=self.config.parquet_engine,
            use_nullable_dtypes=self.config.parquet_use_nullable_dtypes,
        )

    @property  # type: ignore
    @typed_cache
    def impressions_exploded(self) -> pd.DataFrame:
        if not os.path.exists(self.file_impressions_exploded):
            self._impressions_exploded_to_pandas()

        return pd.read_parquet(
            path=self.file_impressions_exploded,
            engine=self.config.parquet_engine,
            use_nullable_dtypes=self.config.parquet_use_nullable_dtypes,
        )

    @property  # type: ignore
    @typed_cache
    def interactions_impressions_metadata(self) -> pd.DataFrame:
        if not os.path.exists(self.file_impressions_metadata):
            self._impressions_metadata_to_pandas()

        return pd.read_parquet(
            path=self.file_impressions_metadata,
            engine=self.config.parquet_engine,
            use_nullable_dtypes=self.config.parquet_use_nullable_dtypes,
        )

    @timeit
    def _interactions_to_pandas(self) -> None:
        self._df_train_and_validation[
            ["impression_id", "user_id", "timestamp", "item_id"]
        ].astype(
            {
                "user_id": "category",
            }
        ).to_parquet(
            path=self.file_interactions,
            engine=self.config.parquet_engine,
        )

    @timeit
    def _interactions_exploded_to_pandas(self) -> None:
        self._df_train_and_validation[
            ["impression_id", "user_id", "timestamp", "item_id"]
        ].explode(
            column="item_id"
        ).astype(
            {
                "user_id": "category",
                "item_id": "category",
            }
        ).to_parquet(
            path=self.file_interactions_exploded,
            engine=self.config.parquet_engine,
        )

    @timeit
    def _impressions_to_pandas(self) -> None:
        self._df_train_and_validation[
            ["impression_id", "user_id", "impressions"]
        ].astype(
            {
                "user_id": "category",
            }
        ).to_parquet(
            path=self.file_impressions,
            engine=self.config.parquet_engine,
        )

    @timeit
    def _impressions_exploded_to_pandas(self) -> None:
        self._df_train_and_validation[
            ["impression_id", "user_id", "impressions"]
        ].explode(
            column="impressions"
        ).astype(
            {
                "user_id": "category",
                "impressions": "category",
            }
        ).to_parquet(
            path=self.file_impressions_exploded,
            engine=self.config.parquet_engine,
        )

    @timeit
    def _impressions_metadata_to_pandas(self) -> None:
        metadata = self._df_train_and_validation[
            ["impression_id", "user_id"]
        ].astype(
            {
                "user_id": "category",
            }
        ).copy()

        metadata["num_interacted_items"] = self._df_train_and_validation["item_id"].progress_apply(len)
        metadata["num_impressions"] = self._df_train_and_validation["impressions"].progress_apply(len)
        metadata["position_interactions"] = self._df_train_and_validation["str_impressions"].progress_apply(
            extract_item_positions_in_impressions
        )

        metadata.to_parquet(
            path=self.file_impressions_metadata,
            engine=self.config.parquet_engine,
        )


class MINDReader(BaseDataReader):
    IS_IMPLICIT = True

    _NAME_URM_ALL = "URM_all"
    _NAME_IMPRESSIONS_ALL = "UIM_all"

    _NAME_TIMESTAMP_URM_TRAIN = "URM_timestamp_train"
    _NAME_TIMESTAMP_URM_VALIDATION = "URM_timestamp_validation"
    _NAME_TIMESTAMP_URM_TEST = "URM_timestamp_test"

    _NAME_TIMESTAMP_IMPRESSIONS_TRAIN = "UIM_timestamp_train"
    _NAME_TIMESTAMP_IMPRESSIONS_VALIDATION = "UIM_timestamp_validation"
    _NAME_TIMESTAMP_IMPRESSIONS_TEST = "UIM_timestamp_test"

    _NAME_LEAVE_LAST_K_OUT_URM_TRAIN = "URM_leave_last_k_out_train"
    _NAME_LEAVE_LAST_K_OUT_URM_VALIDATION = "URM_leave_last_k_out_validation"
    _NAME_LEAVE_LAST_K_OUT_URM_TEST = "URM_leave_last_k_out_test"

    _NAME_LEAVE_LAST_K_OUT_IMPRESSIONS_TRAIN = "UIM_leave_last_k_out_train"
    _NAME_LEAVE_LAST_K_OUT_IMPRESSIONS_VALIDATION = "UIM_leave_last_k_out_validation"
    _NAME_LEAVE_LAST_K_OUT_IMPRESSIONS_TEST = "UIM_leave_last_k_out_test"

    def __init__(
        self,
        config: MINDSmallConfig,
    ):
        super().__init__()

        self.config = config

        self.DATA_FOLDER = os.path.join(
            self.config.data_folder, "data_reader", "",
        )

        self.ORIGINAL_SPLIT_FOLDER = self.DATA_FOLDER

        self._DATA_READER_NAME = (
            "MINDSmallReader"
            if self.config.variant == MINDVariant.SMALL
            else "MINDLargeReader"
        )

        self.DATASET_SUBFOLDER = (
            "MIND-SMALL"
            if self.config.variant == MINDVariant.SMALL
            else "MIND-LARGE"
        )

        self._keep_duplicates = self.config.keep_duplicates
        self._min_number_of_interactions = self.config.min_number_of_interactions
        self._binarize_impressions = self.config.binarize_impressions
        self._raw_data_loader = PandasMINDRawData(
            config=config,
        )

        self._user_id_to_index_mapper: dict[int, int] = dict()
        self._item_id_to_index_mapper: dict[int, int] = dict()

        self._urms: dict[str, sp.csr_matrix] = dict()
        self._impressions: dict[str, sp.csr_matrix] = dict()

        self._icms = None
        self._icm_mappers = None

        self._ucms = None
        self._ucms_mappers = None

    @property  # type: ignore
    @typed_cache
    def dataset(self) -> BinaryImplicitDataset:
        return self.load_data(
            save_folder_path=self.ORIGINAL_SPLIT_FOLDER,
        )

    @property  # type: ignore
    @typed_cache
    def _interactions_filtered(self) -> pd.DataFrame:
        df_interactions_exploded = self._raw_data_loader.interactions_exploded.copy()

        # We place the index as part of the dataframe momentarily. All these functions need unique values in the index.
        # Our dataframes do not have unique values in the indices due that they are used as foreign keys/ids for the
        # dataframes.
        df_interactions_exploded = df_interactions_exploded.reset_index(
            drop=False,
        )

        df_interactions_exploded = df_interactions_exploded.sort_values(
            by=["timestamp"],
            ascending=True,
            axis="index",
            inplace=False,
            ignore_index=False,
        )

        df_interactions_exploded, _ = remove_duplicates_in_interactions(
            df=df_interactions_exploded,
            columns_to_compare=["user_id", "item_id"],
            keep=self._keep_duplicates,
        )

        df_interactions_exploded, _ = remove_users_without_min_number_of_interactions(
            df=df_interactions_exploded,
            users_column="user_id",
            min_number_of_interactions=self._min_number_of_interactions,
        )

        # We return the index to its place.
        df_interactions_exploded = df_interactions_exploded.set_index(
            "index",
        )

        return df_interactions_exploded

    @property  # type: ignore
    @typed_cache
    def _impressions_filtered(self) -> pd.DataFrame:
        df_interactions_filtered = self._interactions_filtered

        df_impressions = self._raw_data_loader.impressions.copy()

        df_impressions_filtered, _ = filter_impressions_by_interactions_index(
            df_impressions=df_impressions,
            df_interactions=df_interactions_filtered,
        )

        return df_impressions_filtered

    @property  # type: ignore
    @typed_cache
    def _interactions_timestamp_split(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        interactions_filtered = self._interactions_filtered

        described = interactions_filtered["timestamp"].describe(
            datetime_is_numeric=True,
            percentiles=[0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )

        validation_threshold = described["80%"]
        test_threshold = described["90%"]

        interactions_train, interactions_test = split_sequential_train_test_by_column_threshold(
            df=interactions_filtered,
            column="timestamp",
            threshold=test_threshold
        )

        interactions_train, interactions_validation = split_sequential_train_test_by_column_threshold(
            df=interactions_train,
            column="timestamp",
            threshold=validation_threshold
        )

        return interactions_train.copy(), interactions_validation.copy(), interactions_test.copy()

    @property  # type: ignore
    @typed_cache
    def _interactions_leave_last_k_out_split(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        interactions_filtered = self._interactions_filtered

        # Reset index to avoid messing up with updates based on indices.
        interactions_filtered = interactions_filtered.reset_index(
            drop=False,
        )

        interactions_train, interactions_test = split_sequential_train_test_by_num_records_on_test(
            df=interactions_filtered,
            group_by_column="user_id",
            num_records_in_test=1,
        )

        interactions_train, interactions_validation = split_sequential_train_test_by_num_records_on_test(
            df=interactions_train,
            group_by_column="user_id",
            num_records_in_test=1,
        )

        # Return the index to its place, so we can filter the impressions out.
        interactions_train = interactions_train.set_index("index")
        interactions_validation = interactions_validation.set_index("index")
        interactions_test = interactions_test.set_index("index")

        return interactions_train.copy(), interactions_validation.copy(), interactions_test.copy()

    @property  # type: ignore
    @typed_cache
    def _impressions_timestamp_split(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        interactions_train, interactions_validation, interactions_test = self._interactions_timestamp_split

        impressions_filtered = self._impressions_filtered

        df_impressions_train, _ = filter_impressions_by_interactions_index(
            df_impressions=impressions_filtered,
            df_interactions=interactions_train,
        )

        df_impressions_validation, _ = filter_impressions_by_interactions_index(
            df_impressions=impressions_filtered,
            df_interactions=interactions_validation,
        )

        df_impressions_test, _ = filter_impressions_by_interactions_index(
            df_impressions=impressions_filtered,
            df_interactions=interactions_test,
        )

        return df_impressions_train.copy(), df_impressions_validation.copy(), df_impressions_test.copy()

    @property  # type: ignore
    @typed_cache
    def _impressions_leave_last_k_out_split(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        interactions_train, interactions_validation, interactions_test = self._interactions_leave_last_k_out_split

        impressions_filtered = self._impressions_filtered

        df_impressions_train, _ = filter_impressions_by_interactions_index(
            df_impressions=impressions_filtered,
            df_interactions=interactions_train,
        )

        df_impressions_validation, _ = filter_impressions_by_interactions_index(
            df_impressions=impressions_filtered,
            df_interactions=interactions_validation,
        )

        df_impressions_test, _ = filter_impressions_by_interactions_index(
            df_impressions=impressions_filtered,
            df_interactions=interactions_test,
        )

        return df_impressions_train.copy(), df_impressions_validation.copy(), df_impressions_test.copy()

    def _get_dataset_name_root(self) -> str:
        return self.DATASET_SUBFOLDER

    def _load_from_original_file(self) -> BinaryImplicitDataset:
        # IMPORTANT: calculate first the impressions, so we have all mappers created.
        self._calculate_uim_all()
        self._calculate_urm_all()

        self._calculate_urm_timestamp_splits()
        self._calculate_uim_timestamp_splits()

        self._calculate_urm_leave_last_k_out_splits()
        self._calculate_uim_leave_last_k_out_splits()

        return BinaryImplicitDataset(
            dataset_name="MINDSmall",
            impressions=self._impressions,
            interactions=self._urms,
            mapper_item_original_id_to_index=self._item_id_to_index_mapper,
            mapper_user_original_id_to_index=self._user_id_to_index_mapper,
        )

    def _calculate_urm_all(self):
        logger.info(
            f"Building URM with name {self._NAME_URM_ALL}."
        )

        df_filtered_interactions = self._interactions_filtered

        builder_urm_all = IncrementalSparseMatrix_FilterIDs(
            preinitialized_col_mapper=self._item_id_to_index_mapper,
            on_new_col="add",
            preinitialized_row_mapper=self._user_id_to_index_mapper,
            on_new_row="add"
        )

        users = df_filtered_interactions['user_id'].to_numpy()
        items = df_filtered_interactions['item_id'].to_numpy()
        data = np.ones_like(users, dtype=np.int32,)

        builder_urm_all.add_data_lists(
            row_list_to_add=users,
            col_list_to_add=items,
            data_list_to_add=data,
        )

        self._urms = {
            **self._urms,
            self._NAME_URM_ALL: builder_urm_all.get_SparseMatrix()
        }
        self._user_id_to_index_mapper = builder_urm_all.get_row_token_to_id_mapper()
        self._item_id_to_index_mapper = builder_urm_all.get_column_token_to_id_mapper()

    def _calculate_uim_all(self):
        logger.info(
            f"Building UIM with name {self._NAME_IMPRESSIONS_ALL}."
        )

        df_impressions_filtered = self._impressions_filtered

        builder_impressions_all = IncrementalSparseMatrix_FilterIDs(
            preinitialized_col_mapper=self._item_id_to_index_mapper,
            on_new_col="add",
            preinitialized_row_mapper=self._user_id_to_index_mapper,
            on_new_row="add",
        )

        for _, df_row in tqdm(df_impressions_filtered.iterrows(), total=df_impressions_filtered.shape[0]):
            impressions = np.array(df_row["impressions"], dtype="object")
            users = np.array([df_row["user_id"]] * len(impressions), dtype="object")
            data = np.ones_like(impressions, dtype=np.int32)

            builder_impressions_all.add_data_lists(
                row_list_to_add=users,
                col_list_to_add=impressions,
                data_list_to_add=data,
            )

        uim_all = builder_impressions_all.get_SparseMatrix()
        if self._binarize_impressions:
            uim_all.data = np.ones_like(uim_all.data)

        self._impressions[self._NAME_IMPRESSIONS_ALL] = uim_all.copy()

        self._user_id_to_index_mapper = builder_impressions_all.get_row_token_to_id_mapper()
        self._item_id_to_index_mapper = builder_impressions_all.get_column_token_to_id_mapper()

    def _calculate_urm_leave_last_k_out_splits(self) -> None:
        df_train, df_validation, df_test = self._interactions_leave_last_k_out_split

        names = [
            self._NAME_LEAVE_LAST_K_OUT_URM_TRAIN,
            self._NAME_LEAVE_LAST_K_OUT_URM_VALIDATION,
            self._NAME_LEAVE_LAST_K_OUT_URM_TEST
        ]
        splits = [
            df_train,
            df_validation,
            df_test
        ]

        logger.info(
            f"Building URMs with name {names}."
        )
        for name, df_split in zip(names, splits):
            builder_urm_split = IncrementalSparseMatrix_FilterIDs(
                preinitialized_col_mapper=self._item_id_to_index_mapper,
                on_new_col="ignore",
                preinitialized_row_mapper=self._user_id_to_index_mapper,
                on_new_row="ignore"
            )

            users = df_split['user_id'].to_numpy()
            items = df_split['item_id'].to_numpy()
            data = np.ones_like(users, dtype=np.int32, )

            builder_urm_split.add_data_lists(
                row_list_to_add=users,
                col_list_to_add=items,
                data_list_to_add=data,
            )

            self._urms[name] = builder_urm_split.get_SparseMatrix()

    def _calculate_uim_leave_last_k_out_splits(self) -> None:
        df_train, df_validation, df_test = self._impressions_leave_last_k_out_split

        names = [
            self._NAME_LEAVE_LAST_K_OUT_IMPRESSIONS_TRAIN,
            self._NAME_LEAVE_LAST_K_OUT_IMPRESSIONS_VALIDATION,
            self._NAME_LEAVE_LAST_K_OUT_IMPRESSIONS_TEST
        ]
        splits = [
            df_train,
            df_validation,
            df_test,
        ]

        logger.info(
            f"Building UIMs with name {names}."
        )
        for name, df_split in zip(names, splits):
            builder_uim_split = IncrementalSparseMatrix_FilterIDs(
                preinitialized_col_mapper=self._item_id_to_index_mapper,
                on_new_col="ignore",
                preinitialized_row_mapper=self._user_id_to_index_mapper,
                on_new_row="ignore"
            )

            for _, df_row in tqdm(df_split.iterrows(), total=df_split.shape[0]):
                impressions = np.array(df_row["impressions"], dtype="object")
                users = np.array([df_row["user_id"]] * len(impressions), dtype="object")
                data = np.ones_like(impressions, dtype=np.int32)

                builder_uim_split.add_data_lists(
                    row_list_to_add=users,
                    col_list_to_add=impressions,
                    data_list_to_add=data,
                )

            uim_split = builder_uim_split.get_SparseMatrix()
            if self._binarize_impressions:
                uim_split.data = np.ones_like(uim_split.data)

            self._impressions[name] = uim_split.copy()

    def _calculate_urm_timestamp_splits(self) -> None:
        df_train, df_validation, df_test = self._interactions_leave_last_k_out_split

        names = [
            self._NAME_TIMESTAMP_URM_TRAIN,
            self._NAME_TIMESTAMP_URM_VALIDATION,
            self._NAME_TIMESTAMP_URM_TEST
        ]
        splits = [
            df_train,
            df_validation,
            df_test
        ]

        logger.info(
            f"Building URMs with name {names}."
        )
        for name, df_split in zip(names, splits):

            builder_urm_split = IncrementalSparseMatrix_FilterIDs(
                preinitialized_col_mapper=self._item_id_to_index_mapper,
                on_new_col="ignore",
                preinitialized_row_mapper=self._user_id_to_index_mapper,
                on_new_row="ignore"
            )

            users = df_split['user_id'].to_numpy()
            items = df_split['item_id'].to_numpy()
            data = np.ones_like(users, dtype=np.int32,)

            builder_urm_split.add_data_lists(
                row_list_to_add=users,
                col_list_to_add=items,
                data_list_to_add=data,
            )

            self._urms[name] = builder_urm_split.get_SparseMatrix()

    def _calculate_uim_timestamp_splits(self) -> None:
        df_train, df_validation, df_test = self._impressions_timestamp_split

        names = [
            self._NAME_TIMESTAMP_IMPRESSIONS_TRAIN,
            self._NAME_TIMESTAMP_IMPRESSIONS_VALIDATION,
            self._NAME_TIMESTAMP_IMPRESSIONS_TEST
        ]
        splits = [
            df_train,
            df_validation,
            df_test,
        ]

        logger.info(
            f"Building UIMs with name {names}."
        )
        for name, df_split in zip(names, splits):
            builder_uim_split = IncrementalSparseMatrix_FilterIDs(
                preinitialized_col_mapper=self._item_id_to_index_mapper,
                on_new_col="ignore",
                preinitialized_row_mapper=self._user_id_to_index_mapper,
                on_new_row="ignore"
            )

            for _, df_row in tqdm(df_split.iterrows(), total=df_split.shape[0]):
                impressions = np.array(df_row["impressions"], dtype="object")
                users = np.array([df_row["user_id"]] * len(impressions), dtype="object")
                data = np.ones_like(impressions, dtype=np.int32)

                builder_uim_split.add_data_lists(
                    row_list_to_add=users,
                    col_list_to_add=impressions,
                    data_list_to_add=data,
                )

            uim_split = builder_uim_split.get_SparseMatrix()
            if self._binarize_impressions:
                uim_split.data = np.ones_like(uim_split.data)

            self._impressions[name] = uim_split.copy()


if __name__ == "__main__":
    config = MINDLargeConfig()

    raw_data = MINDRawData(
        config=config,
    )

    pandas_data = PandasMINDRawData(
        config=config,
    )

    all_data_reader = MINDReader(
        config=config,
    )
    all_dataset = all_data_reader.dataset

    import pdb
    pdb.set_trace()
