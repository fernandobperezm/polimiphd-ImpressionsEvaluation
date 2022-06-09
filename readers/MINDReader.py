""" MINDSmallReader.py
This module reads the small or the large version of the Microsoft News ExperimentCase (MIND).

The MIND datasets are a collection of users with their interactions and impressions on a News Aggregator Site (
Microsoft News). In particular, the collection period was between October 12th, 2019 and November 22nd,
2019, i.e., 6 weeks of data collection. In particular, this dataset is a collection of tuples of the form (User ID,
Impression Timestamp, Click History, Impressions).

The dataset already comes with three splits: train, validation, and test. These splits are time-dependent, i.e.,
they are created based on the impression timestamp. Even though the data was collected during a period of 6 weeks,
there are only impressions for the 5th (train and validation) and 6th week (test). For the training and validation
splits, the first 4 weeks the dataset only contain interactions of users (no impression record occurs before the 5th
week). These interactions are placed inside the Click History part of the tuple. For the test split, Click History
contains all interactions that occurred before the 6th week, i.e., from week 1 to week 5 (this means that it covers
the interactions but not the impressions of the training and validation sets).

The original recommendation task is to compute which item the user will click based on any given impression. This
task differs to top-N recommendation as the first can be seen as a classification problem, while the second is mostly
a recommendation task.

Notes
-----
Columns of the dataset
    User ID
        an anonymous ID used to identify and group all interactions and impressions of users in the dataset.
    Impression Timestamp
        is the timestamp in which the given tuple occurred, this is associated with the _impression_ and not the
        interactions that occurred on the impression.
    Impression ID
        MIND ExperimentCase identifier for the impression. THEY DO NOT COME IN ORDER, i.e., a higher impression ID
        does not mean that the impression occurred later. See user "U1000" where impression "86767" comes first than
        impression "46640", so do not rely on impression ids to partition the dataset. Also, partition id is shuffled
        across users, so two consecutive impression ids might refer to different users.
    History
        is an ordered collection of interactions of users. In particular, these are the interactions that happened in
        the first four weeks, but they have no impression associated with them. This collection is ordered, meaning
        that the first element was clicked first than the second. Although an order for the collection is given,
        is not possible to determine in which week these interactions happened.
    Impressions
        A column telling the impressions and possible interactions that users had with that impression. A user
        may have more than one interaction for a given impression. It is stored as a :class:`str` and with the following
        format "NXXXX-Y" where NXXXX is the item id. Y is 0 or 1 and tells if the user interacted or not with the
        item NXXXX in the impression. Differently from Click History, the train and validation splits only contains the
        impressions of the 5th week. The test split only contains the impressions without interaction information (
        i.e., no -Y part) of the 6th week.
"""
import functools
import os
import re
import zipfile
from enum import Enum
from typing import Optional, NamedTuple, Callable

import attrs  # Use the newest API of attrs package.
import numpy as np
import pandas as pd
import scipy.sparse as sp
from recsys_framework_extensions.data.dataset import BaseDataset
from recsys_framework_extensions.data.features import extract_frequency_user_item, extract_last_seen_user_item, \
    extract_position_user_item, extract_timestamp_user_item
from recsys_framework_extensions.data.mixins import ParquetDataMixin, SparseDataMixin, DatasetConfigBackupMixin
from recsys_framework_extensions.data.reader import DataReader
from recsys_framework_extensions.data.sparse import create_sparse_matrix_from_dataframe
from recsys_framework_extensions.data.splitter import (
    remove_duplicates_in_interactions,
    remove_users_without_min_number_of_interactions,
    E_KEEP,
    filter_impressions_by_interactions_index,
    split_sequential_train_test_by_num_records_on_test,
)
from recsys_framework_extensions.decorators import timeit
from recsys_framework_extensions.decorators import typed_cache
from recsys_framework_extensions.evaluation import EvaluationStrategy
from recsys_framework_extensions.hashing.mixins import MixinSHA256Hash
from recsys_framework_extensions.http import download_remote_file
from recsys_framework_extensions.logging import get_logger
from tqdm import tqdm

tqdm.pandas()


logger = get_logger(
    logger_name=__file__,
)


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


class MINDVariant(Enum):
    SMALL = "SMALL"
    LARGE = "LARGE"


class MINDSplits(NamedTuple):
    df_train: pd.DataFrame
    df_validation: pd.DataFrame
    df_train_validation: pd.DataFrame
    df_test: pd.DataFrame


@attrs.define(kw_only=True, frozen=True, slots=False)
class MINDSmallConfig(MixinSHA256Hash):
    """
    Class that holds the configuration used by the different data reader classes to read the raw data, process,
    split, and compute features on it.
    """

    data_folder = os.path.join(
        os.getcwd(), "data", "MIND-SMALL", "",
    )

    first_str_timestamp_of_dataset_collection = "10/12/2019 12:00:01 AM"

    pandas_to_datetime_kwargs = dict(
        errors="raise",
        dayfirst=False,  # The data is in MM/DD, so tell pandas to read dates in that way.
        yearfirst=False,
        utc=None,  # No information about dates being UTC or not.
        format="%m/%d/%Y %I:%M:%S %p",  # Format: MM/dd/YYYY HH:mm:ss AM or MM/dd/YYYY HH:mm:ss PM
        exact=True,
        infer_datetime_format=False,  # Pandas get rids of the AM/PM if it tries to infer the format.
    )

    pandas_dtypes = {
        "user_id": pd.StringDtype(),
    }

    num_train_data_points = 156_965
    num_validation_data_points = 73_152
    num_test_data_points = 0

    base_url = 'https://mind201910small.blob.core.windows.net/release'

    train_remote_filename = 'MINDsmall_train.zip'
    validation_remote_filename = 'MINDsmall_dev.zip'
    test_remote_filename = ''

    min_number_of_interactions: int = attrs.field(
        default=3,
        validator=[
            attrs.validators.gt(0),
        ]
    )
    binarize_impressions: bool = attrs.field(
        default=True,
        validator=[
            attrs.validators.instance_of(bool),
        ]
    )
    binarize_interactions: bool = attrs.field(
        default=True,
        validator=[
            attrs.validators.instance_of(bool),
        ]
    )
    keep_duplicates: E_KEEP = attrs.field(
        default=E_KEEP.FIRST,
        validator=[
            attrs.validators.in_(E_KEEP),  # type: ignore
        ]
    )
    use_historical_interactions: bool = attrs.field(
        default=True,
        validator=[
            attrs.validators.instance_of(bool),
        ]
    )
    # The SMALL variant of the dataset has no test set, therefore it must always be false.
    use_test_set: bool = attrs.field(
        default=False,
        validator=[
            attrs.validators.instance_of(bool),
            attrs.validators.in_([False]),
        ]
    )
    variant: MINDVariant = attrs.field(
        default=MINDVariant.SMALL,
        validator=[
            attrs.validators.instance_of(MINDVariant),
            attrs.validators.in_([MINDVariant.SMALL]),
        ]
    )


@attrs.define(kw_only=True, frozen=True, slots=False)
class MINDLargeConfig(MINDSmallConfig):
    data_folder = os.path.join(
        os.getcwd(), "data", "MIND-LARGE", "",
    )

    num_train_data_points = 2_232_748
    num_validation_data_points = 376_471
    num_test_data_points = 2_370_727

    train_remote_filename = 'MINDlarge_train.zip'
    validation_remote_filename = 'MINDlarge_dev.zip'
    test_remote_filename = 'MINDlarge_test.zip'

    # The LARGE variant can use the test set.
    variant = attrs.field(
        default=MINDVariant.LARGE,
        validator=[
            attrs.validators.instance_of(MINDVariant),
            attrs.validators.in_([MINDVariant.LARGE]),
        ]
    )
    use_test_set = attrs.field(
        default=False,
        validator=[
            attrs.validators.instance_of(bool),
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


class PandasMINDRawData(ParquetDataMixin):
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
        self._file_data = os.path.join(
            self._dataset_folder, "data.parquet"
        )
        self._file_previous_interactions = os.path.join(
            self._dataset_folder, "previous_interactions.parquet"
        )

        os.makedirs(
            name=self._dataset_folder,
            exist_ok=True,
        )

    @property  # type: ignore
    @typed_cache
    def data(self) -> pd.DataFrame:
        """ Interactions Dask Dataframe.

        The columns of the dataframe are:

        user_id : np.int32
        time_step : np.int32
        item_id : np.int32
            A value of 1 indicates no interactions_exploded with any item in the list. A value of 0 indicates error.
        """
        return self.load_parquet(
            to_pandas_func=self._data_to_pandas,
            file_path=self._file_data,
        )

    @property  # type: ignore
    @typed_cache
    def previous_interactions(self) -> pd.DataFrame:
        """ Interactions Dask Dataframe.

        The columns of the dataframe are:

        user_id : np.int32
        time_step : np.int32
        item_id : np.int32
            A value of 1 indicates no interactions_exploded with any item in the list. A value of 0 indicates error.
        """
        return self.load_parquet(
            to_pandas_func=self._previous_interactions_to_pandas,
            file_path=self._file_previous_interactions,
        )

    @timeit
    def _data_to_pandas(self) -> pd.DataFrame:
        # Impression_id is just an index to refer to any row, even if there were repeated impressions, these are not
        # identified by impression_id, as this is merely an index for the dataframe. To ease working with the dataset
        # later, we set the index momentarily and then reset it again.
        df_train = self.raw_data_loader.train.set_index(
            "impression_id",
        )
        df_validation = self.raw_data_loader.validation.set_index(
            "impression_id",
        )
        df_test = self.raw_data_loader.test.set_index(
            "impression_id",
        )

        df_data = pd.concat(
            objs=[
                df_train,
                df_validation,
                df_test,
            ],
            axis="index",
            ignore_index=True,  # Re-create the index.
            sort=False,
        )

        df_data["timestamp"] = pd.to_datetime(
            arg=df_data.str_timestamp,
            **self.config.pandas_to_datetime_kwargs,
        )
        df_data["item_ids"] = df_data["str_impressions"].progress_map(
            extract_interacted_item_in_impressions,
            na_action=None,  # Our function handles NA, so better to send them to the function.
        )
        df_data["impressions"] = df_data["str_impressions"].progress_map(
            convert_impressions_str_to_array,
            na_action=None,  # Our function handles NA, so better to send them to the function.
        )
        df_data["num_interacted_items"] = df_data["item_ids"].progress_apply(
            len
        )
        df_data["num_impressions"] = df_data["impressions"].progress_apply(
            len
        )
        df_data["position_interactions"] = df_data["str_impressions"].progress_apply(
            extract_item_positions_in_impressions
        )

        return df_data[
            ["timestamp", "user_id", "item_ids", "impressions", "num_interacted_items", "num_impressions", "position_interactions"]
        ]

    @timeit
    def _previous_interactions_to_pandas(self) -> pd.DataFrame:
        # Impression_id is just an index to refer to any row, even if there were repeated impressions, these are not
        # identified by impression_id, as this is merely an index for the dataframe. To ease working with the dataset
        # later, we set the index momentarily and then reset it again.
        df_train = self.raw_data_loader.train.set_index(
            "impression_id",
        )[["user_id", "str_history"]]

        df_validation = self.raw_data_loader.validation.set_index(
            "impression_id",
        )[["user_id", "str_history"]]

        # The historical interactions are unique for the train and validation splits. The train splits contains
        # impressions from week 5 of the dataset (Saturday to Thursday), and the validation contains the impressions
        # of the last day of week 5 (Friday).
        #
        # This code reads the historical interactions, asserts that they are unique, and places them in condensed
        # form in the dataset at the beginning of it, with the timestamp of the first day of the dataset (week 1
        # at 0:00:00).
        #
        # Notes: If you decide to load the test set from MIND Large, then the assertion will fail, given that the
        # click history in the test split contains the interactions happened between week 1 and week 5 in the
        # dataset, opposed to the train/validation splits which only contain interactions between weeks 1 to 4.

        # This dataframe contains the counts of user-str_history pairs and how many times they appear in the
        # dataset. In particular, we're interested to verify that the dataset that every user has only one
        # str_history associated with them, as per the description of the dataset in the paper.
        historical_data_per_user = pd.concat(
            objs=[
                df_train,
                df_validation,
            ],
            axis="index",
            ignore_index=True,  # Re-create the index.
            sort=False,
        )

        previous_history = historical_data_per_user.value_counts(
            subset=["user_id", "str_history"],
            sort=False,
            ascending=False,
        ).to_frame(

        ).reset_index(
            drop=False
        )

        num_unique_previous_interactions_per_user = previous_history.value_counts(
            subset=["user_id"],
            sort=False,
            ascending=False,
        )

        # This assertion ensures that all users only have 1 record of str_history.
        all_users_have_unique_previous_interactions = np.all(num_unique_previous_interactions_per_user == 1)
        if not all_users_have_unique_previous_interactions:
            raise ValueError(
                f"Not all users have unique historical interactions, this probably means that you included the "
                f"test set of the MIND Large dataset. Be aware that the test set contains the historical "
                f"interactions between weeks 1 to 5 of the dataset. The train/validation splits only contain "
                f"historical interactions between weeks 1 to 4."
                f"\n If this is expected, then comment this if clause."
            )

        df_historical_interactions = pd.DataFrame(
            data=None,
        )
        df_historical_interactions["user_id"] = previous_history["user_id"].astype(dtype=pd.StringDtype())
        df_historical_interactions["timestamp"] = pd.to_datetime(
            arg=self.config.first_str_timestamp_of_dataset_collection,
            **self.config.pandas_to_datetime_kwargs,
        )
        df_historical_interactions["item_ids"] = previous_history["str_history"].progress_map(
            convert_impressions_str_to_array,
            na_action=None,  # Our function handles NA, so better to send them to the function.
        )
        df_historical_interactions["num_interacted_items"] = df_historical_interactions["item_ids"].progress_apply(
            len
        )

        return df_historical_interactions


class PandasMINDProcessedData(ParquetDataMixin, DatasetConfigBackupMixin):
    def __init__(
        self,
        config: MINDSmallConfig,
    ):
        self.config = config
        self.config_hash = config.sha256_hash

        self.pandas_raw_data = PandasMINDRawData(
            config=config,
        )

        self._folder_dataset = os.path.join(
            self.config.data_folder, "data-processing", self.config_hash, ""
        )

        self._folder_leave_last_out = os.path.join(
            self._folder_dataset, "leave-last-out", ""
        )

        self._folder_timestamp = os.path.join(
            self._folder_dataset, "timestamp", ""
        )

        self._file_path_filtered_data = os.path.join(
            self._folder_dataset, "filter_data.parquet"
        )

        self._filename_train = "train.parquet"
        self._filename_validation = "validation.parquet"
        self._filename_train_validation = "train_validation.parquet"
        self._filename_test = "test.parquet"

        os.makedirs(
            name=self._folder_dataset,
            exist_ok=True,
        )
        os.makedirs(
            name=self._folder_leave_last_out,
            exist_ok=True,
        )

    def backup_config(self) -> None:
        self.save_config(
            config=self.config,
            folder_path=self._folder_dataset,
        )

    @property  # type: ignore
    def filtered(self) -> pd.DataFrame:
        return self.load_parquet(
            file_path=self._file_path_filtered_data,
            to_pandas_func=self._filtered_to_pandas,
        ).astype(
            dtype=self.config.pandas_dtypes,
        )

    @property  # type: ignore
    def leave_last_out_splits(
        self
    ) -> MINDSplits:
        file_paths = [
            os.path.join(self._folder_leave_last_out, self._filename_train),
            os.path.join(self._folder_leave_last_out, self._filename_validation),
            os.path.join(self._folder_leave_last_out, self._filename_train_validation),
            os.path.join(self._folder_leave_last_out, self._filename_test),
        ]

        df_train, df_validation, df_train_validation, df_test = self.load_parquets(
            file_paths=file_paths,
            to_pandas_func=self._leave_last_out_splits_to_pandas,
        )

        df_train = df_train.astype(dtype=self.config.pandas_dtypes)
        df_validation = df_validation.astype(dtype=self.config.pandas_dtypes)
        df_train_validation = df_train_validation.astype(dtype=self.config.pandas_dtypes)
        df_test = df_test.astype(dtype=self.config.pandas_dtypes)

        assert df_train.shape[0] > 0
        assert df_validation.shape[0] > 0
        assert df_train_validation.shape[0] > 0
        assert df_test.shape[0] > 0

        return MINDSplits(
            df_train=df_train,
            df_validation=df_validation,
            df_train_validation=df_train_validation,
            df_test=df_test,
        )

    @property
    def dataframes(self) -> dict[str, pd.DataFrame]:
        return {
            BaseDataset.NAME_DF_FILTERED: self.filtered,

            BaseDataset.NAME_DF_LEAVE_LAST_K_OUT_TRAIN: self.leave_last_out_splits.df_train,
            BaseDataset.NAME_DF_LEAVE_LAST_K_OUT_VALIDATION: self.leave_last_out_splits.df_validation,
            BaseDataset.NAME_DF_LEAVE_LAST_K_OUT_TRAIN_VALIDATION: self.leave_last_out_splits.df_train_validation,
            BaseDataset.NAME_DF_LEAVE_LAST_K_OUT_TEST: self.leave_last_out_splits.df_test,
        }

    def _filtered_to_pandas(self) -> pd.DataFrame:
        """

        Notes
        -----
        The main dataframe, to which all filters are applied, is `df_interactions` given that
        `df_interactions_condensed` as the interactions in a "condensed" format (user_id, list[item_id]) instead of
        tuple format (user_id, item_id). `df_interaction` has interactions in tuple format.

        The main issue with this is that `df_interaction` *does not have unique indices*, given that it is
        an exploded version of `df_interactions_condensed`.

        What we do to avoid filtering problems is that we first reset the index of `df_interactions_exploded` without
        dropping the index column, then we apply all filters to it, and then set the index again to be the
        previously-reset index column.

        After this, we can ensure that the set of indices values are the same across the three datasets and when we
        filter datasets by their indices we are sure that we're doing the filtering correctly.
        """

        logger.info(
            f"Filtering data sources (interactions, impressions, metadata)."
        )

        df_data = self.pandas_raw_data.data

        if self.config.use_historical_interactions:
            df_previous_interactions = self.pandas_raw_data.previous_interactions
            df_data = pd.concat(
                objs=[
                    df_previous_interactions,
                    df_data,
                ],
                axis="index",
                sort=False,
                ignore_index=True,
            )

        df_data = df_data.sort_values(
            by=["timestamp"],
            ascending=True,
            axis="index",
            inplace=False,
            ignore_index=False,
        )

        df_exploded_interactions = df_data[
            ["timestamp", "user_id", "item_ids"]
        ].explode(
            column="item_ids",
            ignore_index=False,
        ).dropna(
            axis="index",
            how="any",
            inplace=False,
        ).reset_index(
            drop=False,
        )

        df_exploded_interactions, _ = remove_duplicates_in_interactions(
            df=df_exploded_interactions,
            columns_to_compare=["user_id", "item_ids"],
            keep=self.config.keep_duplicates,
        )

        df_exploded_interactions, _ = remove_users_without_min_number_of_interactions(
            df=df_exploded_interactions,
            users_column="user_id",
            min_number_of_interactions=self.config.min_number_of_interactions,
        )

        df_exploded_interactions = df_exploded_interactions.set_index(
            "index",
        )

        df_data, _ = filter_impressions_by_interactions_index(
            df_impressions=df_data,
            df_interactions=df_exploded_interactions,
        )

        return df_data

    def _leave_last_out_splits_to_pandas(self) -> list[pd.DataFrame]:
        df_data_filtered = self.filtered

        df_data_exploded_filtered = df_data_filtered[
            ["timestamp", "user_id", "item_ids"]
        ].explode(
            column="item_ids",
            ignore_index=False,
        ).dropna(
            axis="index",
            how="any",
            inplace=False,
        ).reset_index(
            drop=False,
        )

        df_data_train_validation, df_data_test = split_sequential_train_test_by_num_records_on_test(
            df=df_data_exploded_filtered,
            group_by_column="user_id",
            num_records_in_test=1,
        )

        df_data_train, df_data_validation = split_sequential_train_test_by_num_records_on_test(
            df=df_data_train_validation,
            group_by_column="user_id",
            num_records_in_test=1,
        )

        df_data_train = df_data_train.set_index(
            "index",
        )
        df_data_validation = df_data_validation.set_index(
            "index",
        )
        df_data_train_validation = df_data_train_validation.set_index(
            "index",
        )
        df_data_test = df_data_test.set_index(
            "index",
        )

        df_data_train, _ = filter_impressions_by_interactions_index(
            df_impressions=df_data_filtered,
            df_interactions=df_data_train,
        )
        df_data_validation, _ = filter_impressions_by_interactions_index(
            df_impressions=df_data_filtered,
            df_interactions=df_data_validation,
        )
        df_data_train_validation, _ = filter_impressions_by_interactions_index(
            df_impressions=df_data_filtered,
            df_interactions=df_data_train_validation,
        )
        df_data_test, _ = filter_impressions_by_interactions_index(
            df_impressions=df_data_filtered,
            df_interactions=df_data_test,
        )

        return [df_data_train, df_data_validation, df_data_train_validation, df_data_test]


class PandasMINDImpressionsFeaturesData(ParquetDataMixin, DatasetConfigBackupMixin):
    def __init__(
        self,
        config: MINDSmallConfig,
    ):
        self.config = config
        self.loader_processed_data = PandasMINDProcessedData(
            config=config,
        )

        self._folder_dataset = os.path.join(
            self.config.data_folder, "data-features-impressions", self.config.sha256_hash, ""
        )

        self._folder_leave_last_k_out = os.path.join(
            self._folder_dataset, "leave-last-k-out", ""
        )

        self._file_name_split_train = "train.parquet"
        self._file_name_split_validation = "validation.parquet"
        self._file_name_split_train_validation = "train_validation.parquet"
        self._file_name_split_test = "test.parquet"

        self._feature_funcs: dict[str, Callable[..., tuple[pd.DataFrame, pd.DataFrame]]] = {
            "user_item_frequency": functools.partial(
                extract_frequency_user_item,
                users_column="user_id",
                items_column="impressions",
            ),
            "user_item_last_seen": functools.partial(
                extract_last_seen_user_item,
                users_column="user_id",
                items_column="impressions",
                timestamp_column="timestamp",
            ),
            "user_item_position": functools.partial(
                extract_position_user_item,
                users_column="user_id",
                items_column="impressions",
                positions_column=None,
                to_keep="last",
            ),
            "user_item_timestamp": functools.partial(
                extract_timestamp_user_item,
                users_column="user_id",
                items_column="impressions",
                timestamp_column="timestamp",
                to_keep="last",
            )
        }

        os.makedirs(
            name=self._folder_dataset,
            exist_ok=True,
        )
        os.makedirs(
            name=self._folder_leave_last_k_out,
            exist_ok=True,
        )

        for folder in [self._folder_leave_last_k_out]:
            for sub_folder in self._feature_funcs.keys():
                os.makedirs(
                    name=os.path.join(folder, sub_folder),
                    exist_ok=True,
                )

    def backup_config(self) -> None:
        self.save_config(
            config=self.config,
            folder_path=self._folder_dataset,
        )

    @property
    def features(self) -> dict[str, list[str]]:
        return {
            "user_item_frequency": ["feature-user_id-impressions-frequency"],
            "user_item_last_seen": [
                "feature_last_seen_total_seconds",
                "feature_last_seen_total_minutes",
                "feature_last_seen_total_hours",
                "feature_last_seen_total_days",
                "feature_last_seen_total_weeks",
            ],
            "user_item_position": ["feature-user_id-impressions-position"],
            "user_item_timestamp": ["feature-user_id-impressions-timestamp"],
        }

    @property
    def impressions_features(self) -> dict[str, pd.DataFrame]:
        impression_features: dict[str, pd.DataFrame] = {}

        for evaluation_strategy in [EvaluationStrategy.LEAVE_LAST_K_OUT]:
            for feature_key, feature_columns in self.features.items():
                splits = self.user_item_feature(
                    evaluation_strategy=evaluation_strategy,
                    feature_key=feature_key
                )

                feature_name = f"{evaluation_strategy.value}-{feature_key}"
                impression_features[f"{feature_name}-train"] = splits.df_train
                impression_features[f"{feature_name}-validation"] = splits.df_validation
                impression_features[f"{feature_name}-train_validation"] = splits.df_train_validation
                impression_features[f"{feature_name}-test"] = splits.df_test

        return impression_features

    def user_item_feature(
        self,
        evaluation_strategy: EvaluationStrategy,
        feature_key: str,
    ) -> MINDSplits:
        logger.debug(
            f"Called {self.user_item_feature.__name__} with kwargs: {evaluation_strategy=} - {feature_key=}"
        )
        assert feature_key in self._feature_funcs

        file_paths = self._get_file_paths_by_evaluation_strategy(
            evaluation_strategy=evaluation_strategy,
            feature=feature_key,
        )

        partial_func_user_item_position_to_pandas = functools.partial(
            self._user_item_feature_to_pandas,
            evaluation_strategy=evaluation_strategy,
            feature_key=feature_key,
        )

        df_train: pd.DataFrame
        df_validation: pd.DataFrame
        df_train_validation: pd.DataFrame
        df_test: pd.DataFrame

        df_train, df_validation, df_train_validation, df_test = self.load_parquets(
            file_paths=file_paths,
            to_pandas_func=partial_func_user_item_position_to_pandas,
        )

        return MINDSplits(
            df_train=df_train,
            df_validation=df_validation,
            df_train_validation=df_train_validation,
            df_test=df_test,
        )

    def _get_file_paths_by_evaluation_strategy(
        self,
        evaluation_strategy: EvaluationStrategy,
        feature: str,
    ) -> list[str]:
        if EvaluationStrategy.LEAVE_LAST_K_OUT == evaluation_strategy:
            folder_to_look_up = self._folder_leave_last_k_out
        else:
            raise NotImplementedError(
                f"The only evaluation strategy implemented for the MIND dataset is "
                f"{EvaluationStrategy.LEAVE_LAST_K_OUT}. Passed evaluation strategy was {evaluation_strategy}"
            )

        return [
            os.path.join(
                folder_to_look_up, feature, self._file_name_split_train,
            ),
            os.path.join(
                folder_to_look_up, feature, self._file_name_split_validation,
            ),
            os.path.join(
                folder_to_look_up, feature, self._file_name_split_train_validation,
            ),
            os.path.join(
                folder_to_look_up, feature, self._file_name_split_test,
            ),
        ]

    def _get_splits_by_evaluation_strategy(
        self,
        evaluation_strategy: EvaluationStrategy,
    ) -> MINDSplits:
        if EvaluationStrategy.LEAVE_LAST_K_OUT == evaluation_strategy:
            return self.loader_processed_data.leave_last_out_splits
        else:
            raise NotImplementedError(
                f"The only evaluation strategy implemented for the MIND dataset is "
                f"{EvaluationStrategy.LEAVE_LAST_K_OUT}. Passed evaluation strategy was {evaluation_strategy}"
            )

    def _user_item_feature_to_pandas(
        self,
        evaluation_strategy: EvaluationStrategy,
        feature_key: str
    ) -> list[pd.DataFrame]:
        assert feature_key in self._feature_funcs

        logger.debug(
            f"Called {self._user_item_feature_to_pandas.__name__} with kwargs: "
            f" {feature_key=} - {evaluation_strategy=} -"
        )

        splits = self._get_splits_by_evaluation_strategy(
            evaluation_strategy=evaluation_strategy,
        )

        feature_func = self._feature_funcs[feature_key]

        df_train_user_item_feature, _ = feature_func(df=splits.df_train)
        df_validation_user_item_feature, _ = feature_func(df=splits.df_validation)
        df_train_validation_user_item_feature, _ = feature_func(df=splits.df_train_validation)
        df_test_user_item_feature, _ = feature_func(df=splits.df_test)

        return [
            df_train_user_item_feature,
            df_validation_user_item_feature,
            df_train_validation_user_item_feature,
            df_test_user_item_feature,
        ]


class SparsePandasMINDData(SparseDataMixin, ParquetDataMixin, DatasetConfigBackupMixin):
    def __init__(
        self,
        config: MINDSmallConfig,
    ):
        super().__init__()

        self.config = config

        self.data_loader_processed = PandasMINDProcessedData(
            config=config,
        )
        self.data_loader_impression_features = PandasMINDImpressionsFeaturesData(
            config=config,
        )

        self.users_column = "user_id"
        self.items_column = "item_ids"
        self.impressions_column = "impressions"

        self._folder_data = os.path.join(
            self.config.data_folder, "data-sparse", self.config.variant.value, self.config.sha256_hash, "",
        )
        self._folder_leave_last_out_data = os.path.join(
            self._folder_data, EvaluationStrategy.LEAVE_LAST_K_OUT.value, "",
        )

        self._file_path_item_mapper = os.path.join(self._folder_data, "item_mapper.parquet")
        self._file_path_user_mapper = os.path.join(self._folder_data, "user_mapper.parquet")

        self._file_path_urm_all = os.path.join(self._folder_data, "urm_all.npz")
        self._file_paths_leave_last_out_urms = [
            os.path.join(self._folder_leave_last_out_data, "urm_train.npz"),
            os.path.join(self._folder_leave_last_out_data, "urm_validation.npz"),
            os.path.join(self._folder_leave_last_out_data, "urm_train_validation.npz"),
            os.path.join(self._folder_leave_last_out_data, "urm_test.npz"),
        ]

        self._file_path_uim_all = os.path.join(self._folder_data, "uim_all.npz")
        self._file_paths_leave_last_out_uims = [
            os.path.join(self._folder_leave_last_out_data, "uim_train.npz"),
            os.path.join(self._folder_leave_last_out_data, "uim_validation.npz"),
            os.path.join(self._folder_leave_last_out_data, "uim_train_validation.npz"),
            os.path.join(self._folder_leave_last_out_data, "uim_test.npz"),
        ]

        os.makedirs(self._folder_data, exist_ok=True)
        os.makedirs(self._folder_leave_last_out_data, exist_ok=True)

    def backup_config(self) -> None:
        self.save_config(
            config=self.config,
            folder_path=self._folder_data,
        )

    @functools.cached_property
    def mapper_user_id_to_index(self) -> dict[str, int]:
        df_user_mapper = self.load_parquet(
            file_path=self._file_path_user_mapper,
            to_pandas_func=self._user_mapper_to_pandas,
        )

        return {
            str(df_row.orig_value): int(df_row.mapped_value)
            for df_row in df_user_mapper.itertuples(index=False)
        }

    @functools.cached_property
    def mapper_item_id_to_index(self) -> dict[str, int]:
        df_item_mapper = self.load_parquet(
            file_path=self._file_path_item_mapper,
            to_pandas_func=self._item_mapper_to_pandas,
        )

        return {
            str(df_row.orig_value): int(df_row.mapped_value)
            for df_row in df_item_mapper.itertuples(index=False)
        }

    @property
    def interactions(self) -> dict[str, sp.csr_matrix]:
        sp_urm_all = self.load_sparse_matrix(
            file_path=self._file_path_urm_all,
            to_sparse_matrix_func=self._urm_all_to_sparse,
        )

        (
            sp_llo_urm_train,
            sp_llo_urm_validation,
            sp_llo_urm_train_validation,
            sp_llo_urm_test,
        ) = self.load_sparse_matrices(
            file_paths=self._file_paths_leave_last_out_urms,
            to_sparse_matrices_func=self._urms_leave_last_out_to_sparse,
        )

        return {
            BaseDataset.NAME_URM_ALL: sp_urm_all,

            BaseDataset.NAME_URM_LEAVE_LAST_K_OUT_TRAIN: sp_llo_urm_train,
            BaseDataset.NAME_URM_LEAVE_LAST_K_OUT_VALIDATION: sp_llo_urm_validation,
            BaseDataset.NAME_URM_LEAVE_LAST_K_OUT_TRAIN_VALIDATION: sp_llo_urm_train_validation,
            BaseDataset.NAME_URM_LEAVE_LAST_K_OUT_TEST: sp_llo_urm_test,
        }

    @property
    def impressions(self) -> dict[str, sp.csr_matrix]:
        sp_uim_all = self.load_sparse_matrix(
            file_path=self._file_path_uim_all,
            to_sparse_matrix_func=self._uim_all_to_sparse,
        )

        (
            sp_llo_uim_train,
            sp_llo_uim_validation,
            sp_llo_uim_train_validation,
            sp_llo_uim_test,
        ) = self.load_sparse_matrices(
            file_paths=self._file_paths_leave_last_out_uims,
            to_sparse_matrices_func=self._uims_leave_last_out_to_sparse,
        )

        return {
            BaseDataset.NAME_UIM_ALL: sp_uim_all,

            BaseDataset.NAME_UIM_LEAVE_LAST_K_OUT_TRAIN: sp_llo_uim_train,
            BaseDataset.NAME_UIM_LEAVE_LAST_K_OUT_VALIDATION: sp_llo_uim_validation,
            BaseDataset.NAME_UIM_LEAVE_LAST_K_OUT_TRAIN_VALIDATION: sp_llo_uim_train_validation,
            BaseDataset.NAME_UIM_LEAVE_LAST_K_OUT_TEST: sp_llo_uim_test,
        }

    @property
    def impressions_features(self) -> dict[str, sp.csr_matrix]:
        impression_features: dict[str, sp.csr_matrix] = {}

        for evaluation_strategy in [EvaluationStrategy.LEAVE_LAST_K_OUT]:
            for feature_key, feature_columns in self.data_loader_impression_features.features.items():
                for feature_column in feature_columns:
                    folder = os.path.join(self._folder_data, evaluation_strategy.value)
                    file_paths = [
                        os.path.join(
                            folder, f"impressions_features_{feature_key}_{feature_column}_train.npz"
                        ),
                        os.path.join(
                            folder, f"impressions_features_{feature_key}_{feature_column}_validation.npz"
                        ),
                        os.path.join(
                            folder, f"impressions_features_{feature_key}_{feature_column}_train_validation.npz"
                        ),
                        os.path.join(
                            folder, f"impressions_features_{feature_key}_{feature_column}_test.npz"
                        ),
                    ]

                    partial_func = functools.partial(
                        self._impression_features_to_sparse,
                        evaluation_strategy=evaluation_strategy,
                        feature_key=feature_key,
                        feature_column=feature_column
                    )

                    (
                        sp_impressions_feature_train,
                        sp_impressions_feature_validation,
                        sp_impressions_feature_train_validation,
                        sp_impressions_feature_test,
                    ) = self.load_sparse_matrices(
                        file_paths=file_paths,
                        to_sparse_matrices_func=partial_func,
                    )

                    feature_name = f"{evaluation_strategy.value}-{feature_key}-{feature_column}"
                    impression_features[f"{feature_name}-train"] = sp_impressions_feature_train
                    impression_features[f"{feature_name}-validation"] = sp_impressions_feature_validation
                    impression_features[f"{feature_name}-train_validation"] = sp_impressions_feature_train_validation
                    impression_features[f"{feature_name}-test"] = sp_impressions_feature_test

        return impression_features

    def _item_mapper_to_pandas(self) -> pd.DataFrame:
        df_data_filtered = self.data_loader_processed.filtered[
            [self.items_column, self.impressions_column]
        ]

        non_na_exploded_items = df_data_filtered[self.items_column].explode(
            ignore_index=True,
        ).dropna(
            inplace=False,
        )

        non_na_exploded_impressions = df_data_filtered[self.impressions_column].explode(
            ignore_index=True,
        ).dropna(
            inplace=False,
        )

        unique_items = set(non_na_exploded_items).union(non_na_exploded_impressions)

        return pd.DataFrame.from_records(
            data=[
                (int(mapped_value), str(orig_value))
                for mapped_value, orig_value in enumerate(unique_items)
            ],
            columns=["mapped_value", "orig_value"]
        )

    def _user_mapper_to_pandas(self) -> pd.DataFrame:
        df_data_filtered = self.data_loader_processed.filtered

        non_na_users = df_data_filtered[self.users_column].dropna(
            inplace=False,
        )

        unique_users = set(non_na_users)

        return pd.DataFrame.from_records(
            data=[
                (int(mapped_value), str(orig_value))
                for mapped_value, orig_value in enumerate(unique_users)
            ],
            columns=["mapped_value", "orig_value"]
        )

    def _urm_all_to_sparse(self) -> sp.csr_matrix:
        df_data_filtered = self.data_loader_processed.filtered[
            [self.users_column, self.items_column]
        ]

        urm_all = create_sparse_matrix_from_dataframe(
            df=df_data_filtered,
            users_column=self.users_column,
            items_column=self.items_column,
            binarize_interactions=self.config.binarize_interactions,
            mapper_user_id_to_index=self.mapper_user_id_to_index,
            mapper_item_id_to_index=self.mapper_item_id_to_index,
        )

        return urm_all

    def _urms_leave_last_out_to_sparse(self) -> list[sp.csr_matrix]:
        df_train, df_validation, df_train_validation, df_test = self.data_loader_processed.leave_last_out_splits

        sparse_matrices = []
        for df_split in [
            df_train,
            df_validation,
            df_train_validation,
            df_test,
        ]:
            urm_split = create_sparse_matrix_from_dataframe(
                df=df_split,
                users_column=self.users_column,
                items_column=self.items_column,
                binarize_interactions=self.config.binarize_interactions,
                mapper_user_id_to_index=self.mapper_user_id_to_index,
                mapper_item_id_to_index=self.mapper_item_id_to_index,
            )

            sparse_matrices.append(urm_split)

        return sparse_matrices

    def _uim_all_to_sparse(self) -> sp.csr_matrix:
        uim_all = create_sparse_matrix_from_dataframe(
            df=self.data_loader_processed.filtered,
            users_column=self.users_column,
            items_column=self.impressions_column,
            binarize_interactions=self.config.binarize_impressions,
            mapper_user_id_to_index=self.mapper_user_id_to_index,
            mapper_item_id_to_index=self.mapper_item_id_to_index,
        )
        return uim_all

    def _uims_leave_last_out_to_sparse(self) -> list[sp.csr_matrix]:
        df_train, df_validation, df_train_validation, df_test = self.data_loader_processed.leave_last_out_splits

        sparse_matrices = []
        for df_split in [
            df_train,
            df_validation,
            df_train_validation,
            df_test,
        ]:
            uim_split = create_sparse_matrix_from_dataframe(
                df=df_split,
                users_column=self.users_column,
                items_column=self.impressions_column,
                binarize_interactions=self.config.binarize_impressions,
                mapper_user_id_to_index=self.mapper_user_id_to_index,
                mapper_item_id_to_index=self.mapper_item_id_to_index,
            )

            sparse_matrices.append(uim_split)

        return sparse_matrices

    def _impression_features_to_sparse(
        self, evaluation_strategy: EvaluationStrategy, feature_key: str, feature_column: str
    ) -> list[sp.csr_matrix]:

        sparse_matrices = []

        splits = self.data_loader_impression_features.user_item_feature(
            evaluation_strategy=evaluation_strategy,
            feature_key=feature_key,
        )

        for split_name, df_split in splits._asdict().items():
            feature_name = f"{evaluation_strategy.value}-{split_name}-{feature_key}-{feature_column}"

            logger.debug(
                f"\n* {evaluation_strategy=}"
                f"\n* {feature_key=}"
                f"\n* {feature_column=}"
                f"\n* {split_name=}"
                f"\n* {df_split=}"
                f"\n* {df_split.columns=}"
                f"\n* {feature_name=}"
            )
            sparse_matrix_split = create_sparse_matrix_from_dataframe(
                df=df_split,
                users_column=self.users_column,
                items_column=self.impressions_column,
                data_column=feature_column,
                binarize_interactions=False,
                mapper_user_id_to_index=self.mapper_user_id_to_index,
                mapper_item_id_to_index=self.mapper_item_id_to_index,
            )

            sparse_matrices.append(sparse_matrix_split)

        return sparse_matrices


class MINDReader(DatasetConfigBackupMixin, DataReader):
    def __init__(
        self,
        config: MINDSmallConfig,
    ):
        super().__init__()

        self.config = config
        self.config_hash = config.sha256_hash

        self.data_loader_processed = PandasMINDProcessedData(
            config=config,
        )
        self.data_loader_sparse_data = SparsePandasMINDData(
            config=config,
        )
        self.data_loader_impressions_features = PandasMINDImpressionsFeaturesData(
            config=config,
        )

        self.DATA_FOLDER = os.path.join(
            self.config.data_folder, "data_reader", self.config_hash, "",
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

        self.IS_IMPLICIT = self.config.binarize_interactions

    def backup_config(self) -> None:
        self.save_config(
            config=self.config,
            folder_path=self.DATA_FOLDER,
        )

    @property  # type: ignore
    @typed_cache
    def dataset(self) -> BaseDataset:
        return self.load_data(
            save_folder_path=self.ORIGINAL_SPLIT_FOLDER,
        )

    def _get_dataset_name_root(self) -> str:
        return self.DATASET_SUBFOLDER

    def _load_from_original_file(self) -> BaseDataset:
        dataframes = self.data_loader_processed.dataframes

        mapper_user_original_id_to_index = self.data_loader_sparse_data.mapper_user_id_to_index
        mapper_item_original_id_to_index = self.data_loader_sparse_data.mapper_item_id_to_index

        interactions = self.data_loader_sparse_data.interactions
        impressions = self.data_loader_sparse_data.impressions
        impressions_features_sparse_matrices = self.data_loader_sparse_data.impressions_features
        impressions_features_dataframes = self.data_loader_impressions_features.impressions_features

        # backup all configs that created this dataset.
        self.data_loader_processed.backup_config()
        self.data_loader_impressions_features.backup_config()
        self.data_loader_sparse_data.backup_config()
        self.backup_config()

        return BaseDataset(
            dataset_name=self.DATASET_SUBFOLDER,
            dataset_config=attrs.asdict(self.config),
            dataset_sha256_hash=self.config.sha256_hash,
            dataframes=dataframes,
            interactions=interactions,
            impressions=impressions,
            impressions_features_dataframes=impressions_features_dataframes,
            impressions_features_sparse_matrices=impressions_features_sparse_matrices,
            mapper_user_original_id_to_index=mapper_user_original_id_to_index,
            mapper_item_original_id_to_index=mapper_item_original_id_to_index,
            is_impressions_implicit=self.config.binarize_impressions,
            is_interactions_implicit=self.config.binarize_interactions,
        )


if __name__ == "__main__":
    config = MINDSmallConfig()

    data_reader = MINDReader(
        config=config,
    )
    dataset = data_reader.dataset
