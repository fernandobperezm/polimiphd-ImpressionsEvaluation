""" MINDSmallReader.py
This module reads the small or the large version of the Microsoft News Dataset (MIND).

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
        MIND Dataset identifier for the impression. THEY DO NOT COME IN ORDER, i.e., a higher impression ID
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
import os
import re
import zipfile
from enum import Enum
from typing import Optional

import attrs  # Use the novel API for attrs package.
import numpy as np
import pandas as pd
import scipy.sparse as sp
from recsys_framework_extensions.data.dataset import BaseDataset
from recsys_framework_extensions.data.mixins import ParquetDataMixin
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
from recsys_framework_extensions.hashing import compute_sha256_hash_from_object_repr
from recsys_framework_extensions.http import download_remote_file
from recsys_framework_extensions.logging import get_logger
from tqdm import tqdm

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


class MINDVariant(Enum):
    SMALL = "SMALL"
    LARGE = "LARGE"


@attrs.define(kw_only=True, frozen=True, slots=False)
class MINDSmallConfig:
    data_folder = os.path.join(
        ".", "data", "MIND-SMALL",
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
        "user_id": "category",
    }

    num_train_data_points = 156_965
    num_validation_data_points = 73_152
    num_test_data_points = 0

    base_url = 'https://mind201910small.blob.core.windows.net/release'

    train_remote_filename = 'MINDsmall_train.zip'
    validation_remote_filename = 'MINDsmall_dev.zip'
    test_remote_filename = ''

    min_number_of_interactions = attrs.field(
        default=3,
        validator=[
            attrs.validators.gt(0),
        ]
    )
    binarize_impressions = attrs.field(
        default=True,
        validator=[
            attrs.validators.instance_of(bool),
        ]
    )
    binarize_interactions = attrs.field(
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
    use_historical_interactions = attrs.field(
        default=True,
        validator=[
            attrs.validators.instance_of(bool),
        ]
    )
    # The SMALL variant of the dataset has no test set, therefore it must always be false.
    use_test_set = attrs.field(
        default=False,
        validator=[
            attrs.validators.instance_of(bool),
            attrs.validators.in_([False]),
        ]
    )
    variant = attrs.field(
        default=MINDVariant.SMALL,
        validator=[
            attrs.validators.instance_of(MINDVariant),
            attrs.validators.in_([MINDVariant.SMALL]),
        ]
    )


@attrs.define(kw_only=True, frozen=True, slots=False)
class MINDLargeConfig(MINDSmallConfig):
    data_folder = os.path.join(
        ".", "data", "MIND-LARGE",
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


class PandasMINDProcessedData(ParquetDataMixin):
    def __init__(
        self,
        config: MINDSmallConfig,
    ):
        self.config = config
        self.config_hash = compute_sha256_hash_from_object_repr(
            obj=config
        )

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
        self._filename_test = "test.parquet"

        os.makedirs(
            name=self._folder_dataset,
            exist_ok=True,
        )
        os.makedirs(
            name=self._folder_leave_last_out,
            exist_ok=True,
        )
        os.makedirs(
            name=self._folder_timestamp,
            exist_ok=True,
        )

    @property  # type: ignore
    @typed_cache
    def filtered(self) -> pd.DataFrame:
        return self.load_parquet(
            file_path=self._file_path_filtered_data,
            to_pandas_func=self._filtered_to_pandas,
        ).astype(
            dtype=self.config.pandas_dtypes,
        )

    @property  # type: ignore
    @typed_cache
    def leave_last_out_splits(
        self
    ) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame,
    ]:
        file_paths = [
            os.path.join(self._folder_leave_last_out, self._filename_train),
            os.path.join(self._folder_leave_last_out, self._filename_validation),
            os.path.join(self._folder_leave_last_out, self._filename_test),
        ]

        df_train, df_validation, df_test = self.load_parquets(
            file_paths=file_paths,
            to_pandas_func=self._leave_last_out_splits_to_pandas,
        )

        df_train = df_train.astype(dtype=self.config.pandas_dtypes)
        df_validation = df_validation.astype(dtype=self.config.pandas_dtypes)
        df_test = df_test.astype(dtype=self.config.pandas_dtypes)

        return df_train, df_validation, df_test

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

        df_data_train, df_data_test = split_sequential_train_test_by_num_records_on_test(
            df=df_data_exploded_filtered,
            group_by_column="user_id",
            num_records_in_test=1,
        )

        df_data_train, df_data_validation = split_sequential_train_test_by_num_records_on_test(
            df=df_data_train,
            group_by_column="user_id",
            num_records_in_test=1,
        )

        df_data_train = df_data_train.set_index(
            "index",
        )
        df_data_validation = df_data_validation.set_index(
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
        df_data_test, _ = filter_impressions_by_interactions_index(
            df_impressions=df_data_filtered,
            df_interactions=df_data_test,
        )

        return [df_data_train, df_data_validation, df_data_test]


class MINDReader(DataReader):
    def __init__(
        self,
        config: MINDSmallConfig,
    ):
        super().__init__()

        self.config = config
        self.config_hash = compute_sha256_hash_from_object_repr(
            obj=config
        )
        self.processed_data_loader = PandasMINDProcessedData(
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

        self.users_column = "user_id"
        self.items_column = "item_ids"
        self.impressions_column = "impressions"

        self._user_id_to_index_mapper: dict[str, int] = dict()
        self._item_id_to_index_mapper: dict[str, int] = dict()

        self._interactions: dict[str, sp.csr_matrix] = dict()
        self._impressions: dict[str, sp.csr_matrix] = dict()
        self._dataframes: dict[str, pd.DataFrame] = dict()

        self._icms = None
        self._icm_mappers = None

        self._ucms = None
        self._ucms_mappers = None

    @property  # type: ignore
    @typed_cache
    def dataset(self) -> BaseDataset:
        return self.load_data(
            save_folder_path=self.ORIGINAL_SPLIT_FOLDER,
        )

    def _get_dataset_name_root(self) -> str:
        return self.DATASET_SUBFOLDER

    def _load_from_original_file(self) -> BaseDataset:
        # IMPORTANT: calculate first the impressions, so we have all mappers created.
        self._calculate_dataframes()

        self._compute_user_mappers()
        self._compute_item_mappers()

        self._calculate_uim_all()
        self._calculate_urm_all()

        self._calculate_uim_leave_last_k_out_splits()
        self._calculate_urm_leave_last_k_out_splits()

        return BaseDataset(
            dataset_name=self.DATASET_SUBFOLDER,
            impressions=self._impressions,
            interactions=self._interactions,
            dataframes=self._dataframes,
            mapper_item_original_id_to_index=self._item_id_to_index_mapper,
            mapper_user_original_id_to_index=self._user_id_to_index_mapper,
            is_impressions_implicit=self.config.binarize_impressions,
            is_interactions_implicit=self.config.binarize_interactions,
        )

    def _calculate_dataframes(self) -> None:
        self._dataframes[BaseDataset.NAME_DF_FILTERED] = self.processed_data_loader.filtered

        self._dataframes[BaseDataset.NAME_DF_LEAVE_LAST_K_OUT_TRAIN] = \
            self.processed_data_loader.leave_last_out_splits[0]
        self._dataframes[BaseDataset.NAME_DF_LEAVE_LAST_K_OUT_VALIDATION] = \
            self.processed_data_loader.leave_last_out_splits[1]
        self._dataframes[BaseDataset.NAME_DF_LEAVE_LAST_K_OUT_TEST] = \
            self.processed_data_loader.leave_last_out_splits[2]

        # self._dataframes[BaseDataset.NAME_DF_TIMESTAMP_TRAIN] = \
        #     self.processed_data_loader.timestamp_splits[0]
        # self._dataframes[BaseDataset.NAME_DF_TIMESTAMP_VALIDATION] = \
        #     self.processed_data_loader.timestamp_splits[1]
        # self._dataframes[BaseDataset.NAME_DF_TIMESTAMP_TEST] = \
        #     self.processed_data_loader.timestamp_splits[2]

    def _compute_item_mappers(self) -> None:
        df_data_filtered = self.processed_data_loader.filtered[
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

        # Original values in this dataset are strings.
        self._item_id_to_index_mapper = {
            str(orig_value): int(mapped_value)
            for mapped_value, orig_value in enumerate(unique_items)
        }

    def _compute_user_mappers(self) -> None:
        df_data_filtered = self.processed_data_loader.filtered

        non_na_users = df_data_filtered[self.users_column].dropna(
            inplace=False,
        )

        unique_users = set(non_na_users)

        self._user_id_to_index_mapper = {
            str(orig_value): int(mapped_value)
            for mapped_value, orig_value in enumerate(unique_users)
        }

    def _calculate_urm_all(self):
        logger.info(
            f"Building URM with name {BaseDataset.NAME_URM_ALL}."
        )

        df_data_filtered = self.processed_data_loader.filtered[
            [self.users_column, self.items_column]
        ]

        urm_all = create_sparse_matrix_from_dataframe(
            df=df_data_filtered,
            users_column=self.users_column,
            items_column=self.items_column,
            binarize_interactions=self.config.binarize_interactions,
            mapper_user_id_to_index=self._user_id_to_index_mapper,
            mapper_item_id_to_index=self._item_id_to_index_mapper,
        )

        self._interactions[BaseDataset.NAME_URM_ALL] = urm_all

        # _, df_interactions_filtered, _, _ = self._data_filtered
        #
        # builder_urm_all = IncrementalSparseMatrix_FilterIDs(
        #     preinitialized_col_mapper=self._item_id_to_index_mapper,
        #     on_new_col="add",
        #     preinitialized_row_mapper=self._user_id_to_index_mapper,
        #     on_new_row="add"
        # )
        #
        # users = df_interactions_filtered['user_id'].to_numpy()
        # items = df_interactions_filtered['item_id'].to_numpy()
        # data = np.ones_like(users, dtype=np.int32,)
        #
        # builder_urm_all.add_data_lists(
        #     row_list_to_add=users,
        #     col_list_to_add=items,
        #     data_list_to_add=data,
        # )
        #
        # urm_all = builder_urm_all.get_SparseMatrix()
        # if self._binarize_interactions:
        #     urm_all.data = np.ones_like(urm_all.data, dtype=np.int32)
        #
        # self._interactions[BaseDataset.NAME_URM_ALL] = urm_all
        #
        # self._user_id_to_index_mapper = builder_urm_all.get_row_token_to_id_mapper()
        # self._item_id_to_index_mapper = builder_urm_all.get_column_token_to_id_mapper()

    def _calculate_uim_all(self):
        logger.info(
            f"Building UIM with name {BaseDataset.NAME_UIM_ALL}."
        )

        uim_all = create_sparse_matrix_from_dataframe(
            df=self.processed_data_loader.filtered,
            users_column=self.users_column,
            items_column=self.impressions_column,
            binarize_interactions=self.config.binarize_interactions,
            mapper_user_id_to_index=self._user_id_to_index_mapper,
            mapper_item_id_to_index=self._item_id_to_index_mapper,
        )

        self._impressions[BaseDataset.NAME_UIM_ALL] = uim_all.copy()

        # logger.info(
        #     f"Building UIM with name {BaseDataset.NAME_UIM_ALL}."
        # )
        #
        # _, _, df_impressions_filtered, _ = self._data_filtered
        #
        # builder_impressions_all = IncrementalSparseMatrix_FilterIDs(
        #     preinitialized_col_mapper=self._item_id_to_index_mapper,
        #     on_new_col="add",
        #     preinitialized_row_mapper=self._user_id_to_index_mapper,
        #     on_new_row="add",
        # )
        #
        # df_split_chunk: pd.DataFrame
        # for df_split_chunk in tqdm(
        #     np.array_split(df_impressions_filtered, indices_or_sections=self._num_parts_split_dataset)
        # ):
        #     # Explosions of empty lists in impressions are transformed into NAs, NA values must be removed before
        #     # being inserted into the csr_matrix.
        #     df_split_chunk = df_split_chunk.explode(
        #         column="impressions",
        #         ignore_index=False,
        #     )
        #     df_split_chunk = df_split_chunk[
        #         df_split_chunk["impressions"].notna()
        #     ]
        #
        #     impressions = df_split_chunk["impressions"].to_numpy()
        #     users = df_split_chunk["user_id"].to_numpy()
        #     data = np.ones_like(impressions, dtype=np.int32)
        #
        #     builder_impressions_all.add_data_lists(
        #         row_list_to_add=users,
        #         col_list_to_add=impressions,
        #         data_list_to_add=data,
        #     )
        #
        # uim_all = builder_impressions_all.get_SparseMatrix()
        # if self._binarize_impressions:
        #     uim_all.data = np.ones_like(uim_all.data, dtype=np.int32)
        #
        # self._impressions[BaseDataset.NAME_UIM_ALL] = uim_all.copy()
        #
        # self._user_id_to_index_mapper = builder_impressions_all.get_row_token_to_id_mapper()
        # self._item_id_to_index_mapper = builder_impressions_all.get_column_token_to_id_mapper()

    def _calculate_urm_leave_last_k_out_splits(self) -> None:
        df_train, df_validation, df_test = self.processed_data_loader.leave_last_out_splits

        names = [
            BaseDataset.NAME_URM_LEAVE_LAST_K_OUT_TRAIN,
            BaseDataset.NAME_URM_LEAVE_LAST_K_OUT_VALIDATION,
            BaseDataset.NAME_URM_LEAVE_LAST_K_OUT_TEST,
        ]
        splits = [
            df_train,
            df_validation,
            df_test,
        ]

        logger.info(
            f"Building URMs with name {names}."
        )
        for name, df_split in zip(names, splits):
            urm_split = create_sparse_matrix_from_dataframe(
                df=df_split,
                users_column=self.users_column,
                items_column=self.items_column,
                binarize_interactions=self.config.binarize_interactions,
                mapper_user_id_to_index=self._user_id_to_index_mapper,
                mapper_item_id_to_index=self._item_id_to_index_mapper,
            )

            self._interactions[name] = urm_split.copy()

        # _, _, _, df_train, df_validation, df_test, _, _, _, _, _, _ = self._data_leave_last_k_out_split
        #
        # names = [
        #     BaseDataset.NAME_URM_LEAVE_LAST_K_OUT_TRAIN,
        #     BaseDataset.NAME_URM_LEAVE_LAST_K_OUT_VALIDATION,
        #     BaseDataset.NAME_URM_LEAVE_LAST_K_OUT_TEST,
        # ]
        # splits = [
        #     df_train,
        #     df_validation,
        #     df_test
        # ]
        #
        # logger.info(
        #     f"Building URMs with name {names}."
        # )
        # for name, df_split in zip(names, splits):
        #     builder_urm_split = IncrementalSparseMatrix_FilterIDs(
        #         preinitialized_col_mapper=self._item_id_to_index_mapper,
        #         on_new_col="ignore",
        #         preinitialized_row_mapper=self._user_id_to_index_mapper,
        #         on_new_row="ignore"
        #     )
        #
        #     users = df_split['user_id'].to_numpy()
        #     items = df_split['item_id'].to_numpy()
        #     data = np.ones_like(users, dtype=np.int32, )
        #
        #     builder_urm_split.add_data_lists(
        #         row_list_to_add=users,
        #         col_list_to_add=items,
        #         data_list_to_add=data,
        #     )
        #
        #     urm_split = builder_urm_split.get_SparseMatrix()
        #     if self._binarize_interactions:
        #         urm_split.data = np.ones_like(urm_split.data, dtype=np.int32)
        #
        #     self._interactions[name] = urm_split.copy()

    def _calculate_uim_leave_last_k_out_splits(self) -> None:
        df_train, df_validation, df_test = self.processed_data_loader.leave_last_out_splits

        names = [
            BaseDataset.NAME_UIM_LEAVE_LAST_K_OUT_TRAIN,
            BaseDataset.NAME_UIM_LEAVE_LAST_K_OUT_VALIDATION,
            BaseDataset.NAME_UIM_LEAVE_LAST_K_OUT_TEST,
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
            uim_split = create_sparse_matrix_from_dataframe(
                df=df_split,
                users_column=self.users_column,
                items_column=self.impressions_column,
                binarize_interactions=self.config.binarize_interactions,
                mapper_user_id_to_index=self._user_id_to_index_mapper,
                mapper_item_id_to_index=self._item_id_to_index_mapper,
            )

            self._impressions[name] = uim_split.copy()
        # _, _, _, _, _, _, df_train, df_validation, df_test, _, _, _ = self._data_leave_last_k_out_split
        #
        # names = [
        #     BaseDataset.NAME_UIM_LEAVE_LAST_K_OUT_TRAIN,
        #     BaseDataset.NAME_UIM_LEAVE_LAST_K_OUT_VALIDATION,
        #     BaseDataset.NAME_UIM_LEAVE_LAST_K_OUT_TEST,
        # ]
        # splits = [
        #     df_train,
        #     df_validation,
        #     df_test,
        # ]
        #
        # logger.info(
        #     f"Building UIMs with name {names}."
        # )
        # for name, df_split in zip(names, splits):
        #     builder_uim_split = IncrementalSparseMatrix_FilterIDs(
        #         preinitialized_col_mapper=self._item_id_to_index_mapper,
        #         on_new_col="ignore",
        #         preinitialized_row_mapper=self._user_id_to_index_mapper,
        #         on_new_row="ignore"
        #     )
        #
        #     # for _, df_row in tqdm(df_split.iterrows(), total=df_split.shape[0]):
        #     #     impressions = np.array(df_row["impressions"], dtype="object")
        #     #     users = np.array([df_row["user_id"]] * len(impressions), dtype="object")
        #     #     data = np.ones_like(impressions, dtype=np.int32)
        #
        #     for df_split_chunk in tqdm(
        #         np.array_split(df_split, indices_or_sections=self._num_parts_split_dataset)
        #     ):
        #         # Explosions of empty lists in impressions are transformed into NAs, NA values must be removed before
        #         # being inserted into the csr_matrix.
        #         df_split_chunk = df_split_chunk.explode(
        #             column="impressions",
        #             ignore_index=False,
        #         )
        #         df_split_chunk = df_split_chunk[
        #             df_split_chunk["impressions"].notna()
        #         ]
        #
        #         impressions = df_split_chunk["impressions"].to_numpy()
        #         users = df_split_chunk["user_id"].to_numpy()
        #         data = np.ones_like(impressions, dtype=np.int32)
        #
        #         builder_uim_split.add_data_lists(
        #             row_list_to_add=users,
        #             col_list_to_add=impressions,
        #             data_list_to_add=data,
        #         )
        #
        #     uim_split = builder_uim_split.get_SparseMatrix()
        #     if self._binarize_impressions:
        #         uim_split.data = np.ones_like(uim_split.data, dtype=np.int32)
        #
        #     self._impressions[name] = uim_split.copy()


if __name__ == "__main__":
    configs = [
        MINDSmallConfig(),
        # MINDLargeConfig(),
    ]

    for config in configs:
        data_reader = MINDReader(
            config=config,
        )
        dataset = data_reader.dataset

        print(dataset.get_loaded_UIM_names())
        print(dataset.get_loaded_URM_names())
