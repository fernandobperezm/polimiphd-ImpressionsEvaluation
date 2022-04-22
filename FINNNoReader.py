""" FINNNoReader.py

This module holds the classes to read the FINN.NO Slates dataset.
This dataset contains clicks and *no-actions* of users with items in a norwegian marketplace. In particular,
the clicks and no-clicks are with items that may be recommended or searched. These interactions also contain their
corresponding impression record (the list of items shown to the user).

Notes
-----
Creating the UIM ALL requires 110 GB of RAM.

Arrays of the dataset in its original form
    userId
        Identifier of the users (User ID).
    click
        Identifier of interacted items with two exceptions. click=0 means a filler item, it appears if the
        recommendation list was empty for the user.
    click_idx
        The position where the interacted item was placed in the impression list.
    slate
        The impressions, a list of recommendations, searches, or unknown.
    interaction_type
        If the impression came from a recommendation, search, or unknown source.
    slate_lengths
        The number of items present in the recommendation list.
"""
import functools
import os
from enum import Enum
from typing import cast, NamedTuple, Callable

import attrs
import numba
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numba import jit
from numpy.lib.npyio import NpzFile
from recsys_framework_extensions.data.dataset import BaseDataset
from recsys_framework_extensions.data.features import (
    extract_frequency_user_item,
    extract_last_seen_user_item,
    extract_timestamp_user_item,
    extract_position_user_item,
)
from recsys_framework_extensions.data.mixins import ParquetDataMixin, SparseDataMixin, NumpyDataMixin, \
    DatasetConfigBackupMixin
from recsys_framework_extensions.data.reader import DataReader
from recsys_framework_extensions.data.sparse import create_sparse_matrix_from_dataframe
from recsys_framework_extensions.data.splitter import (
    split_sequential_train_test_by_num_records_on_test,
    split_sequential_train_test_by_column_threshold,
    remove_users_without_min_number_of_interactions,
    remove_duplicates_in_interactions,
    E_KEEP,
    remove_records_by_threshold,
    apply_custom_function, randomly_sample_by_column_values,
)
from recsys_framework_extensions.decorators import timeit
from recsys_framework_extensions.evaluation import EvaluationStrategy
from recsys_framework_extensions.hashing.mixins import MixinSHA256Hash
from recsys_framework_extensions.logging import get_logger
from recsys_slates_dataset.data_helper import (
    download_data_files as download_finn_no_slate_files,
)
from tqdm import tqdm

tqdm.pandas()


logger = get_logger(
    logger_name=__file__,
)


_MIN_ITEM_ID = 3
_NUM_ITEMS_IN_IMPRESSIONS_RAW_DATA = 25


@jit(nopython=True, parallel=False)
def compute_impressions_positions_based_on_impressions_values(
    impressions: np.ndarray,
    min_item_id: int = _MIN_ITEM_ID,
) -> np.ndarray:
    assert len(impressions.shape) == 1

    num_cols = impressions.shape[0]

    # We start at 1 as the first element of the impressions array always contains the item 1.
    # This item is the 'no-click' item and is added to all impressions in the first position.
    # This value however, was never recommended, therefore, the first real item must begin with
    # position 0.
    position_value = 0
    positions_arr = []
    for col_idx in range(1, num_cols):
        if impressions[col_idx] >= min_item_id:
            positions_arr.append(position_value)
        position_value += 1

    return np.array(positions_arr, dtype=np.int32)


list(map(
    compute_impressions_positions_based_on_impressions_values,
    [
        np.array([0, 1, 2, 0, 0], dtype=np.int32),
        np.array([0, 0, 0, 0, 0], dtype=np.int32),
        np.array([0, 1, 2, 3, 4], dtype=np.int32),
        np.array([0, 1, 2, 3, 4], dtype=np.int32),
        np.arange(start=100, stop=125, dtype=np.int32),
        np.arange(start=2, stop=27, dtype=np.int32),
    ],
))


@jit(nopython=True, parallel=False)
def remove_non_clicks_on_impressions(
    impressions: np.ndarray,
    min_item_id: int = _MIN_ITEM_ID,
) -> np.ndarray:
    assert len(impressions.shape) == 1
    num_cols = impressions.shape[0]

    return np.array(
        [
            impressions[col_idx]
            for col_idx in numba.prange(num_cols)
            if impressions[col_idx] >= min_item_id
        ],
        dtype=np.int32,
    )


map(
    remove_non_clicks_on_impressions,
    [
        np.array([0, 1, 2, 0, 0], dtype=np.int32),
        np.array([0, 0, 0, 0, 0], dtype=np.int32),
        np.array([0, 1, 2, 3, 4], dtype=np.int32),
    ],
)


@jit(nopython=True, parallel=False)
def is_item_in_impression(
    impressions: np.ndarray,
    item_ids: np.ndarray,
) -> np.ndarray:
    """Numba-compiled-non-parallel function that calculates if an item is inside an impression.

    This function iterates over the data points of both ndarrays and determines if each item in `item_ids` was
    impressed in `impressions`.

    Parameters
    ----------
    impressions
        A (M, 25) matrix where rows represent data points and columns the position of each item in the impressions
        list. `impressions[i,j]` returns the impressed item_id for the data point `i` in position `j`.
    item_ids
        A (M, ) vector where M represents data points. `item_ids[i]` returns the `item_id` for the data point `i`.

    Returns
    -------
    np.ndarray
        A (M, ) vector of booleans. `a[i] == True` means that `item_ids[i]` was impressed in `impressions[i,:]`
    """

    assert impressions.shape[0] == item_ids.shape[0]
    assert impressions.shape[1] == 25

    num_impressions: int = impressions.shape[0]
    arr_item_in_impressions = np.empty_like(item_ids, dtype=np.bool8)

    for idx in range(num_impressions):
        arr_item_in_impressions[idx] = item_ids[idx] in impressions[idx]

    return arr_item_in_impressions


# Ensure `is_item_in_impression` is JIT-compiled with expected array type,
# to avoid time in compiling.
is_item_in_impression(impressions=np.array([range(25)]), item_ids=np.array([1]))
is_item_in_impression(impressions=np.array([range(25)]), item_ids=np.array([27]))
is_item_in_impression(impressions=np.array([range(25), range(25)]), item_ids=np.array([1, 27]))


class FinnNoSlatesSplits(NamedTuple):
    df_train: pd.DataFrame
    df_validation: pd.DataFrame
    df_train_validation: pd.DataFrame
    df_test: pd.DataFrame


@attrs.define(kw_only=True, frozen=True, slots=False)
class FinnNoSlatesConfig(MixinSHA256Hash):
    """Configuration class used by data readers of the FINN.No dataset"""
    data_folder = os.path.join(
        ".", "data", "FINN-NO-SLATE",
    )

    try_low_memory_usage = True

    num_raw_data_points = 2_277_645
    num_time_steps = 20
    num_items_in_each_impression = 25
    num_data_points: int  # = 45_552_900  # MUST BE NUM_RAW_DATA_POINTS * NUM_TIME_STEPS

    pandas_dtypes = {
        "user_id": np.int32,
        "item_id": np.int32,
        "time_step": np.int32,
        "impressions_origin": "category",
        "position_interaction": np.int32,
        "num_impressions": np.int32,
        "is_item_in_impression": pd.BooleanDtype(),
    }

    # The dataset treats 0s, 1s, and 2s, as error, no-clicks, and non-identifiable items, respectively.
    reproducibility_seed = attrs.field(
        default=1234567890,
        validator=[
            attrs.validators.instance_of(int),
            attrs.validators.gt(0),
        ]
    )
    frac_users_to_keep = attrs.field(
        default=1.0,
        validator=[
            attrs.validators.instance_of(float),
            attrs.validators.ge(0.0),
            attrs.validators.le(1.0),
        ]
    )
    item_ids_to_exclude = attrs.field(
        default=[0, 1, 2],
        validator=[
            attrs.validators.instance_of(list),
        ]
    )
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

    def __attrs_post_init__(self):
        # We need to use object.__setattr__ because the benchmark_config is an immutable class, this is the attrs way to
        # circumvent assignment in immutable classes.
        object.__setattr__(self, "num_data_points", self.num_raw_data_points * self.num_time_steps)
        assert 45_552_900 == self.num_data_points


class FINNNoImpressionOrigin(Enum):
    """Enum indicating the types of impressions present in the FINN.No dataset"""
    UNDEFINED = 0
    SEARCH = 1
    RECOMMENDATION = 2


class FINNNoSlateRawData:
    """Class that reads the 'raw' FINN.No data from disk.

    We refer to raw data as the original representation of the data,
    without cleaning or much processing. The dataset originally has XYZ-fernando-debugger data points.
    """

    def __init__(
        self,
        config: FinnNoSlatesConfig,
    ):
        self.config = config

        self._original_dataset_folder = os.path.join(
            self.config.data_folder, "original",
        )
        self._original_dataset_data_file = os.path.join(
            self._original_dataset_folder, "data.npz",
        )
        self._original_dataset_mapper_file = os.path.join(
            self._original_dataset_folder, "ind2val.json",
        )
        self._original_dataset_item_attr_file = os.path.join(
            self._original_dataset_folder, "itemattr.npz",
        )
        self._original_dataset_pandas_file = os.path.join(
            self._original_dataset_folder, "raw_data.parquet"
        )

    @property  # type: ignore
    def raw_data(self) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        self._download_dataset()

        with cast(
            NpzFile, np.load(file=self._original_dataset_data_file)
        ) as interactions:
            user_ids = cast(np.ndarray, interactions["userId"])
            item_ids = cast(np.ndarray, interactions["click"])
            click_indices_in_impressions = cast(np.ndarray, interactions["click_idx"])
            impressions = cast(np.ndarray, interactions["slate"])
            impression_origins = cast(np.ndarray, interactions["interaction_type"])
            impressions_length_list = cast(np.ndarray, interactions["slate_lengths"])

            assert (self.config.num_raw_data_points,) == user_ids.shape
            assert (self.config.num_raw_data_points, self.config.num_time_steps) == item_ids.shape
            assert (self.config.num_raw_data_points, self.config.num_time_steps) == click_indices_in_impressions.shape
            assert (self.config.num_raw_data_points, self.config.num_time_steps) == impression_origins.shape
            assert (self.config.num_raw_data_points, self.config.num_time_steps) == impressions_length_list.shape
            assert (self.config.num_raw_data_points, self.config.num_time_steps,
                    self.config.num_items_in_each_impression) == impressions.shape

            logger.debug(
                f"Loaded raw data (user_ids, item_ids, click_indices_in_impressions, impressions, impression_origins, "
                f"impressions_length_list) from {self.__class__.__name__}."
            )
            return (
                user_ids,
                item_ids,
                click_indices_in_impressions,
                impressions,
                impression_origins,
                impressions_length_list,
            )

    def _download_dataset(self) -> None:
        os.makedirs(
            name=self._original_dataset_folder,
            exist_ok=True,
        )

        if not os.path.exists(self._original_dataset_data_file):
            logger.info(
                "Downloading dataset from the original source and creating parquet parts."
            )

            # This function is provided by the dataset's author. It downloads three files:
            # * data.npz
            # * ind2val.json
            # * itemattr.npz
            download_finn_no_slate_files(
                data_dir=self._original_dataset_folder,
                overwrite=False,
                progbar=False,
                use_int32=True,
            )


class PandasFinnNoSlateRawData(ParquetDataMixin, NumpyDataMixin):
    """A class that reads the FINN.No data using Pandas Dataframes."""

    def __init__(
        self,
        config: FinnNoSlatesConfig
    ):
        self.config = config
        self.raw_data_loader = FINNNoSlateRawData(
            config=config,
        )

        self._folder_dataset = os.path.join(
            self.config.data_folder, "pandas", "original", "",
        )
        self._file_path_data = os.path.join(
            self._folder_dataset, "data.parquet"
        )
        self._file_name_impressions_arr = "impressions.npz"

        os.makedirs(
            name=self._folder_dataset,
            exist_ok=True,
        )

    @property  # type: ignore
    def data(self) -> pd.DataFrame:
        df_data = self.load_parquet(
            file_path=self._file_path_data,
            to_pandas_func=self._data_to_pandas,
        ).astype(
            dtype=self.config.pandas_dtypes,
        )

        impressions = self.load_from_numpy(
            file_path=os.path.join(self._folder_dataset, self._file_name_impressions_arr),
            to_numpy_func=self._data_impressions_to_np_arr,
        )

        assert df_data.shape[0] == impressions.shape[0]

        df_data["impressions"] = list(impressions)

        logger.debug(
            f"Loaded pandas data (df_data with impressions) from {self.__class__.__name__}."
        )

        return df_data

    def _data_to_pandas(self) -> pd.DataFrame:
        (
            user_ids,
            item_ids,
            click_indices_in_impressions,
            impressions,
            impression_origins,
            impressions_length_list,
        ) = self.raw_data_loader.raw_data

        user_id_arr = user_ids.repeat(
            repeats=self.config.num_time_steps,
        )
        item_id_arr = item_ids.reshape(
            (self.config.num_data_points,)
        )
        time_step_arr = np.array(
            [range(self.config.num_time_steps)] * user_ids.shape[0]
        ).reshape(
            (self.config.num_data_points,)
        ).astype(
            dtype=np.int32,
        )
        impression_origin_arr = impression_origins.reshape(
            (self.config.num_data_points,)
        )
        position_interactions_arr = click_indices_in_impressions.reshape(
            (self.config.num_data_points,)
        )
        impressions_arr = impressions.reshape(
            (self.config.num_data_points, self.config.num_items_in_each_impression)
        )
        num_impressions_arr = impressions_length_list.reshape(
            (self.config.num_data_points,)
        )
        is_item_in_impression_arr = is_item_in_impression(
            impressions=impressions_arr,
            item_ids=item_id_arr,
        )

        assert (self.config.num_data_points,) == user_id_arr.shape
        assert (self.config.num_data_points,) == item_id_arr.shape
        assert (self.config.num_data_points,) == time_step_arr.shape
        assert (self.config.num_data_points,) == position_interactions_arr.shape
        assert (self.config.num_data_points,) == impression_origin_arr.shape
        assert (self.config.num_data_points,) == num_impressions_arr.shape
        assert (self.config.num_data_points,) == is_item_in_impression_arr.shape
        assert (self.config.num_data_points, self.config.num_items_in_each_impression) == impressions_arr.shape

        df_data = pd.DataFrame(
            data={
                "user_id": user_id_arr,
                "item_id": item_id_arr,
                "time_step": time_step_arr,
                "position_interaction": position_interactions_arr,
                "impressions_origin": impression_origin_arr,
                "num_impressions": num_impressions_arr,
                "is_item_in_impression": is_item_in_impression_arr,
            },
        ).astype(
            dtype=self.config.pandas_dtypes,
        )

        assert (self.config.num_data_points, 7) == df_data.shape

        return df_data

    def _data_impressions_to_np_arr(self) -> np.ndarray:
        (
            _,
            _,
            _,
            impressions,
            _,
            _,
        ) = self.raw_data_loader.raw_data

        impressions_arr = impressions.reshape(
            (self.config.num_data_points, self.config.num_items_in_each_impression)
        )

        assert (self.config.num_data_points, self.config.num_items_in_each_impression) == impressions_arr.shape

        return impressions_arr


class PandasFinnNoSlateProcessData(ParquetDataMixin, DatasetConfigBackupMixin):
    def __init__(
        self,
        config: FinnNoSlatesConfig,
    ):
        self.config = config

        self.pandas_raw_data = PandasFinnNoSlateRawData(
            config=config,
        )

        self._folder_dataset = os.path.join(
            self.config.data_folder, "data-processing", self.config.sha256_hash, ""
        )

        self._folder_leave_last_k_out = os.path.join(
            self._folder_dataset, "leave-last-k-out", ""
        )

        self._folder_timestamp = os.path.join(
            self._folder_dataset, "timestamp", ""
        )

        self._file_path_filter_data = os.path.join(
            self._folder_dataset, "filter_data.parquet"
        )
        self._file_path_sampled_data = os.path.join(
            self._folder_dataset, "sampled_data.parquet",
        )
        self._file_path_sampled_out_data = os.path.join(
            self._folder_dataset, "sampled_out_data.parquet",
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
            name=self._folder_leave_last_k_out,
            exist_ok=True,
        )
        os.makedirs(
            name=self._folder_timestamp,
            exist_ok=True,
        )

        self.save_config(
            folder_path=self._folder_dataset,
            config=self.config,
        )

    @property
    def _sampled(self) -> pd.DataFrame:
        df_data_sampled = self.load_parquet(
            file_path=self._file_path_sampled_data,
            to_pandas_func=self._sampled_to_pandas,
        ).astype(
            dtype=self.config.pandas_dtypes,
        )

        return df_data_sampled

    @property
    def filtered(self) -> pd.DataFrame:
        df_data_filtered = self.load_parquet(
            file_path=self._file_path_filter_data,
            to_pandas_func=self._filtered_to_pandas,
        ).astype(
            dtype=self.config.pandas_dtypes,
        )

        logger.debug(
            f"Loaded pandas filtered data (df_data_filtered) from {self.__class__.__name__}."
        )

        return df_data_filtered

    @property
    def timestamp_splits(
        self
    ) -> FinnNoSlatesSplits:
        file_paths = [
            os.path.join(self._folder_timestamp, self._filename_train),
            os.path.join(self._folder_timestamp, self._filename_validation),
            os.path.join(self._folder_timestamp, self._filename_train_validation),
            os.path.join(self._folder_timestamp, self._filename_test),
        ]

        df_train, df_validation, df_train_validation, df_test = self.load_parquets(
            file_paths=file_paths,
            to_pandas_func=self._timestamp_splits_to_pandas,
        )

        df_train = df_train.astype(dtype=self.config.pandas_dtypes)
        df_validation = df_validation.astype(dtype=self.config.pandas_dtypes)
        df_train_validation = df_train_validation.astype(dtype=self.config.pandas_dtypes)
        df_test = df_test.astype(dtype=self.config.pandas_dtypes)

        logger.debug(
            f"Loaded pandas time step data splits (df_train, df_validation, df_test) from {self.__class__.__name__}."
        )

        return FinnNoSlatesSplits(
            df_train=df_train,
            df_validation=df_validation,
            df_train_validation=df_train_validation,
            df_test=df_test,
        )

    @property
    def leave_last_k_out_splits(
        self
    ) -> FinnNoSlatesSplits:
        file_paths = [
            os.path.join(self._folder_leave_last_k_out, self._filename_train),
            os.path.join(self._folder_leave_last_k_out, self._filename_validation),
            os.path.join(self._folder_leave_last_k_out, self._filename_train_validation),
            os.path.join(self._folder_leave_last_k_out, self._filename_test),
        ]

        df_train, df_validation, df_train_validation, df_test = self.load_parquets(
            file_paths=file_paths,
            to_pandas_func=self._leave_last_k_out_splits_to_pandas,
        )

        df_train = df_train.astype(dtype=self.config.pandas_dtypes)
        df_validation = df_validation.astype(dtype=self.config.pandas_dtypes)
        df_train_validation = df_train_validation.astype(dtype=self.config.pandas_dtypes)
        df_test = df_test.astype(dtype=self.config.pandas_dtypes)

        logger.debug(
            f"Loaded pandas leave-last-out data splits (df_train, df_validation, df_test) from"
            f" {self.__class__.__name__}."
        )

        return FinnNoSlatesSplits(
            df_train=df_train,
            df_validation=df_validation,
            df_train_validation=df_train_validation,
            df_test=df_test,
        )

    @property
    def dataframes(self) -> dict[str, pd.DataFrame]:
        return {
            BaseDataset.NAME_DF_FILTERED: self.filtered,

            BaseDataset.NAME_DF_LEAVE_LAST_K_OUT_TRAIN: self.leave_last_k_out_splits.df_train,
            BaseDataset.NAME_DF_LEAVE_LAST_K_OUT_VALIDATION: self.leave_last_k_out_splits.df_validation,
            BaseDataset.NAME_DF_LEAVE_LAST_K_OUT_TRAIN_VALIDATION: self.leave_last_k_out_splits.df_train_validation,
            BaseDataset.NAME_DF_LEAVE_LAST_K_OUT_TEST: self.leave_last_k_out_splits.df_test,

            BaseDataset.NAME_DF_TIMESTAMP_TRAIN: self.timestamp_splits.df_train,
            BaseDataset.NAME_DF_TIMESTAMP_VALIDATION: self.timestamp_splits.df_validation,
            BaseDataset.NAME_DF_TIMESTAMP_TRAIN_VALIDATION: self.timestamp_splits.df_train_validation,
            BaseDataset.NAME_DF_TIMESTAMP_TEST: self.timestamp_splits.df_test,
        }

    @timeit
    def _sampled_to_pandas(self) -> pd.DataFrame:
        df_data = self.pandas_raw_data.data

        df_data, df_removed = randomly_sample_by_column_values(
            df=df_data,
            frac_to_keep=self.config.frac_users_to_keep,
            column="user_id",
            seed=self.config.reproducibility_seed,
        )

        # Save the sampled out data into disk in case we need it in the future.
        self._to_parquet(
            df=df_removed,
            file_path=self._file_path_sampled_out_data,
        )

        return df_data

    @timeit
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

        df_data = self._sampled

        # We don't need to reset the index first because this dataset does not have exploded values.
        df_data = df_data.sort_values(
            by="time_step",
            ascending=True,
            axis="index",
            inplace=False,
            ignore_index=False,
        )

        # This filter removes error logs and non-interactions of the dataset.
        df_data, _ = remove_records_by_threshold(
            df=df_data,
            column="item_id",
            threshold=self.config.item_ids_to_exclude,
            how="not_isin",
        )

        df_data, _ = remove_duplicates_in_interactions(
            df=df_data,
            columns_to_compare=["user_id", "item_id"],
            keep=self.config.keep_duplicates,
        )

        df_data, _ = remove_users_without_min_number_of_interactions(
            df=df_data,
            users_column="user_id",
            min_number_of_interactions=self.config.min_number_of_interactions,
        )

        df_data["impressions_positions"] = df_data["impressions"].progress_apply(
            func=compute_impressions_positions_based_on_impressions_values,
        )

        # This filter removes error logs and non-interactions of the dataset.
        df_data, _ = apply_custom_function(
            df=df_data,
            column="impressions",
            func=remove_non_clicks_on_impressions,
            func_name=remove_non_clicks_on_impressions.__name__,
            axis="columns",
        )

        # Given that we removed the 0s and 1s in the impressions, then we must subtract 1 to the columns
        # `position_interaction` and `num_impressions` given that they account for the existence of one extra element.
        df_data["position_interaction"] -= 1
        df_data["num_impressions"] -= 1

        # We don't have to process the column "is_item_in_impression" in the metadata because we already removed the
        # non-clicks on the dataset interactions and impressions.

        return df_data

    @timeit
    def _timestamp_splits_to_pandas(self) -> list[pd.DataFrame]:
        df_filtered_data = self.filtered

        described = df_filtered_data["time_step"].describe(
            include="all",
            percentiles=[0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        )

        validation_threshold = described["80%"]
        test_threshold = described["90%"]

        df_train_validation, df_test = split_sequential_train_test_by_column_threshold(
            df=df_filtered_data,
            column="time_step",
            threshold=test_threshold,
        )

        df_train, df_validation = split_sequential_train_test_by_column_threshold(
            df=df_train_validation,
            column="time_step",
            threshold=validation_threshold,
        )

        return [df_train, df_validation, df_train_validation, df_test]

    @timeit
    def _leave_last_k_out_splits_to_pandas(self) -> list[pd.DataFrame]:
        df_data_filtered = self.filtered

        df_train_validation, df_test = split_sequential_train_test_by_num_records_on_test(
            df=df_data_filtered,
            group_by_column="user_id",
            num_records_in_test=1,
        )

        df_train, df_validation = split_sequential_train_test_by_num_records_on_test(
            df=df_train_validation,
            group_by_column="user_id",
            num_records_in_test=1,
        )

        return [df_train, df_validation, df_train_validation, df_test]


class PandasFinnNoSlateImpressionsFeaturesData(ParquetDataMixin, DatasetConfigBackupMixin):
    def __init__(
        self,
        config: FinnNoSlatesConfig,
    ):
        self.config = config
        self.loader_processed_data = PandasFinnNoSlateProcessData(
            config=config,
        )

        self._folder_dataset = os.path.join(
            self.config.data_folder, "data-features-impressions", self.config.sha256_hash, ""
        )

        self._folder_leave_last_k_out = os.path.join(
            self._folder_dataset, "leave-last-k-out", ""
        )

        self._folder_timestamp = os.path.join(
            self._folder_dataset, "timestamp", ""
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
                timestamp_column="time_step",
            ),
            "user_item_position": functools.partial(
                extract_position_user_item,
                users_column="user_id",
                items_column="impressions",
                positions_column="impressions_positions",
                to_keep="last",
            ),
            "user_item_timestamp": functools.partial(
                extract_timestamp_user_item,
                users_column="user_id",
                items_column="impressions",
                timestamp_column="time_step",
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
        os.makedirs(
            name=self._folder_timestamp,
            exist_ok=True,
        )

        for folder in [self._folder_leave_last_k_out, self._folder_timestamp]:
            for sub_folder in self._feature_funcs.keys():
                os.makedirs(
                    name=os.path.join(folder, sub_folder),
                    exist_ok=True,
                )

        self.save_config(
            config=self.config,
            folder_path=self._folder_dataset,
        )

    @property
    def features(self) -> dict[str, list[str]]:
        return {
            "user_item_frequency": ["feature-user_id-impressions-frequency"],
            "user_item_last_seen": ["feature_last_seen_euclidean"],
            "user_item_position": ["feature-user_id-impressions-position"],
            "user_item_timestamp": ["feature-user_id-impressions-timestamp"],
        }

    @property
    def impressions_features(self) -> dict[str, pd.DataFrame]:
        impression_features: dict[str, pd.DataFrame] = {}

        for evaluation_strategy in EvaluationStrategy:
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
    ) -> FinnNoSlatesSplits:
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

        return FinnNoSlatesSplits(
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
            folder_to_look_up = self._folder_timestamp

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
    ) -> FinnNoSlatesSplits:
        if EvaluationStrategy.LEAVE_LAST_K_OUT == evaluation_strategy:
            return self.loader_processed_data.leave_last_k_out_splits
        else:
            return self.loader_processed_data.timestamp_splits

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


class SparseFinnNoSlateData(SparseDataMixin, ParquetDataMixin, DatasetConfigBackupMixin):
    def __init__(
        self,
        config: FinnNoSlatesConfig,
    ):
        super().__init__()

        self.config = config

        self.data_loader_processed = PandasFinnNoSlateProcessData(
            config=config,
        )
        self.data_loader_impression_features = PandasFinnNoSlateImpressionsFeaturesData(
            config=config,
        )

        self.users_column = "user_id"
        self.items_column = "item_id"
        self.impressions_column = "impressions"

        self._folder_data = os.path.join(
            self.config.data_folder, "data-sparse", self.config.sha256_hash, "",
        )
        self._folder_leave_last_out_data = os.path.join(
            self._folder_data, EvaluationStrategy.LEAVE_LAST_K_OUT.value, "",
        )
        self._folder_timestamp_data = os.path.join(
            self._folder_data, EvaluationStrategy.TIMESTAMP.value, "",
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
        self._file_paths_timestamp_urms = [
            os.path.join(self._folder_timestamp_data, "urm_train.npz"),
            os.path.join(self._folder_timestamp_data, "urm_validation.npz"),
            os.path.join(self._folder_timestamp_data, "urm_train_validation.npz"),
            os.path.join(self._folder_timestamp_data, "urm_test.npz"),
        ]

        self._file_path_uim_all = os.path.join(self._folder_data, "uim_all.npz")
        self._file_paths_leave_last_out_uims = [
            os.path.join(self._folder_leave_last_out_data, "uim_train.npz"),
            os.path.join(self._folder_leave_last_out_data, "uim_validation.npz"),
            os.path.join(self._folder_leave_last_out_data, "uim_train_validation.npz"),
            os.path.join(self._folder_leave_last_out_data, "uim_test.npz"),
        ]
        self._file_paths_timestamp_uims = [
            os.path.join(self._folder_timestamp_data, "uim_train.npz"),
            os.path.join(self._folder_timestamp_data, "uim_validation.npz"),
            os.path.join(self._folder_timestamp_data, "uim_train_validation.npz"),
            os.path.join(self._folder_timestamp_data, "uim_test.npz"),
        ]

        os.makedirs(self._folder_data, exist_ok=True)
        os.makedirs(self._folder_leave_last_out_data, exist_ok=True)
        os.makedirs(self._folder_timestamp_data, exist_ok=True)

        self.save_config(
            config=self.config,
            folder_path=self._folder_data,
        )

    @functools.cached_property
    def mapper_user_id_to_index(self) -> dict[int, int]:
        df_user_mapper = self.load_parquet(
            file_path=self._file_path_user_mapper,
            to_pandas_func=self._user_mapper_to_pandas,
        )

        return {
            int(df_row.orig_value): int(df_row.mapped_value)
            for df_row in df_user_mapper.itertuples(index=False)
        }

    @functools.cached_property
    def mapper_item_id_to_index(self) -> dict[int, int]:
        df_item_mapper = self.load_parquet(
            file_path=self._file_path_item_mapper,
            to_pandas_func=self._item_mapper_to_pandas,
        )

        return {
            int(df_row.orig_value): int(df_row.mapped_value)
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

        (
            sp_timestamp_urm_train,
            sp_timestamp_urm_validation,
            sp_timestamp_urm_train_validation,
            sp_timestamp_urm_test,
        ) = self.load_sparse_matrices(
            file_paths=self._file_paths_timestamp_urms,
            to_sparse_matrices_func=self._urms_timestamp_to_sparse,
        )

        return {
            BaseDataset.NAME_URM_ALL: sp_urm_all,

            BaseDataset.NAME_URM_LEAVE_LAST_K_OUT_TRAIN: sp_llo_urm_train,
            BaseDataset.NAME_URM_LEAVE_LAST_K_OUT_VALIDATION: sp_llo_urm_validation,
            BaseDataset.NAME_URM_LEAVE_LAST_K_OUT_TRAIN_VALIDATION: sp_llo_urm_train_validation,
            BaseDataset.NAME_URM_LEAVE_LAST_K_OUT_TEST: sp_llo_urm_test,

            BaseDataset.NAME_URM_TIMESTAMP_TRAIN: sp_timestamp_urm_train,
            BaseDataset.NAME_URM_TIMESTAMP_VALIDATION: sp_timestamp_urm_validation,
            BaseDataset.NAME_URM_TIMESTAMP_TRAIN_VALIDATION: sp_timestamp_urm_train_validation,
            BaseDataset.NAME_URM_TIMESTAMP_TEST: sp_timestamp_urm_test,
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

        (
            sp_timestamp_uim_train,
            sp_timestamp_uim_validation,
            sp_timestamp_uim_train_validation,
            sp_timestamp_uim_test,
        ) = self.load_sparse_matrices(
            file_paths=self._file_paths_timestamp_uims,
            to_sparse_matrices_func=self._uims_timestamp_to_sparse,
        )

        return {
            BaseDataset.NAME_UIM_ALL: sp_uim_all,

            BaseDataset.NAME_UIM_LEAVE_LAST_K_OUT_TRAIN: sp_llo_uim_train,
            BaseDataset.NAME_UIM_LEAVE_LAST_K_OUT_VALIDATION: sp_llo_uim_validation,
            BaseDataset.NAME_UIM_LEAVE_LAST_K_OUT_TRAIN_VALIDATION: sp_llo_uim_train_validation,
            BaseDataset.NAME_UIM_LEAVE_LAST_K_OUT_TEST: sp_llo_uim_test,

            BaseDataset.NAME_UIM_TIMESTAMP_TRAIN: sp_timestamp_uim_train,
            BaseDataset.NAME_UIM_TIMESTAMP_VALIDATION: sp_timestamp_uim_validation,
            BaseDataset.NAME_UIM_TIMESTAMP_TRAIN_VALIDATION: sp_timestamp_uim_train_validation,
            BaseDataset.NAME_UIM_TIMESTAMP_TEST: sp_timestamp_uim_test,
        }

    @property
    def impressions_features(self) -> dict[str, sp.csr_matrix]:
        impression_features: dict[str, sp.csr_matrix] = {}

        for evaluation_strategy in EvaluationStrategy:
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

        non_na_items = df_data_filtered[self.items_column].dropna(
            inplace=False,
        )

        non_na_exploded_impressions = df_data_filtered[self.impressions_column].explode(
            ignore_index=True,
        ).dropna(
            inplace=False,
        )

        unique_items = set(non_na_items).union(non_na_exploded_impressions)

        return pd.DataFrame.from_records(
            data=[
                (int(mapped_value), int(orig_value))
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
                (int(mapped_value), int(orig_value))
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
        df_train, df_validation, df_train_validation, df_test = self.data_loader_processed.leave_last_k_out_splits

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

    def _urms_timestamp_to_sparse(self) -> list[sp.csr_matrix]:
        df_train, df_validation, df_train_validation, df_test = self.data_loader_processed.timestamp_splits

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
        df_train, df_validation, df_train_validation, df_test = self.data_loader_processed.leave_last_k_out_splits

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

    def _uims_timestamp_to_sparse(self) -> list[sp.csr_matrix]:
        df_train, df_validation, df_train_validation, df_test = self.data_loader_processed.timestamp_splits

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


class FINNNoSlateReader(DatasetConfigBackupMixin, DataReader):
    def __init__(
        self,
        config: FinnNoSlatesConfig,
    ):
        super().__init__()

        self.config = config

        self.data_loader_processed = PandasFinnNoSlateProcessData(
            config=config,
        )
        self.data_loader_impressions_features = PandasFinnNoSlateImpressionsFeaturesData(
            config=config,
        )
        self.data_loader_sparse = SparseFinnNoSlateData(
            config=config,
        )

        self.DATA_FOLDER = os.path.join(
            self.config.data_folder, "data_reader", self.config.sha256_hash, "",
        )

        self._DATA_READER_NAME = "FINNNoSlateReader"
        self.DATASET_SUBFOLDER = "FINN-NO-SLATE/"
        self.IS_IMPLICIT = self.config.binarize_interactions
        self.ORIGINAL_SPLIT_FOLDER = self.DATA_FOLDER

        os.makedirs(
            self.DATA_FOLDER,
            exist_ok=True
        )

        self.save_config(
            config=self.config,
            folder_path=self.DATA_FOLDER,
        )

    @property  # type: ignore
    def dataset(self) -> BaseDataset:
        return self.load_data(
            save_folder_path=self.ORIGINAL_SPLIT_FOLDER,
        )

    def _get_dataset_name_root(self) -> str:
        return self.DATASET_SUBFOLDER

    def _load_from_original_file(self) -> BaseDataset:
        dataframes = self.data_loader_processed.dataframes
        interactions = self.data_loader_sparse.interactions
        impressions = self.data_loader_sparse.impressions

        mapper_item_id_to_index = self.data_loader_sparse.mapper_item_id_to_index
        mapper_user_id_to_index = self.data_loader_sparse.mapper_user_id_to_index

        impressions_features_dataframes = self.data_loader_impressions_features.impressions_features
        impressions_features_sparse_matrices = self.data_loader_sparse.impressions_features

        return BaseDataset(
            dataset_name="FINNNoSlate",
            dataset_config=attrs.asdict(self.config),
            dataset_sha256_hash=self.config.sha256_hash,
            impressions=impressions,
            interactions=interactions,
            dataframes=dataframes,
            impressions_features_dataframes=impressions_features_dataframes,
            impressions_features_sparse_matrices=impressions_features_sparse_matrices,
            mapper_item_original_id_to_index=mapper_item_id_to_index,
            mapper_user_original_id_to_index=mapper_user_id_to_index,
            is_impressions_implicit=self.config.binarize_impressions,
            is_interactions_implicit=self.config.binarize_interactions,
        )


if __name__ == "__main__":
    config = FinnNoSlatesConfig(
        # frac_users_to_keep=0.05,
        frac_users_to_keep=0.001,
    )

    data_reader = FINNNoSlateReader(
        config=config,
    )

    dataset = data_reader.dataset

    print(dataset.dataframe_available_features())
    print(dataset.sparse_matrices_available_features())
