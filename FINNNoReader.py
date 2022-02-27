""" FINNNoReader.py

This module holds the classes to read the FINN.NO Slates dataset.
This dataset contains clicks and *no-actions* of users with items in a norwegian marketplace. In particular,
the clicks and no-clicks are with items that may be recommended or searched. These interactions also contain their
corresponding impression record (the list of items shown to the user).

Notes
-----
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

import os
from enum import Enum
from typing import cast, Any
from memory_profiler import profile

import attr
import dask.dataframe as dd
import numba
import numpy as np
import pandas as pd
import scipy.sparse as sp
from dask.delayed import delayed
from numba import jit
from numpy.lib.npyio import NpzFile
from recsys_framework.Data_manager.DataReader import DataReader
from recsys_framework.Data_manager.Dataset import gini_index
from recsys_framework.Data_manager.IncrementalSparseMatrix import (
    IncrementalSparseMatrix_FilterIDs,
)
from recsys_framework.Recommenders.DataIO import DataIO
from recsys_framework.Utils.conf_logging import get_logger
from recsys_framework.Utils.decorators import timeit
from recsys_slates_dataset.data_helper import (
    download_data_files as download_finn_no_slate_files,
)
from tqdm import tqdm

from data_splitter import filter_impressions_by_interactions_index, split_sequential_train_test_by_num_records_on_test, \
    split_sequential_train_test_by_column_threshold, remove_users_without_min_number_of_interactions, \
    remove_duplicates_in_interactions, T_KEEP, remove_records_by_threshold, apply_custom_function
from mixins import BaseDataset, ParquetDataMixin
from utils import typed_cache

tqdm.pandas()


logger = get_logger(
    logger_name=__file__,
)


_MIN_ITEM_ID = 3


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

    return np.array(
        [
            item_ids[idx] in impressions[idx]
            for idx in range(num_impressions)
        ],
    )


# Ensure `is_item_in_impression` is JIT-compiled with expected array type,
# to avoid time in compiling.
is_item_in_impression(impressions=np.array([range(25)]), item_ids=np.array([1]))
is_item_in_impression(impressions=np.array([range(25)]), item_ids=np.array([1]))


@jit(nopython=True, parallel=False)
def compute_interaction_position_in_impression(
    impressions: np.ndarray,
    item_id: int,
) -> np.ndarray:
    # This returns a tuple of arrays. First one is row_indices and second position is column indices.
    # given that impressions are 1D arrays, we only ened the first position of the tuple.
    return np.asarray(impressions == item_id).nonzero()[0]


@attr.frozen
class FinnNoSlatesConfig:
    """Configuration class used by data readers of the FINN.No dataset"""
    data_folder = os.path.join(
        ".",
        "data",
        "FINN-NO-SLATE",
    )

    num_raw_data_points = 2_277_645
    num_time_steps = 20
    num_items_in_each_impression = 25
    num_data_points = 45_552_900  # MUST BE NUM_RAW_DATA_POINTS * NUM_TIME_STEPS

    # The dataset treats 0s and 1s as error, and no-clicks, respectively.
    min_item_id = 3

    min_number_of_interactions = _MIN_ITEM_ID
    binarize_impressions = True
    binarize_interactions = True
    keep_duplicates: T_KEEP = "first"


class FINNNoImpressionOrigin(Enum):
    """Enum indicating the types of impressions present in the FINN.No dataset"""
    UNDEFINED = 0
    SEARCH = 1
    RECOMMENDATION = 2


class FINNNoSlateRawData(ParquetDataMixin):
    """Class that reads the 'raw' FINN.No data from disk.

    We refer to raw data as the original representation of the data,
    without cleaning or much processing. The dataset originally has XYZ-fernando-debugger data points.
    """

    def __init__(
        self,
        config: FinnNoSlatesConfig,
    ):
        self._config = config

        self._original_dataset_folder = os.path.join(
            self._config.data_folder, "original",
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

        self.num_raw_data_points = self._config.num_raw_data_points
        self.num_time_steps = self._config.num_time_steps
        self.num_items_in_each_impression = self._config.num_items_in_each_impression
        self.num_data_points = self.num_raw_data_points * self.num_time_steps

        self._data_loaded = False

    @property  # type: ignore
    @typed_cache
    def raw_data(self) -> pd.DataFrame:
        return self.load_parquet(
            file_path=self._original_dataset_pandas_file,
            to_pandas_func=self._raw_data_to_pandas
        )

    def _raw_data_to_pandas(self):
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

            assert (self.num_raw_data_points,) == user_ids.shape
            assert (self.num_raw_data_points, self.num_time_steps) == item_ids.shape
            assert (self.num_raw_data_points, self.num_time_steps) == click_indices_in_impressions.shape
            assert (self.num_raw_data_points, self.num_time_steps) == impression_origins.shape
            assert (self.num_raw_data_points, self.num_time_steps) == impressions_length_list.shape
            assert (self.num_raw_data_points, self.num_time_steps, self.num_items_in_each_impression) == \
                   impressions.shape

            user_id_arr = user_ids.repeat(
                repeats=self.num_time_steps,
            )
            item_id_arr = item_ids.reshape(
                (self.num_data_points,)
            )
            time_step_arr = np.array(
                [range(self.num_time_steps)] * user_ids.shape[0]
            ).reshape(
                (self.num_data_points,)
            )
            impression_origin_arr = impression_origins.reshape(
                (self.num_data_points,)
            )
            position_interactions_arr = click_indices_in_impressions.reshape(
                (self.num_data_points,)
            )
            impressions_arr = impressions.reshape(
                (self.num_data_points, self.num_items_in_each_impression)
            )
            num_impressions_arr = impressions_length_list.reshape(
                (self.num_data_points, )
            )
            is_item_in_impression_arr = is_item_in_impression(
                impressions=impressions_arr,
                item_ids=item_id_arr,
            )

            assert (self.num_data_points,) == user_id_arr.shape
            assert (self.num_data_points,) == item_id_arr.shape
            assert (self.num_data_points,) == time_step_arr.shape
            assert (self.num_data_points,) == position_interactions_arr.shape
            assert (self.num_data_points,) == impression_origin_arr.shape
            assert (self.num_data_points,) == num_impressions_arr.shape
            assert (self.num_data_points,) == is_item_in_impression_arr.shape
            assert (self.num_data_points, self.num_items_in_each_impression) == impressions_arr.shape

            df_data = pd.DataFrame(
                data={
                    "user_id": user_id_arr,
                    "item_id": item_id_arr,
                    "time_step": time_step_arr,
                    "impressions": list(impressions_arr),  # Needed to include a list of lists as a series of lists
                    # into the dataframe.
                    "position_interaction": position_interactions_arr,
                    "impressions_origin": impression_origin_arr,
                    "num_impressions": num_impressions_arr,
                    "is_item_in_impression": is_item_in_impression_arr,
                },
            ).astype(
                dtype={
                    "user_id": "category",
                    "item_id": "category",
                    "time_step": "category",
                    "impressions_origin": "category",
                    "impressions": "object",
                    "position_interaction": np.int32,
                    "num_impressions": np.int32,
                    "is_item_in_impression": pd.BooleanDtype(),
                }
            )

            assert (self.num_data_points, 8) == df_data.shape
            #assert (1000000, 8) == df_data.shape

            return df_data

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


class FINNNoSlateDataFrames(Enum):
    INTERACTIONS = "INTERACTIONS"
    IMPRESSIONS = "IMPRESSIONS"
    INTERACTIONS_IMPRESSIONS_METADATA = "INTERACTIONS_IMPRESSIONS_METADATA"


# class DaskFinnNoSlateRawData:
#     """A class that reads the FINN.No data using Dask Dataframes."""
#
#     def __init__(
#         self,
#     ):
#         self.config = FinnNoSlatesConfig()
#         self.raw_data_loader = FINNNoSlateRawData()
#
#         self._dataset_folder = os.path.join(
#             self.config.data_folder, "dask", "original",
#         )
#
#         self.folder_interactions = os.path.join(
#             self._dataset_folder, "interactions_exploded", ""
#         )
#         self.folder_impressions = os.path.join(
#             self._dataset_folder, "impressions", ""
#         )
#         self.folder_impressions_metadata = os.path.join(
#             self._dataset_folder, "impressions_metadata", ""
#         )
#
#
#         self._interactions: Optional[dd.DataFrame] = None
#         self._impressions: Optional[dd.DataFrame] = None
#         self._interactions_impressions_metadata: Optional[dd.DataFrame] = None
#
#     @property
#     def interactions(self) -> dd.DataFrame:
#         """ Interactions Dask Dataframe.
#
#         The columns of the dataframe are:
#
#         user_id : np.int32
#         time_step : np.int32
#         item_id : np.int32
#             A value of 1 indicates no interactions_exploded with any item in the list. A value of 0 indicates error.
#         """
#         if self._interactions is None:
#             self._interactions = self.load(
#                 df_name=FINNNoSlateDataFrames.INTERACTIONS
#             )
#
#         return self._interactions
#
#     @property
#     def impressions(self) -> dd.DataFrame:
#         if self._impressions is None:
#             self._impressions = self.load(
#                 df_name=FINNNoSlateDataFrames.IMPRESSIONS
#             )
#
#         return self._impressions
#
#     @property
#     def interactions_impressions_metadata(self) -> dd.DataFrame:
#         if self._interactions_impressions_metadata is None:
#             self._interactions_impressions_metadata = self.load(
#                 df_name=FINNNoSlateDataFrames.INTERACTIONS_IMPRESSIONS_METADATA
#             )
#
#         return self._interactions_impressions_metadata
#
#     @timeit
#     def load(self, df_name: FINNNoSlateDataFrames) -> dd.DataFrame:
#         """Returns the Finn.NO data in three ~dask.DataFrame dataframes.
#
#         Parameters
#         ----------
#         df_name
#             An member of the enum `FINNNoSlateDataFrames` indicating which dataframe to return.
#
#         Returns
#         -------
#         dask.DataFrame
#             A dataframe containing the requested record by `df_name`.
#
#         Raises
#         ------
#         ValueError
#             If parameter `df_name` is not a valid `FINNNoSlateDataFrames` member.
#         """
#
#         self.to_dask()
#
#         if df_name == FINNNoSlateDataFrames.INTERACTIONS:
#             return dd.read_parquet(
#                 path=self.folder_interactions,
#                 engine=FinnNoSlatesConfig.parquet_engine,
#             )
#
#         elif df_name == FINNNoSlateDataFrames.IMPRESSIONS:
#             dd.read_parquet(
#                 path=self.folder_impressions,
#                 engine=FinnNoSlatesConfig.parquet_engine,
#             ),
#
#         elif df_name == FINNNoSlateDataFrames.IMPRESSIONS:
#             dd.read_parquet(
#                 path=self.folder_impressions_metadata,
#                 engine=FinnNoSlatesConfig.parquet_engine,
#             )
#
#         raise ValueError(
#             f"{df_name=} not valid. Valid values are {list(FINNNoSlateDataFrames)}"
#         )
#
#     @timeit
#     def to_dask(self) -> None:
#         if (
#             os.path.exists(self.folder_interactions)
#             and len(os.listdir(self.folder_interactions)) > 0
#             and os.path.exists(self.folder_impressions)
#             and len(os.listdir(self.folder_impressions)) > 0
#             and os.path.exists(self.folder_impressions_metadata)
#             and len(os.listdir(self.folder_impressions_metadata)) > 0
#         ):
#             return
#
#         self.raw_data_loader.load_data()
#
#         index_arr = np.arange(
#             start=0,
#             step=1,
#             stop=self.raw_data_loader.num_data_points,
#         )
#
#         @timeit
#         def interactions_to_dask() -> None:
#             dd.from_pandas(
#                 data=pd.DataFrame(
#                     data={
#                         "user_id": self.raw_data_loader.user_id_arr,
#                         "time_step": self.raw_data_loader.time_step_arr,
#                         "item_id": self.raw_data_loader.item_id_arr,
#                     },
#                     index=index_arr,
#                 ).astype(
#                     dtype={
#                         "user_id": np.int32,
#                         "time_step": np.int32,
#                         "item_id": np.int32,
#                     },
#                 ),
#                 npartitions=100,
#             ).to_parquet(
#                 path=self.folder_interactions,
#                 engine=self.config.parquet_engine,
#             )
#
#         @timeit
#         def impressions_to_dask() -> None:
#             dd.from_pandas(
#                 data=pd.DataFrame(
#                     data=self.raw_data_loader.impressions_arr,
#                     index=index_arr,
#                     columns=[
#                         f"pos_{i}"
#                         for i in range(
#                             self.raw_data_loader.num_items_in_each_impression
#                         )
#                     ],
#                 ).astype(
#                     np.int32,
#                 ),
#                 npartitions=100,
#             ).to_parquet(
#                 path=self.folder_impressions,
#                 engine=self.config.parquet_engine,
#             )
#
#         @timeit
#         def impressions_metadata_to_dask() -> None:
#             dd.from_pandas(
#                 data=pd.DataFrame(
#                     data={
#                         "click_idx_in_impression": self.raw_data_loader.click_idx_in_impression_arr,
#                         "impression_type": self.raw_data_loader.impression_type_arr,
#                         "item_in_impression": self.raw_data_loader.item_in_impression_arr,
#                     },
#                     index=index_arr,
#                 ).astype(
#                     dtype={
#                         "click_idx_in_impression": np.int32,
#                         "impression_type": np.int32,
#                         "item_in_impression": np.bool,
#                     },
#                 ),
#                 npartitions=100,
#             ).to_parquet(
#                 path=self.folder_impressions_metadata,
#                 engine=self.config.parquet_engine,
#             )
#
#         interactions_to_dask()
#         impressions_metadata_to_dask()
#         impressions_to_dask()


class PandasFinnNoSlateRawData(ParquetDataMixin):
    """A class that reads the FINN.No data using Pandas Dataframes."""

    def __init__(
        self,
        config: FinnNoSlatesConfig
    ):
        self.config = config
        self.raw_data_loader = FINNNoSlateRawData(
            config=config,
        )

        self._dataset_folder = os.path.join(
            self.config.data_folder, "pandas", "original",
        )
        self.file_interactions = os.path.join(
            self._dataset_folder, "interactions.parquet"
        )
        self.file_impressions = os.path.join(
            self._dataset_folder, "impressions.parquet"
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
        return self.raw_data_loader.raw_data

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
        return self.load_parquet(
            to_pandas_func=self._interactions_to_pandas,
            file_path=self.file_interactions,
        )

    @property  # type: ignore
    @typed_cache
    def impressions(self) -> pd.DataFrame:
        return self.load_parquet(
            to_pandas_func=self._impressions_to_pandas,
            file_path=self.file_impressions,
        )

    @property  # type: ignore
    @typed_cache
    def interactions_impressions_metadata(self) -> pd.DataFrame:
        return self.load_parquet(
            to_pandas_func=self._impressions_metadata_to_pandas,
            file_path=self.file_impressions_metadata,
        )

    @timeit
    def _interactions_to_pandas(self) -> None:
        return self._df_train_and_validation[
            ["user_id", "time_step", "item_id"]
        ]

    @timeit
    def _impressions_to_pandas(self) -> None:
        return self._df_train_and_validation[
            ["user_id", "time_step", "impressions"]
        ]

    @timeit
    def _impressions_metadata_to_pandas(self) -> None:
        return self._df_train_and_validation[
            ["user_id", "time_step", "impressions_origin", "position_interaction", "num_impressions", "is_item_in_impression"]
        ]


# class StatisticsFinnNoSlate:
#     def __init__(self):
#         self.data_loader = DaskFinnNoSlateRawData()
#
#         (
#             self.dd_interactions,
#             self.dd_impressions,
#             self.dd_impressions_metadata,
#         ) = (
#             self.data_loader.interactions,
#             self.data_loader.impressions,
#             self.data_loader.interactions_impressions_metadata,
#         )
#
#         self.filters_boolean = [
#             ("full", slice(None)),
#             # ("no-data", self.dd_interactions["item_id"] == 0),
#             ("no-clicks", self.dd_interactions["item_id"] == 1),
#             ("interactions_exploded-&-no-clicks", self.dd_interactions["item_id"] > 0),
#             ("interactions_exploded", self.dd_interactions["item_id"] > 1),
#             # ("only-recommendations",
#             #  self.dd_impressions_metadata["impression_type"] == FINNNoImpressionOrigin.RECOMMENDATION.value),
#             # ("only-search", self.dd_impressions_metadata["impression_type"] == FINNNoImpressionOrigin.SEARCH.value),
#             # ("only-undefined", self.dd_impressions_metadata["impression_type"] == FINNNoImpressionOrigin.UNDEFINED.value),
#         ]
#
#     def statistics_interactions(self) -> None:
#         # if os.path.exists("data/FINN-NO-SLATE/statistics/interactions_exploded.zip"):
#         #     return
#
#         interactions: dd.DataFrame = dd.concat(
#             [self.dd_interactions, self.dd_impressions_metadata],
#             axis="columns",
#         )
#
#         statistics_: dict[str, Any] = {}
#
#         datasets: list[tuple[str, dd.DataFrame]] = [
#             (name, interactions[filter_boolean])
#             for name, filter_boolean in self.filters_boolean
#         ] + [
#             (
#                 f"{name}_no_dup",
#                 interactions[filter_boolean].drop_duplicates(
#                     subset=["user_id", "item_id"],
#                     # keep="first",
#                     ignore_index=False,
#                 ),
#             )
#             for name, filter_boolean in self.filters_boolean
#         ]
#
#         columns_for_unique = [
#             "user_id",
#             "item_id",
#             "time_step",
#             "click_idx_in_impression",
#             "impression_type",
#             "item_in_impression",
#         ]
#
#         columns_for_profile_length = [
#             "user_id",
#             "item_id",
#             "time_step",
#             "click_idx_in_impression",
#             "impression_type",
#             "item_in_impression",
#         ]
#
#         columns_for_gini = [
#             "user_id",
#             "item_id",
#             "time_step",
#             "click_idx_in_impression",
#             "impression_type",
#             "item_in_impression",
#         ]
#
#         columns_to_group_by = [
#             ("user_id", "item_id"),
#             ("item_id", "user_id"),
#             ("time_step", "user_id"),
#             ("click_idx_in_impression", "user_id"),
#             ("impression_type", "user_id"),
#             ("item_in_impression", "user_id"),
#         ]
#
#         logger.info(f"Calculating statistics for several datasets.")
#         name: str
#         dataset_: dd.DataFrame
#         series_column: dd.Series
#         for name, dataset_ in tqdm(datasets):
#             statistics_[name] = dict()
#
#             statistics_[name]["num_records"] = dataset_.shape[0]
#             # statistics_[name]["describe"] = delayed(
#             #     dataset_.astype({
#             #         "user_id": 'category',
#             #         "item_id": 'category',
#             #         "time_step": 'category',
#             #         "impression_type": 'category',
#             #         "click_idx_in_impression": 'category',
#             #     }).describe(
#             #         include="all"
#             #     ).astype({
#             #         # Given that this type is boolean, and the describe mixes ints with boolean values, then it must
#             #         # be parsed to either string or ints.
#             #         "item_in_impression": pd.StringDtype(),
#             #     })
#             # )
#
#             # for column in columns_for_unique:
#             #     if column not in statistics_[name]:
#             #         statistics_[name][column] = dict()
#             #
#             #     series_column = dataset_[column]
#             #
#             #     statistics_[name][column][f"num_unique"] = series_column.nunique()
#             #     statistics_[name][column][f"unique"] = series_column.unique()
#             #
#             for column in columns_for_profile_length:
#                 if column not in statistics_[name]:
#                     statistics_[name][column] = dict()
#
#                 series_column = dataset_[column]
#
#                 statistics_[name][column][f"profile_length"] = (
#                     series_column.value_counts(
#                         ascending=False,
#                         sort=False,
#                         normalize=False,
#                         dropna=True,
#                     )
#                     .rename("profile_length")
#                     .to_frame()
#                 )
#
#                 statistics_[name][column][f"profile_length_normalized"] = (
#                     series_column.value_counts(
#                         ascending=False,
#                         sort=False,
#                         normalize=True,
#                         dropna=True,
#                     )
#                     .rename("profile_length_normalized")
#                     .to_frame()
#                 )
#
#             # for column in columns_for_gini:
#             #     if column not in statistics_[name]:
#             #         statistics_[name][column] = dict()
#             #
#             #     series_column = dataset_[column]
#             #
#             #     # notna is there because columns might be NA.
#             #     statistics_[name][column]["gini_index_values_labels"] = delayed(
#             #         gini_index
#             #     )(array=series_column)
#             #     statistics_[name][column]["gini_index_values_counts"] = delayed(
#             #         gini_index
#             #     )(
#             #         array=series_column.value_counts(
#             #             dropna=True,
#             #             normalize=False,
#             #         ),
#             #     )
#             #
#             # for column_to_group_by, column_for_statistics in columns_to_group_by:
#             #     if column_to_group_by not in statistics_[name]:
#             #         statistics_[name][column_to_group_by] = dict()
#             #
#             #     df_group_by = dataset_.groupby(
#             #         by=[column_to_group_by],
#             #     )
#             #
#             # statistics_[name][column_to_group_by][f"group_by_profile_length"] = df_group_by[
#             #     column_for_statistics
#             # ].count()
#             # statistics_[name][column_to_group_by][f"group_by_describe"] = delayed(
#             #     df_group_by.agg([
#             #         "min",
#             #         "max",
#             #         "count",
#             #         "size",
#             #         "first",
#             #         "last",
#             #         # "var",
#             #         # "std",
#             #         # "mean",
#             #     ])
#             # )
#
#             # Create URM using original indices.
#             # num_users = dataset_["user_id"].max() + 1
#             # num_items = dataset_["item_id"].max() + 1
#             #
#             # row_indices = dataset_["user_id"]
#             # col_indices = dataset_["item_id"]
#             # data = delayed(np.ones_like)(
#             #     a=row_indices,
#             #     dtype=np.int32,
#             # )
#             # # assert row_indices.shape == col_indices.shape and row_indices.shape == data.shape
#             #
#             # urm_all_csr: sp.csr_matrix = delayed(sp.csr_matrix)(
#             #     (
#             #         data,
#             #         (row_indices, col_indices)
#             #     ),
#             #     shape=(num_users, num_items),
#             #     dtype=np.int32,
#             # )
#             #
#             # statistics_[name]["urm_all"] = dict()
#             # statistics_[name]["urm_all"]["matrix"] = urm_all_csr
#             #
#             # user_profile_length: np.ndarray = delayed(np.ediff1d)(urm_all_csr.indptr)
#             # user_profile_stats: DescribeResult = delayed(
#             #     st.describe
#             # )(
#             #     a=user_profile_length,
#             #     axis=0,
#             #     nan_policy="raise",
#             # )
#             # statistics_[name]["urm_all"]["interactions_by_users"] = {
#             #     "num_observations": user_profile_stats.nobs,
#             #     "min": user_profile_stats.minmax[0],
#             #     "max": user_profile_stats.minmax[1],
#             #     "mean": user_profile_stats.mean,
#             #     "variance": user_profile_stats.variance,
#             #     "skewness": user_profile_stats.skewness,
#             #     "kurtosis": user_profile_stats.kurtosis,
#             #     "gini_index": delayed(gini_index)(
#             #         array=user_profile_length,
#             #     ),
#             # }
#             #
#             # urm_all_csc: sp.csc_matrix = urm_all_csr.tocsc()
#             # item_profile_length: np.ndarray = delayed(np.ediff1d)(urm_all_csc.indptr)
#             # item_profile_stats: DescribeResult = delayed(
#             #     st.describe
#             # )(
#             #     a=item_profile_length,
#             #     axis=0,
#             #     nan_policy="omit",
#             # )
#             # statistics_[name]["urm_all"]["interactions_by_items"] = {
#             #     "num_observations": item_profile_stats.nobs,
#             #     "min": item_profile_stats.minmax[0],
#             #     "max": item_profile_stats.minmax[1],
#             #     "mean": item_profile_stats.mean,
#             #     "variance": item_profile_stats.variance,
#             #     "skewness": item_profile_stats.skewness,
#             #     "kurtosis": item_profile_stats.kurtosis,
#             #     "gini_index": delayed(gini_index)(
#             #         array=item_profile_length
#             #     ),
#             # }
#
#         logger.info("Sending compute to cluster.")
#         computed_statistics = dask_interface._client.compute(statistics_).result()
#
#         logger.info("Saving computed statistics.")
#
#         print(
#             computed_statistics["interactions_exploded"]["user_id"]["profile_length"],
#             type(computed_statistics["interactions_exploded"]["user_id"]["profile_length"]),
#         )
#         print(
#             computed_statistics["interactions_no_dup"]["user_id"]["profile_length"],
#             type(
#                 computed_statistics["interactions_no_dup"]["user_id"]["profile_length"]
#             ),
#         )
#
#         print(
#             f'{np.array_equal(computed_statistics["interactions_exploded"]["user_id"]["profile_length"].index, computed_statistics["interactions_no_dup"]["user_id"]["profile_length"].index)=}'
#         )
#
#         import pdb
##
#         data_io = DataIO(folder_path="data/FINN-NO-SLATE/statistics/")
#         data_io.save_data(
#             file_name="interactions_exploded.zip",
#             data_dict_to_save=computed_statistics,
#         )
#
#     def statistics_impressions(self):
#         pass
#
#     def statistics_impressions_metadata(self):
#         # if os.path.exists("data/FINN-NO-SLATE/statistics/impressions_metadata.zip"):
#         #     return
#
#         interactions = dd.concat(
#             [self.dd_interactions, self.dd_impressions_metadata],
#             axis="columns",
#         )
#
#         statistics_: dict[str, Any] = {}
#
#         columns_to_calculate_normalizations = [
#             "click_idx_in_impression",
#             "impression_type",
#         ]
#
#         columns_for_gini = [
#             "impression_type",
#             "click_idx_in_impression",
#         ]
#
#         columns_to_compare = [
#             ("impression_type", "click_idx_in_impression"),
#             ("click_idx_in_impression", "impression_type"),
#         ]
#
#         columns_to_group_by = [
#             ("impression_type", "click_idx_in_impression"),
#             ("click_idx_in_impression", "impression_type"),
#         ]
#
#         name: str
#         dataset_: dd.DataFrame
#         series_column: dd.Series
#         for name, dataset_ in tqdm(datasets):
#             statistics_[name] = dict()
#
#             for column in columns_to_calculate_normalizations:
#                 if column not in statistics_[name]:
#                     statistics_[name][column] = dict()
#
#                 series_column = dataset_[column]
#
#                 statistics_[name][column]["normalized"] = series_column.value_counts(
#                     dropna=False,
#                     normalize=True,
#                 )
#                 statistics_[name][column][
#                     "non-normalized"
#                 ] = series_column.value_counts(
#                     dropna=False,
#                     normalize=False,
#                 )
#
#             for column in columns_for_gini:
#                 if column not in statistics_[name]:
#                     statistics_[name][column] = dict()
#
#                 series_column = dataset_[column]
#
#                 # notna is there because columns might be NA.
#                 statistics_[name][column]["gini_index"] = delayed(gini_index)(
#                     array=delayed(
#                         series_column.values.compute
#                     )(),  # .to_numpy(copy=True)
#                 )
#                 # dropna needed because columns might be NA.
#                 statistics_[name][column][
#                     f"unique_num_{column}"
#                 ] = series_column.nunique(
#                     # dropna=True
#                 )
#
#             for column, other_col in columns_to_compare:
#                 # This might contain duplicates. Do not normalize it because we're interested in absolute counts.
#                 series_column = dataset_[column]
#
#                 statistics_[name][column][
#                     f"profile_length_{column}"
#                 ] = series_column.value_counts(
#                     dropna=True,
#                     normalize=False,
#                 )
#                 # This might contain duplicates. Do not normalize it because we're interested in absolute counts.
#                 statistics_[name][column][
#                     f"profile_length_{column}_without_duplicates"
#                 ] = dataset_.drop_duplicates(
#                     subset=[column, other_col],
#                     # inplace=False,
#                 )[
#                     column
#                 ].value_counts(
#                     dropna=True,
#                     normalize=False,
#                 )
#
#             for column_to_group_by, column_for_statistics in columns_to_group_by:
#                 if column_to_group_by not in statistics_[name]:
#                     statistics_[name][column_to_group_by] = dict()
#
#                 df_group_by = dataset_.groupby(
#                     by=[column_to_group_by],
#                 )
#
#                 profile_length = df_group_by[column_for_statistics].count()
#                 statistics_[name][column_to_group_by][
#                     f"group_by_profile_length"
#                 ] = profile_length
#                 statistics_[name][column_to_group_by][
#                     f"group_by_describe"
#                 ] = profile_length.describe()
#
#         computed_statistics = dask_interface._client.compute(statistics_).result()
#
#         print(computed_statistics)
#
#         data_io = DataIO(folder_path="data/FINN-NO-SLATE/statistics/")
#         data_io.save_data(
#             file_name="impressions_metadata.zip",
#             data_dict_to_save=computed_statistics,
#         )


class FINNNoSlateReader(DataReader):
    IS_IMPLICIT = True

    def __init__(
        self,
        config: FinnNoSlatesConfig,
    ):
        super().__init__()

        self.config = config

        self.DATA_FOLDER = os.path.join(
            self.config.data_folder, "data_reader", "",
        )
        self.ORIGINAL_SPLIT_FOLDER = self.DATA_FOLDER
        self.DATASET_SUBFOLDER = "FINN-NO-SLATE/"
        self.IS_IMPLICIT = self.config.binarize_interactions
        self._DATA_READER_NAME = "FINNNoSlateReader"

        self._keep_duplicates = self.config.keep_duplicates
        self._min_number_of_interactions = self.config.min_number_of_interactions
        self._binarize_impressions = self.config.binarize_impressions
        self._binarize_interactions = self.config.binarize_interactions
        self._num_parts_split_dataset = 500

        self._raw_data_loader = PandasFinnNoSlateRawData(
            config=config,
        )

        self._user_id_to_index_mapper: dict[int, int] = dict()
        self._item_id_to_index_mapper: dict[int, int] = dict()

        self._interactions: dict[str, sp.csr_matrix] = dict()
        self._impressions: dict[str, sp.csr_matrix] = dict()

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

    @property  # type: ignore
    @typed_cache
    def _data_filtered(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info(
            f"Filtering data sources (interactions, impressions, metadata)."
        )

        df_interactions = self._raw_data_loader.interactions.copy()
        df_impressions = self._raw_data_loader.impressions.copy()
        df_metadata = self._raw_data_loader.interactions_impressions_metadata.copy()

        # We don't need to reset the index first because this dataset does not have exploded values.
        df_interactions = df_interactions.sort_values(
            by=["time_step"],
            ascending=True,
            axis="index",
            inplace=False,
            ignore_index=False,
        )

        # This filter removes error logs and non-interactions of the dataset.
        df_interactions, _ = remove_records_by_threshold(
            df=df_interactions,
            column="item_id",
            threshold=self.config.min_item_id,
        )

        df_interactions, _ = remove_duplicates_in_interactions(
            df=df_interactions,
            columns_to_compare=["user_id", "item_id"],
            keep=self._keep_duplicates,
        )

        df_interactions, _ = remove_users_without_min_number_of_interactions(
            df=df_interactions,
            users_column="user_id",
            min_number_of_interactions=self._min_number_of_interactions,
        )

        df_impressions, _ = filter_impressions_by_interactions_index(
            df_impressions=df_impressions,
            df_interactions=df_interactions,
        )

        # This filter removes error logs and non-interactions of the dataset.
        df_impressions, _ = apply_custom_function(
            df=df_impressions,
            column="impressions",
            func=remove_non_clicks_on_impressions,
            func_name=remove_non_clicks_on_impressions.__name__,
            axis="columns",
        )

        # Given that we removed the 0s and 1s in the impressions, then we must substract
        df_metadata["position_interaction"] -= 1
        df_metadata["num_impressions"] -= 1
        # We don't have to process the column "is_item_in_impression" in the metadata because we already removed the
        # non-clicks on the dataset interactions and impressions.

        return df_interactions, df_impressions, df_metadata

    @property  # type: ignore
    @typed_cache
    def _data_time_step_split(
        self
    ) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame,
        pd.DataFrame, pd.DataFrame, pd.DataFrame,
        pd.DataFrame, pd.DataFrame, pd.DataFrame
    ]:
        df_interactions_filtered, df_impressions_filtered, df_metadata_filtered = self._data_filtered

        described = df_interactions_filtered["time_step"].describe(
            datetime_is_numeric=True,
            percentiles=[0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )

        validation_threshold = described["80%"]
        test_threshold = described["90%"]

        df_interactions_train, df_interactions_test = split_sequential_train_test_by_column_threshold(
            df=df_interactions_filtered,
            column="time_step",
            threshold=test_threshold
        )

        df_interactions_train, df_interactions_validation = split_sequential_train_test_by_column_threshold(
            df=df_interactions_train,
            column="time_step",
            threshold=validation_threshold
        )

        df_impressions_train, _ = filter_impressions_by_interactions_index(
            df_impressions=df_impressions_filtered,
            df_interactions=df_interactions_train,
        )

        df_impressions_validation, _ = filter_impressions_by_interactions_index(
            df_impressions=df_impressions_filtered,
            df_interactions=df_interactions_validation,
        )

        df_impressions_test, _ = filter_impressions_by_interactions_index(
            df_impressions=df_impressions_filtered,
            df_interactions=df_interactions_test,
        )

        df_metadata_train, _ = filter_impressions_by_interactions_index(
            df_impressions=df_metadata_filtered,
            df_interactions=df_interactions_train,
        )

        df_metadata_validation, _ = filter_impressions_by_interactions_index(
            df_impressions=df_metadata_filtered,
            df_interactions=df_interactions_validation,
        )

        df_metadata_test, _ = filter_impressions_by_interactions_index(
            df_impressions=df_metadata_filtered,
            df_interactions=df_interactions_test,
        )

        return (
            df_interactions_train.copy(), df_interactions_validation.copy(), df_interactions_test.copy(),
            df_impressions_train.copy(), df_impressions_validation.copy(), df_impressions_test.copy(),
            df_metadata_train.copy(), df_metadata_validation.copy(), df_metadata_test.copy(),
        )

    @property  # type: ignore
    @typed_cache
    def _data_leave_last_k_out_split(
        self
    ) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame,
        pd.DataFrame, pd.DataFrame, pd.DataFrame,
        pd.DataFrame, pd.DataFrame, pd.DataFrame
    ]:
        df_interactions_filtered, df_impressions_filtered, df_metadata_filtered = self._data_filtered

        df_interactions_train, df_interactions_test = split_sequential_train_test_by_num_records_on_test(
            df=df_interactions_filtered,
            group_by_column="user_id",
            num_records_in_test=1,
        )

        df_interactions_train, df_interactions_validation = split_sequential_train_test_by_num_records_on_test(
            df=df_interactions_train,
            group_by_column="user_id",
            num_records_in_test=1,
        )

        df_impressions_train, _ = filter_impressions_by_interactions_index(
            df_impressions=df_impressions_filtered,
            df_interactions=df_interactions_train,
        )

        df_impressions_validation, _ = filter_impressions_by_interactions_index(
            df_impressions=df_impressions_filtered,
            df_interactions=df_interactions_validation,
        )

        df_impressions_test, _ = filter_impressions_by_interactions_index(
            df_impressions=df_impressions_filtered,
            df_interactions=df_interactions_test,
        )

        df_metadata_train, _ = filter_impressions_by_interactions_index(
            df_impressions=df_metadata_filtered,
            df_interactions=df_interactions_train,
        )

        df_metadata_validation, _ = filter_impressions_by_interactions_index(
            df_impressions=df_metadata_filtered,
            df_interactions=df_interactions_validation,
        )

        df_metadata_test, _ = filter_impressions_by_interactions_index(
            df_impressions=df_metadata_filtered,
            df_interactions=df_interactions_test,
        )

        return (
            df_interactions_train.copy(), df_interactions_validation.copy(), df_interactions_test.copy(),
            df_impressions_train.copy(), df_impressions_validation.copy(), df_impressions_test.copy(),
            df_metadata_train.copy(), df_metadata_validation.copy(), df_metadata_test.copy(),
        )

    def _get_dataset_name_root(self) -> str:
        return self.DATASET_SUBFOLDER

    def _load_from_original_file(self) -> BaseDataset:
        # IMPORTANT: calculate first the impressions, so we have all mappers created.
        self._calculate_uim_all()
        self._calculate_urm_all()

        self._calculate_uim_leave_last_k_out_splits()
        self._calculate_urm_leave_last_k_out_splits()

        self._calculate_uim_time_step_splits()
        self._calculate_urm_time_step_splits()

        return BaseDataset(
            dataset_name="FINNNoSlate",
            impressions=self._impressions,
            interactions=self._interactions,
            mapper_item_original_id_to_index=self._item_id_to_index_mapper,
            mapper_user_original_id_to_index=self._user_id_to_index_mapper,
            is_impressions_implicit=self._binarize_impressions,
            is_interactions_implicit=self._binarize_interactions,
        )

    def _calculate_urm_all(self):
        logger.info(
            f"Building URM with name {BaseDataset.NAME_URM_ALL}."
        )
        df_interactions_filtered, _, _ = self._data_filtered

        builder_urm_all = IncrementalSparseMatrix_FilterIDs(
            preinitialized_col_mapper=self._item_id_to_index_mapper,
            on_new_col="add",
            preinitialized_row_mapper=self._user_id_to_index_mapper,
            on_new_row="add"
        )

        users = df_interactions_filtered['user_id'].to_numpy()
        items = df_interactions_filtered['item_id'].to_numpy()
        data = np.ones_like(users, dtype=np.int32, )

        builder_urm_all.add_data_lists(
            row_list_to_add=users,
            col_list_to_add=items,
            data_list_to_add=data,
        )

        urm_all = builder_urm_all.get_SparseMatrix()
        if self._binarize_interactions:
            urm_all.data = np.ones_like(urm_all.data, dtype=np.int32)

        self._interactions[BaseDataset.NAME_URM_ALL] = urm_all

        self._user_id_to_index_mapper = builder_urm_all.get_row_token_to_id_mapper()
        self._item_id_to_index_mapper = builder_urm_all.get_column_token_to_id_mapper()

    def _calculate_uim_all(self):
        logger.info(
            f"Building UIM with name {BaseDataset.NAME_UIM_ALL}."
        )
        _, df_impressions_filtered, _ = self._data_filtered

        builder_impressions_all = IncrementalSparseMatrix_FilterIDs(
            preinitialized_col_mapper=self._item_id_to_index_mapper,
            on_new_col="add",
            preinitialized_row_mapper=self._user_id_to_index_mapper,
            on_new_row="add",
        )

        df_split_chunk: pd.DataFrame
        for df_split_chunk in tqdm(
            np.array_split(df_impressions_filtered, indices_or_sections=self._num_parts_split_dataset)
        ):
            # Explosions of empty lists in impressions are transformed into NAs, NA values must be removed before
            # being inserted into the csr_matrix.
            df_split_chunk = df_split_chunk.explode(
                column="impressions",
                ignore_index=False,
            )
            df_split_chunk = df_split_chunk[
                df_split_chunk["impressions"].notna()
            ]

            impressions = df_split_chunk["impressions"].to_numpy()
            users = df_split_chunk["user_id"].to_numpy()
            data = np.ones_like(impressions, dtype=np.int32)

            builder_impressions_all.add_data_lists(
                row_list_to_add=users,
                col_list_to_add=impressions,
                data_list_to_add=data,
            )

        uim_all = builder_impressions_all.get_SparseMatrix()
        if self._binarize_impressions:
            uim_all.data = np.ones_like(uim_all.data, dtype=np.int32)

        self._impressions[BaseDataset.NAME_UIM_ALL] = uim_all.copy()

        self._user_id_to_index_mapper = builder_impressions_all.get_row_token_to_id_mapper()
        self._item_id_to_index_mapper = builder_impressions_all.get_column_token_to_id_mapper()

    def _calculate_urm_leave_last_k_out_splits(self) -> None:
        df_train, df_validation, df_test, _, _, _, _, _, _ = self._data_leave_last_k_out_split

        names = [
            BaseDataset.NAME_URM_LEAVE_LAST_K_OUT_TRAIN,
            BaseDataset.NAME_URM_LEAVE_LAST_K_OUT_VALIDATION,
            BaseDataset.NAME_URM_LEAVE_LAST_K_OUT_TEST,
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

            urm_split = builder_urm_split.get_SparseMatrix()
            if self._binarize_interactions:
                urm_split.data = np.ones_like(urm_split.data, dtype=np.int32)

            self._interactions[name] = urm_split.copy()

    def _calculate_uim_leave_last_k_out_splits(self) -> None:
        _, _, _, df_train, df_validation, df_test, _, _, _ = self._data_leave_last_k_out_split

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
            builder_uim_split = IncrementalSparseMatrix_FilterIDs(
                preinitialized_col_mapper=self._item_id_to_index_mapper,
                on_new_col="ignore",
                preinitialized_row_mapper=self._user_id_to_index_mapper,
                on_new_row="ignore"
            )

            df_split_chunk: pd.DataFrame
            for df_split_chunk in tqdm(
                np.array_split(df_split, indices_or_sections=self._num_parts_split_dataset)
            ):
                df_split_chunk = df_split_chunk.explode(
                    column="impressions",
                    ignore_index=False,
                )
                df_split_chunk = df_split_chunk[
                    df_split_chunk["impressions"].notna()
                ]
                impressions = df_split_chunk["impressions"].to_numpy()
                users = df_split_chunk["user_id"].to_numpy()
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

    def _calculate_urm_time_step_splits(self) -> None:
        df_train, df_validation, df_test, _, _, _, _, _, _ = self._data_time_step_split

        names = [
            BaseDataset.NAME_URM_TIMESTAMP_TRAIN,
            BaseDataset.NAME_URM_TIMESTAMP_VALIDATION,
            BaseDataset.NAME_URM_TIMESTAMP_TEST,
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

            urm_split = builder_urm_split.get_SparseMatrix()
            if self._binarize_interactions:
                urm_split.data = np.ones_like(urm_split.data, dtype=np.int32)

            self._interactions[name] = urm_split.copy()

    def _calculate_uim_time_step_splits(self) -> None:
        _, _, _, df_train, df_validation, df_test, _, _, _ = self._data_time_step_split

        names = [
            BaseDataset.NAME_UIM_TIMESTAMP_TRAIN,
            BaseDataset.NAME_UIM_TIMESTAMP_VALIDATION,
            BaseDataset.NAME_UIM_TIMESTAMP_TEST,
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

            for df_split_chunk in tqdm(
                np.array_split(df_split, indices_or_sections=self._num_parts_split_dataset)
            ):
                # Explosions of empty lists in impressions are transformed into NAs, NA values must be removed before
                # being inserted into the csr_matrix.
                df_split_chunk = df_split_chunk.explode(
                    column="impressions",
                    ignore_index=False,
                )
                df_split_chunk = df_split_chunk[
                    df_split_chunk["impressions"].notna()
                ]
                impressions = df_split_chunk["impressions"].to_numpy()
                users = df_split_chunk["user_id"].to_numpy()
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


# def create_mapper(
#     values: pd.Series,
#     mapper_name: str,
# ) -> pd.DataFrame:
#     original_column_name = f"original_{mapper_name}_indices"
#     mapped_column_name = f"mapped_{mapper_name}_indices"
#
#     return (
#         pd.DataFrame(
#             data={
#                 original_column_name: values.unique(),
#             },
#         )
#         .sort_values(
#             by=[original_column_name],
#             ascending=True,
#             inplace=False,
#             ignore_index=True,  # Sorting unique values in ascending order.
#         )
#         .reset_index(
#             drop=False,
#             inplace=False,
#         )
#         .rename(
#             columns={"index": mapped_column_name},
#             inplace=False,
#         )
#     )


if __name__ == "__main__":
    # dask_interface = configure_dask_cluster()

    config = FinnNoSlatesConfig()

    raw_data_loader = FINNNoSlateRawData(
        config=config,
    )

    pandas_data_loader = PandasFinnNoSlateRawData(
        config=config,
    )

    print(pandas_data_loader._df_train_and_validation)

    print(pandas_data_loader.interactions_impressions_metadata)
    print(pandas_data_loader.interactions)
    print(pandas_data_loader.impressions)

    data_reader = FINNNoSlateReader(
        config=config,
    )

    dataset = data_reader.dataset

    print(dataset.get_loaded_UIM_names())
    print(dataset.get_loaded_URM_names())
    quit(255)

    # FINNNoSlateRawData().load_data()
    # quit(0)

    # DaskFinnNoSlateRawData().to_dask()
    # PandasFinnNoSlateRawData().to_pandas()
    # quit(0)

    finn_no_statistics = StatisticsFinnNoSlate()
    finn_no_statistics.statistics_interactions()
    # finn_no_statistics.statistics_impressions_metadata()
    quit(0)

    # data_reader = FINNNoSlateReader()
    # dataset = data_reader.dataset
    # statistics = data_reader.statistics

    # Create a training-validation-test split, for example by leave-1-out
    # This splitter requires the DataReader object and the number of elements to holdout
    # data_splitter = DataSplitter_leave_k_out(
    #     dataReader_object=data_reader,
    #     k_out_value=1,
    #     use_validation_set=True,
    #     leave_random_out=True,
    # )

    # The load_data function will split the data and save it in the desired folder.
    # Once the split is saved, further calls to the load_data will load the split data ensuring
    # you always use the same split
    # data_splitter.load_data(
    #     save_folder_path="./result_experiments/FINN-NO-SLATE/data-leave-1-random-out/"
    #     # save_folder_path="./result_experiments/FINN-NO-SLATE/data-leave-1-random/"
    # )
    #
    # # We can access the three URMs.
    # urm_train, urm_validation, urm_test = data_splitter.get_holdout_split()
