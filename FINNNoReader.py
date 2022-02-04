import os
from enum import Enum
from typing import Optional, cast, Any, Literal

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.sparse as sp
from dask.delayed import delayed
from numba import jit, prange
from numpy.lib.npyio import NpzFile
from recsys_framework.Data_manager.DataReader import DataReader
from recsys_framework.Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from recsys_framework.Data_manager.Dataset import Dataset, gini_index
from recsys_framework.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs
from recsys_framework.Recommenders.DataIO import DataIO
from recsys_framework.Utils.conf_dask import configure_dask_cluster
from recsys_framework.Utils.conf_logging import get_logger
from recsys_framework.Utils.decorators import timeit
from recsys_slates_dataset.data_helper import download_data_files as download_finn_no_slate_files
from scipy.stats.stats import DescribeResult
from tqdm import tqdm

tqdm.pandas()

logger = get_logger(
    logger_name=__file__,
)


@jit(nopython=True, parallel=False)
def is_item_in_impression(
    impressions: np.ndarray,
    item_ids: np.ndarray,
) -> np.ndarray:
    assert impressions.shape[0] == item_ids.shape[0]
    assert impressions.shape[1] == 25

    return np.array(
        [
            item_ids[idx] in impressions[idx]
            for idx in prange(impressions.shape[0])
        ],
    )


# Ensure `is_item_in_impression` is JIT-compiled with expected array type,
# to avoid time in compiling.
# JIT-compile numba functions
is_item_in_impression(
    impressions=np.array([range(25)]),
    item_ids=np.array([1])
)
is_item_in_impression(
    impressions=np.array([range(25)]),
    item_ids=np.array([1])
)


@jit(nopython=True, parallel=False)
def calculate_items_in_impressions(
    impressions: np.ndarray,
    item_ids: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    assert impressions.shape[0] == item_ids.shape[0]
    assert impressions.shape[1] == 25

    item_positions = []
    num_items_in_impressions = np.empty(
        shape=(impressions.shape[0], ),
        dtype=np.int32,
    )

    for idx in range(impressions.shape[0]):
        idx_positions = np.where(impressions[idx] == item_ids[idx])[0]

        item_positions.append(idx_positions)
        num_items_in_impressions[idx] = idx_positions.shape[0]

    return item_positions, num_items_in_impressions


class FinnNoSlatesConfig:
    data_folder = os.path.join(
        ".",
        "data",
        "FINN-NO-SLATE",
    )
    parquet_engine = "pyarrow"
    parquet_use_nullable_dtypes = True
    parquet_num_parts_by_user = 100
    parquet_num_parts_by_time_step = 20


class FINNNoImpressionOrigin(Enum):
    UNDEFINED = 0
    SEARCH = 1
    RECOMMENDATION = 2


class FINNNoSlateRawData:
    def __init__(
        self,
    ):
        self._config = FinnNoSlatesConfig

        self._original_dataset_folder = os.path.join(
            self._config.data_folder,
            "original",
        )
        self._original_dataset_data_file = os.path.join(
            self._original_dataset_folder,
            "data.npz",
        )
        self._original_dataset_mapper_file = os.path.join(
            self._original_dataset_folder,
            "ind2val.json",
        )
        self._original_dataset_item_attr_file = os.path.join(
            self._original_dataset_folder,
            "itemattr.npz",
        )
        self.num_time_steps = 20
        self.num_items_in_each_impression = 25
        self._data_loaded = False

    def load_data(self) -> None:
        if self._data_loaded:
            return

        self._download_dataset()

        with cast(
            NpzFile,
            np.load(file=self._original_dataset_data_file)
        ) as interactions:
            user_ids = cast(np.ndarray, interactions["userId"])
            item_ids = cast(np.ndarray, interactions["click"])
            click_indices_in_impressions = cast(np.ndarray, interactions["click_idx"])
            impressions = cast(np.ndarray, interactions["slate"])
            impression_types = cast(np.ndarray, interactions["interaction_type"])
            # Unreliable, not loaded
            # self.impressions_length_list = cast(np.ndarray, interactions["slate_lengths"])

            self.num_points = user_ids.shape[0] * self.num_time_steps
            self.user_id_arr = user_ids.repeat(
                repeats=self.num_time_steps,
            )
            self.item_id_arr = item_ids.reshape(
                (self.num_points,)
            )
            self.time_step_arr = np.array([range(20)] * user_ids.shape[0]).reshape(
                (self.num_points,)
            )
            self.impression_type_arr = impression_types.reshape(
                (self.num_points,)
            )
            self.click_idx_in_impression_arr = click_indices_in_impressions.reshape(
                (self.num_points,)
            )
            self.impressions_arr = impressions.reshape(
                (self.num_points, self.num_items_in_each_impression)
            )
            self.item_in_impression_arr = timeit(is_item_in_impression)(
                impressions=self.impressions_arr,
                item_ids=self.item_id_arr,
            )

            assert (self.num_points, ) == self.user_id_arr.shape
            assert (self.num_points, ) == self.item_id_arr.shape
            assert (self.num_points, ) == self.time_step_arr.shape
            assert (self.num_points, ) == self.click_idx_in_impression_arr.shape
            assert (self.num_points, ) == self.impression_type_arr.shape
            assert (self.num_points, ) == self.item_in_impression_arr.shape
            assert (self.num_points, 25) == self.impressions_arr.shape

        self._data_loaded = True

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

    @staticmethod
    def _clean_data(
        user_id: int,
        user_impression_list: np.ndarray,
        user_interaction_type_list: np.ndarray,
        user_clicked_list: np.ndarray,
        user_clicked_index_list: np.ndarray,
        time_step: int
    ) -> dict[str, Any]:
        item_id: Optional[int] = None
        item_in_impressions: bool
        item_interacted: bool
        item_position: Optional[int] = None

        # Originally all impressions have the "no-clicked" item in the first position
        # and zeroes at the end if the number of items is lower than 25.
        # We remove it alongside not need it.
        impressions: pd.DataFrame = user_impression_list[time_step]
        impressions = impressions[
            (impressions > 1)
        ]

        # Originally all impressions are of size 25, we subtract one from the reported length given that we remove
        # the "no-action" item from the impressions.
        # The number of impressions in the dataset (reported as slate_lengths) seem unreliable, therefore we
        # just calculate it again based on the length of the remaining impression.
        # num_impressions = interactions_data["slate_lengths"][data_idx][time_step] - 1
        impressions_length = impressions.shape[0]

        # Impression type
        impressions_type = user_interaction_type_list[time_step]

        # If the user clicked an item, then item_id is an integer, else None.
        click = cast(int, user_clicked_list[time_step])
        if click > 1:
            item_id = click

        # Some items are not in the impressions
        item_in_impressions = item_id in impressions

        # Originally if the user did not click any interaction, then the index is 1, we use a boolean flag to indicate
        # if the user interacted or not with any impression.
        item_interacted = item_id is not None

        # If this was a no-action, then it is None
        # else the position on screen - 1 because the original data counts the no-action as being in the impression_list
        if item_id is not None and item_in_impressions:
            item_position = user_clicked_index_list[time_step] - 1

        return {
            "user_id": user_id,
            "item_id": item_id,
            "time_step": time_step,
            "item_interacted": item_interacted,
            "item_in_impressions": item_in_impressions,
            "item_position": item_position,
            "impressions": impressions,
            "impressions_length": impressions_length,
            "impressions_type": impressions_type,
        }


class DaskFinnNoSlateRawData:
    def __init__(
        self,
    ):
        self.config = FinnNoSlatesConfig()
        self.raw_data_loader = FINNNoSlateRawData()

        self._dataset_folder = os.path.join(
            self.config.data_folder,
            "dask",
            "original",
        )

        self.folder_interactions = os.path.join(
            self._dataset_folder,
            "interactions",
            ""
        )
        self.folder_impressions = os.path.join(
            self._dataset_folder,
            "impressions",
            ""
        )
        self.folder_impressions_metadata = os.path.join(
            self._dataset_folder,
            "impressions_metadata",
            ""
        )

    @timeit
    def load(self) -> tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame]:
        self.to_dask()

        return (
            dd.read_parquet(
                path=self.folder_interactions,
                engine=FinnNoSlatesConfig.parquet_engine,
            ),
            dd.read_parquet(
                path=self.folder_impressions,
                engine=FinnNoSlatesConfig.parquet_engine,
            ),
            dd.read_parquet(
                path=self.folder_impressions_metadata,
                engine=FinnNoSlatesConfig.parquet_engine,
            ),
        )

    @timeit
    def to_dask(self) -> None:
        if (
            os.path.exists(self.folder_interactions) and len(os.listdir(self.folder_interactions)) > 0
            and os.path.exists(self.folder_impressions) and len(os.listdir(self.folder_impressions)) > 0
            and os.path.exists(self.folder_impressions_metadata) and len(os.listdir(self.folder_impressions_metadata)) > 0
        ):
            return

        self.raw_data_loader.load_data()

        index_arr = np.arange(
            start=0,
            step=1,
            stop=self.raw_data_loader.num_points,
        )

        @timeit
        def interactions_to_dask() -> None:
            dd.from_pandas(
                data=pd.DataFrame(
                    data={
                        "user_id": self.raw_data_loader.user_id_arr,
                        "time_step": self.raw_data_loader.time_step_arr,
                        "item_id": self.raw_data_loader.item_id_arr,
                    },
                    index=index_arr,
                ).astype(
                    dtype={
                        "user_id": np.int32,
                        "time_step": np.int32,
                        "item_id": np.int32,
                    },
                ),
                npartitions=100,
            ).to_parquet(
                path=self.folder_interactions,
                engine=self.config.parquet_engine,
            )

        @timeit
        def impressions_to_dask() -> None:
            dd.from_pandas(
                data=pd.DataFrame(
                    data=self.raw_data_loader.impressions_arr,
                    index=index_arr,
                    columns=[f"pos_{i}" for i in range(self.raw_data_loader.num_items_in_each_impression)]
                ).astype(
                    np.int32,
                ),
                npartitions=100,
            ).to_parquet(
                path=self.folder_impressions,
                engine=self.config.parquet_engine,
            )

        @timeit
        def impressions_metadata_to_dask() -> None:
            dd.from_pandas(
                data=pd.DataFrame(
                    data={
                        "click_idx_in_impression": self.raw_data_loader.click_idx_in_impression_arr,
                        "impression_type": self.raw_data_loader.impression_type_arr,
                        "item_in_impression": self.raw_data_loader.item_in_impression_arr,
                    },
                    index=index_arr,
                ).astype(
                    dtype={
                        "click_idx_in_impression": np.int32,
                        "impression_type": np.int32,
                        "item_in_impression": np.bool,
                    },
                ),
                npartitions=100,
            ).to_parquet(
                path=self.folder_impressions_metadata,
                engine=self.config.parquet_engine,
            )

        interactions_to_dask()
        impressions_metadata_to_dask()
        impressions_to_dask()


class PandasFinnNoSlateRawData:
    def __init__(
        self,
    ):
        self.config = FinnNoSlatesConfig()
        self.raw_data_loader = FINNNoSlateRawData()

        self._dataset_folder = os.path.join(
            self.config.data_folder,
            "pandas",
            "original",
        )
        self.file_interactions = os.path.join(
            self._dataset_folder,
            "interactions.parquet"
        )
        self.file_impressions = os.path.join(
            self._dataset_folder,
            "impressions.parquet"
        )
        self.file_impressions_metadata = os.path.join(
            self._dataset_folder,
            "impressions_metadata.parquet"
        )

        os.makedirs(
            name=self._dataset_folder,
            exist_ok=True,
        )

    @timeit
    def load(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.to_pandas()

        return (
            pd.read_parquet(
                path=self.file_interactions,
                engine=self.config.parquet_engine,
                use_nullable_dtypes=self.config.parquet_use_nullable_dtypes,
            ),
            pd.read_parquet(
                path=self.file_impressions,
                engine=self.config.parquet_engine,
                use_nullable_dtypes=self.config.parquet_use_nullable_dtypes,
            ),
            pd.read_parquet(
                path=self.file_impressions_metadata,
                engine=self.config.parquet_engine,
                use_nullable_dtypes=self.config.parquet_use_nullable_dtypes,
            ),
        )

    @timeit
    def to_pandas(self) -> None:
        if (
            os.path.isfile(self.file_interactions)
            and os.path.isfile(self.file_impressions)
            and os.path.isfile(self.file_impressions_metadata)
        ):
            return

        self.raw_data_loader.load_data()

        index_arr = np.arange(
            start=0,
            step=1,
            stop=self.raw_data_loader.num_points,
        )

        @timeit
        def interactions_to_pandas() -> None:
            pd.DataFrame(
                data={
                    "user_id": self.raw_data_loader.user_id_arr,
                    "time_step": self.raw_data_loader.time_step_arr,
                    "item_id": self.raw_data_loader.item_id_arr,
                },
                index=index_arr,
            ).astype(
                dtype={
                    "user_id": np.int32,
                    "time_step": np.int32,
                    "item_id": np.int32,
                },
            ).to_parquet(
                path=self.file_interactions,
                engine=self.config.parquet_engine,
            )

        @timeit
        def impressions_to_pandas() -> None:
            pd.DataFrame(
                data={"impression": list(self.raw_data_loader.impressions_arr)},
                index=index_arr,
                # columns=[f"pos_{i}" for i in range(self.raw_data_loader.num_impressions)]
            ).astype(
                {
                    "impression": "object"
                }  # np.int32,
            ).to_parquet(
                path=self.file_impressions,
                engine=self.config.parquet_engine,
            )

        @timeit
        def impressions_metadata_to_pandas() -> None:
            pd.DataFrame(
                data={
                    "click_idx_in_impression": self.raw_data_loader.click_idx_in_impression_arr,
                    "impression_type": self.raw_data_loader.impression_type_arr,
                    "item_in_impression": self.raw_data_loader.item_in_impression_arr,
                },
                index=index_arr,
            ).astype(
                dtype={
                    "click_idx_in_impression": np.int32,
                    "impression_type": np.int32,
                    "item_in_impression": np.bool,
                },
            ).to_parquet(
                path=self.file_impressions_metadata,
                engine=self.config.parquet_engine,
            )

        interactions_to_pandas()
        impressions_metadata_to_pandas()
        impressions_to_pandas()


class StatisticsFinnNoSlate:
    def __init__(self):
        self.data_loader = DaskFinnNoSlateRawData()
        # self.pd_interactions, self.pd_impressions, self.pd_impressions_metadata = PandasFinnNoSlateRawData().load()
        self.dd_interactions, self.dd_impressions, self.dd_impressions_metadata = DaskFinnNoSlateRawData().load()

        self.filters_boolean = [
            ("full", slice(None)),
            # ("no-data", self.dd_interactions["item_id"] == 0),
            ("no-clicks", self.dd_interactions["item_id"] == 1),
            ("interactions-&-no-clicks", self.dd_interactions["item_id"] > 0),
            ("interactions", self.dd_interactions["item_id"] > 1),
            # ("only-recommendations",
            #  self.dd_impressions_metadata["impression_type"] == FINNNoImpressionOrigin.RECOMMENDATION.value),
            # ("only-search", self.dd_impressions_metadata["impression_type"] == FINNNoImpressionOrigin.SEARCH.value),
            # ("only-undefined", self.dd_impressions_metadata["impression_type"] == FINNNoImpressionOrigin.UNDEFINED.value),
        ]

    def statistics_interactions(self) -> None:
        # if os.path.exists("data/FINN-NO-SLATE/statistics/interactions.zip"):
        #     return

        interactions: dd.DataFrame = dd.concat(
            [
                self.dd_interactions,
                self.dd_impressions_metadata
            ],
            axis="columns",
        )

        statistics_: dict[str, Any] = {}

        datasets: list[tuple[str, dd.DataFrame]] = [
            (name, interactions[filter_boolean])
            for name, filter_boolean in self.filters_boolean
        ] + [
            (f"{name}_no_dup", interactions[filter_boolean].drop_duplicates(
                subset=["user_id", "item_id"],
                # keep="first",
                ignore_index=False,
            ))
            for name, filter_boolean in self.filters_boolean
        ]

        columns_for_unique = [
            "user_id",
            "item_id",
            "time_step",
            "click_idx_in_impression",
            "impression_type",
            "item_in_impression",
        ]

        columns_for_profile_length = [
            "user_id",
            "item_id",
            "time_step",
            "click_idx_in_impression",
            "impression_type",
            "item_in_impression",
        ]

        columns_for_gini = [
            "user_id",
            "item_id",
            "time_step",
            "click_idx_in_impression",
            "impression_type",
            "item_in_impression",
        ]

        columns_to_group_by = [
            ("user_id", "item_id"),
            ("item_id", "user_id"),
            ("time_step", "user_id"),
            ("click_idx_in_impression", "user_id"),
            ("impression_type", "user_id"),
            ("item_in_impression", "user_id"),
        ]

        logger.info(
            f"Calculating statistics for several datasets."
        )
        name: str
        dataset_: dd.DataFrame
        series_column: dd.Series
        for name, dataset_ in tqdm(datasets):
            statistics_[name] = dict()

            statistics_[name]["num_records"] = dataset_.shape[0]
            # statistics_[name]["describe"] = delayed(
            #     dataset_.astype({
            #         "user_id": 'category',
            #         "item_id": 'category',
            #         "time_step": 'category',
            #         "impression_type": 'category',
            #         "click_idx_in_impression": 'category',
            #     }).describe(
            #         include="all"
            #     ).astype({
            #         # Given that this type is boolean, and the describe mixes ints with boolean values, then it must
            #         # be parsed to either string or ints.
            #         "item_in_impression": pd.StringDtype(),
            #     })
            # )

            # for column in columns_for_unique:
            #     if column not in statistics_[name]:
            #         statistics_[name][column] = dict()
            #
            #     series_column = dataset_[column]
            #
            #     statistics_[name][column][f"num_unique"] = series_column.nunique()
            #     statistics_[name][column][f"unique"] = series_column.unique()
            #
            for column in columns_for_profile_length:
                if column not in statistics_[name]:
                    statistics_[name][column] = dict()

                series_column = dataset_[column]

                statistics_[name][column][f"profile_length"] = series_column.value_counts(
                    ascending=False,
                    sort=False,
                    normalize=False,
                    dropna=True,
                ).rename(
                    "profile_length"
                ).to_frame()

                statistics_[name][column][f"profile_length_normalized"] = series_column.value_counts(
                    ascending=False,
                    sort=False,
                    normalize=True,
                    dropna=True,
                ).rename(
                    "profile_length_normalized"
                ).to_frame()

            # for column in columns_for_gini:
            #     if column not in statistics_[name]:
            #         statistics_[name][column] = dict()
            #
            #     series_column = dataset_[column]
            #
            #     # notna is there because columns might be NA.
            #     statistics_[name][column]["gini_index_values_labels"] = delayed(
            #         gini_index
            #     )(array=series_column)
            #     statistics_[name][column]["gini_index_values_counts"] = delayed(
            #         gini_index
            #     )(
            #         array=series_column.value_counts(
            #             dropna=True,
            #             normalize=False,
            #         ),
            #     )
            #
            # for column_to_group_by, column_for_statistics in columns_to_group_by:
            #     if column_to_group_by not in statistics_[name]:
            #         statistics_[name][column_to_group_by] = dict()
            #
            #     df_group_by = dataset_.groupby(
            #         by=[column_to_group_by],
            #     )
                #
                # statistics_[name][column_to_group_by][f"group_by_profile_length"] = df_group_by[
                #     column_for_statistics
                # ].count()
                # statistics_[name][column_to_group_by][f"group_by_describe"] = delayed(
                #     df_group_by.agg([
                #         "min",
                #         "max",
                #         "count",
                #         "size",
                #         "first",
                #         "last",
                #         # "var",
                #         # "std",
                #         # "mean",
                #     ])
                # )

            # Create URM using original indices.
            # num_users = dataset_["user_id"].max() + 1
            # num_items = dataset_["item_id"].max() + 1
            #
            # row_indices = dataset_["user_id"]
            # col_indices = dataset_["item_id"]
            # data = delayed(np.ones_like)(
            #     a=row_indices,
            #     dtype=np.int32,
            # )
            # # assert row_indices.shape == col_indices.shape and row_indices.shape == data.shape
            #
            # urm_all_csr: sp.csr_matrix = delayed(sp.csr_matrix)(
            #     (
            #         data,
            #         (row_indices, col_indices)
            #     ),
            #     shape=(num_users, num_items),
            #     dtype=np.int32,
            # )
            #
            # statistics_[name]["urm_all"] = dict()
            # statistics_[name]["urm_all"]["matrix"] = urm_all_csr
            #
            # user_profile_length: np.ndarray = delayed(np.ediff1d)(urm_all_csr.indptr)
            # user_profile_stats: DescribeResult = delayed(
            #     st.describe
            # )(
            #     a=user_profile_length,
            #     axis=0,
            #     nan_policy="raise",
            # )
            # statistics_[name]["urm_all"]["interactions_by_users"] = {
            #     "num_observations": user_profile_stats.nobs,
            #     "min": user_profile_stats.minmax[0],
            #     "max": user_profile_stats.minmax[1],
            #     "mean": user_profile_stats.mean,
            #     "variance": user_profile_stats.variance,
            #     "skewness": user_profile_stats.skewness,
            #     "kurtosis": user_profile_stats.kurtosis,
            #     "gini_index": delayed(gini_index)(
            #         array=user_profile_length,
            #     ),
            # }
            #
            # urm_all_csc: sp.csc_matrix = urm_all_csr.tocsc()
            # item_profile_length: np.ndarray = delayed(np.ediff1d)(urm_all_csc.indptr)
            # item_profile_stats: DescribeResult = delayed(
            #     st.describe
            # )(
            #     a=item_profile_length,
            #     axis=0,
            #     nan_policy="omit",
            # )
            # statistics_[name]["urm_all"]["interactions_by_items"] = {
            #     "num_observations": item_profile_stats.nobs,
            #     "min": item_profile_stats.minmax[0],
            #     "max": item_profile_stats.minmax[1],
            #     "mean": item_profile_stats.mean,
            #     "variance": item_profile_stats.variance,
            #     "skewness": item_profile_stats.skewness,
            #     "kurtosis": item_profile_stats.kurtosis,
            #     "gini_index": delayed(gini_index)(
            #         array=item_profile_length
            #     ),
            # }

        logger.info(
            "Sending compute to cluster."
        )
        computed_statistics = dask_interface._client.compute(
            statistics_
        ).result()

        logger.info(
            "Saving computed statistics."
        )

        print(computed_statistics["interactions"]["user_id"]["profile_length"], type(
            computed_statistics["interactions"]["user_id"]["profile_length"]))
        print(computed_statistics["interactions_no_dup"]["user_id"]["profile_length"], type(
            computed_statistics["interactions_no_dup"]["user_id"]["profile_length"]))

        print(
            f'{np.array_equal(computed_statistics["interactions"]["user_id"]["profile_length"].index, computed_statistics["interactions_no_dup"]["user_id"]["profile_length"].index)=}'
        )

        import pdb
        pdb.set_trace()

        data_io = DataIO(
            folder_path="data/FINN-NO-SLATE/statistics/"
        )
        data_io.save_data(
            file_name="interactions.zip",
            data_dict_to_save=computed_statistics,
        )

    def statistics_impressions(self):
        impressions = self.impressions

        datasets: list[tuple[str, dd.DataFrame]] = [
            ("full", impressions),
        ]

        unique_items = np.unique(
            impressions
        )

    def statistics_impressions_metadata(self):
        # if os.path.exists("data/FINN-NO-SLATE/statistics/impressions_metadata.zip"):
        #     return

        interactions = dd.concat(
            [self.dd_interactions, self.dd_impressions_metadata],
            axis="columns",
        )

        statistics_: dict[str, Any] = {}

        columns_to_calculate_normalizations = [
            "click_idx_in_impression",
            "impression_type",
        ]

        columns_for_gini = [
            "impression_type",
            "click_idx_in_impression",
        ]

        columns_to_compare = [
            ("impression_type", "click_idx_in_impression"),
            ("click_idx_in_impression", "impression_type"),
        ]

        columns_to_group_by = [
            ("impression_type", "click_idx_in_impression"),
            ("click_idx_in_impression", "impression_type"),
        ]

        name: str
        dataset_: dd.DataFrame
        series_column: dd.Series
        for name, dataset_ in tqdm(datasets):
            statistics_[name] = dict()

            for column in columns_to_calculate_normalizations:
                if column not in statistics_[name]:
                    statistics_[name][column] = dict()

                series_column = dataset_[column]

                statistics_[name][column]["normalized"] = series_column.value_counts(
                    dropna=False,
                    normalize=True,
                )
                statistics_[name][column]["non-normalized"] = series_column.value_counts(
                    dropna=False,
                    normalize=False,
                )

            for column in columns_for_gini:
                if column not in statistics_[name]:
                    statistics_[name][column] = dict()

                series_column = dataset_[column]

                # notna is there because columns might be NA.
                statistics_[name][column]["gini_index"] = delayed(gini_index)(
                    array=delayed(series_column.values.compute)(),  # .to_numpy(copy=True)
                )
                # dropna needed because columns might be NA.
                statistics_[name][column][f"unique_num_{column}"] = series_column.nunique(
                    # dropna=True
                )

            for column, other_col in columns_to_compare:
                # This might contain duplicates. Do not normalize it because we're interested in absolute counts.
                series_column = dataset_[column]

                statistics_[name][column][f"profile_length_{column}"] = series_column.value_counts(
                    dropna=True,
                    normalize=False,
                )
                # This might contain duplicates. Do not normalize it because we're interested in absolute counts.
                statistics_[name][column][f"profile_length_{column}_without_duplicates"] = dataset_.drop_duplicates(
                    subset=[column, other_col],
                    # inplace=False,
                )[
                    column
                ].value_counts(
                    dropna=True,
                    normalize=False,
                )

            for column_to_group_by, column_for_statistics in columns_to_group_by:
                if column_to_group_by not in statistics_[name]:
                    statistics_[name][column_to_group_by] = dict()

                df_group_by = dataset_.groupby(
                    by=[column_to_group_by],
                )

                profile_length = df_group_by[column_for_statistics].count()
                statistics_[name][column_to_group_by][f"group_by_profile_length"] = profile_length
                statistics_[name][column_to_group_by][f"group_by_describe"] = profile_length.describe()

        computed_statistics = dask_interface._client.compute(
            statistics_
        ).result()

        print(computed_statistics)

        data_io = DataIO(
            folder_path="data/FINN-NO-SLATE/statistics/"
        )
        data_io.save_data(
            file_name="impressions_metadata.zip",
            data_dict_to_save=computed_statistics,
        )

    def statistics(self) -> None:
        # Ensure the dataset is loaded.
        logger.debug(
            f"Statistics"
        )
        interactions, impressions = self.data_loader.load()

        interactions_impressions = self.load_dask().astype(
            dtype={
                "user_id": np.int32,
                "item_id": np.int32,
                "time_step": np.int32,
                "impressions": "object",
                "impressions_type": np.int32,
            }
        )

        def get_item_positions(df_row: pd.DataFrame) -> Any:
            item_positions = cast(
                np.ndarray,
                np.where(
                    df_row["impressions"] == df_row["item_id"]
                )[0]
            )

            return (
                item_positions,
                item_positions.shape[0],
            )

        interactions_impressions = cast(
            dd.DataFrame,
            interactions_impressions.sample(
                frac=0.1,
                replace=False,
                # weights=None,
                # ignore_index=True,
            )
        )

        interactions_impressions[["item_positions", "num_item_in_impressions"]] = interactions_impressions.apply(
            get_item_positions,
            axis="columns",
            result_type="expand",
            meta=[
                (0, "object"),
                (1, np.int32)
            ]
        ).astype(
            dtype={
                0: "object",
                1: np.int32,
            }
        )

        logger.debug(
            f"applied"
        )

        statistics_: dict[str, Any] = {}

        datasets: list[tuple[str, dd.DataFrame]] = [
            ("full", interactions_impressions),
            ("no-data", interactions_impressions[
                interactions_impressions["item_id"] == 0
                ]),
            ("non-interactions", interactions_impressions[
                interactions_impressions["item_id"] == 1
                ]),
            ("interactions", interactions_impressions[
                interactions_impressions["item_id"] > 1
                ]),
            ("only-recommendations", interactions_impressions[
                interactions_impressions["impressions_type"] == FINNNoImpressionOrigin.RECOMMENDATION.value
                ]),
            ("only-search", interactions_impressions[
                interactions_impressions["impressions_type"] == FINNNoImpressionOrigin.SEARCH.value
                ]),
        ]

        columns_to_calculate_normalizations = [
            "impressions_type",
            "num_item_in_impressions",
        ]

        columns_to_explode = [
            "impressions",
            "item_positions",
        ]

        columns_for_gini = [
            "user_id",
            "item_id",
            "time_step",
        ]

        columns_to_compare = [
            ("user_id", "item_id"),
            ("user_id", "time_step"),
            ("item_id", "user_id"),
            ("item_id", "time_step"),
        ]

        columns_to_group_by = [
            ("user_id", "item_id"),
            ("item_id", "user_id"),
        ]

        logger.info(
            f"Calculating statistics for several datasets."
        )
        name: str
        dataset_: dd.DataFrame
        series_column: dd.Series
        for name, dataset_ in tqdm(datasets):
            statistics_[name] = dict()

            for column in columns_to_calculate_normalizations:
                if column not in statistics_[name]:
                    statistics_[name][column] = dict()

                series_column = dataset_[column]

                statistics_[name][column]["normalized"] = series_column.value_counts(
                    dropna=False,
                    normalize=True,
                )
                statistics_[name][column]["non-normalized"] = series_column.value_counts(
                    dropna=False,
                    normalize=False,
                )

            for column in columns_for_gini:
                if column not in statistics_[name]:
                    statistics_[name][column] = dict()

                series_column = dataset_[column]

                # notna is there because columns might be NA.
                statistics_[name][column]["gini_index"] = delayed(gini_index)(
                    array=delayed(series_column.values.compute)(),  # .to_numpy(copy=True)
                )
                # dropna needed because columns might be NA.
                statistics_[name][column][f"unique_num_{column}"] = series_column.nunique(
                    # dropna=True
                )

            for column, other_col in columns_to_compare:
                # This might contain duplicates. Do not normalize it because we're interested in absolute counts.
                series_column = dataset_[column]

                statistics_[name][column][f"profile_length_{column}"] = series_column.value_counts(
                    dropna=True,
                    normalize=False,
                )
                # This might contain duplicates. Do not normalize it because we're interested in absolute counts.
                statistics_[name][column][f"profile_length_{column}_without_duplicates"] = dataset_.drop_duplicates(
                    subset=[column, other_col],
                    # inplace=False,
                )[
                    column
                ].value_counts(
                    dropna=True,
                    normalize=False,
                )

            for column in columns_to_explode:
                if column not in statistics_[name]:
                    statistics_[name][column] = dict()

                series_column = dataset_[column]

                exploded_series = cast(
                    dd.Series,
                    series_column.explode(
                        # ignore_index=True,
                    )
                )

                statistics_[name][column]["normalized"] = exploded_series.value_counts(
                    dropna=False,
                    normalize=True,
                )
                statistics_[name][column]["non-normalized"] = exploded_series.value_counts(
                    dropna=False,
                    normalize=False,
                )

            for column_to_group_by, column_for_statistics in columns_to_group_by:
                if column_to_group_by not in statistics_[name]:
                    statistics_[name][column_to_group_by] = dict()

                df_group_by = interactions_impressions.groupby(
                    by=[column_to_group_by],
                )

                profile_length = df_group_by[column_for_statistics].count()
                statistics_[name][column_to_group_by][f"group_by_profile_length"] = profile_length
                statistics_[name][column_to_group_by][f"group_by_describe"] = profile_length.describe()

            # Create URM using original indices.
            num_users = dataset_["user_id"].max() + 1
            num_items = dataset_["item_id"].max() + 1

            row_indices = dataset_["user_id"]
            col_indices = dataset_["item_id"]
            data = delayed(np.ones_like)(
                a=row_indices,
                dtype=np.int32,
            )
            # assert row_indices.shape == col_indices.shape and row_indices.shape == data.shape

            urm_all = delayed(sp.csr_matrix)(
                (
                    data,
                    (row_indices, col_indices)
                ),
                shape=(num_users, num_items),
                dtype=np.int32,
            )

            statistics_[name]["urm_all"] = dict()
            statistics_[name]["urm_all"]["matrix"] = urm_all

            user_profile_length = delayed(np.ediff1d)(urm_all.indptr)
            statistics_[name]["urm_all"]["interactions_by_users"] = {
                "max": user_profile_length.max(),
                "mean": user_profile_length.mean(),
                "min": user_profile_length.min(),
                "gini_index": delayed(gini_index)(
                    array=user_profile_length,
                ),
            }

            urm_all = urm_all.tocsc()
            item_profile_length = delayed(np.ediff1d)(urm_all.indptr)
            statistics_[name]["urm_all"]["interactions_by_items"] = {
                "max": item_profile_length.max(),
                "mean": item_profile_length.mean(),
                "min": item_profile_length.min(),
                "gini_index": delayed(gini_index)(
                    array=item_profile_length
                ),
            }

        computed_statistics = dask_interface._client.compute(
            statistics_
        ).result()

        data_io = DataIO(
            folder_path="data/bk-FINN-NO-SLATE/statistics/"
        )
        data_io.save_data(
            file_name="statistics.zip",
            data_dict_to_save=computed_statistics,
        )


class DatasetWithImpressions(Dataset):  # type: ignore
    DATASET_NAME: Optional[str] = None

    def __init__(
        self,
        URM_dictionary=None,
        ICM_dictionary=None,
        ICM_feature_mapper_dictionary=None,
        UCM_dictionary=None,
        UCM_feature_mapper_dictionary=None,
        user_original_ID_to_index=None,
        item_original_ID_to_index=None,
        is_implicit=False,
        additional_data_mapper=None,
        impressions_dictionary=None
    ):
        super().__init__(
            self,
            URM_dictionary=URM_dictionary,
            ICM_dictionary=ICM_dictionary,
            ICM_feature_mapper_dictionary=ICM_feature_mapper_dictionary,
            UCM_dictionary=UCM_dictionary,
            UCM_feature_mapper_dictionary=UCM_feature_mapper_dictionary,
            user_original_ID_to_index=user_original_ID_to_index,
            item_original_ID_to_index=item_original_ID_to_index,
            is_implicit=is_implicit,
            additional_data_mapper=additional_data_mapper,
        )
        self.impressions_dictionary = impressions_dictionary


class FINNNoSlateDataset(Dataset):
    DATASET_NAME = "FINNNoSlate"

    def __init__(
        self,
        urms: dict[str, sp.csr_matrix],
        # impressions: dict[str, sp.csr_matrix],
        user_id_to_index_mapper: dict[int, int],
        item_id_to_index_mapper: dict[int, int],
        is_implicit: bool,
    ):
        super().__init__(
            URM_dictionary=urms,
            ICM_dictionary=None,
            ICM_feature_mapper_dictionary=None,
            UCM_dictionary=None,
            UCM_feature_mapper_dictionary=None,
            user_original_ID_to_index=user_id_to_index_mapper,
            item_original_ID_to_index=item_id_to_index_mapper,
            is_implicit=is_implicit,
            # impressions_dictionary=impressions
        )


class FINNNoSlateReader(DataReader):
    _DATA_READER_NAME = "FINNNoSlateReader"

    DATASET_SUBFOLDER = "FINN-NO-SLATE/"

    IS_IMPLICIT = True

    _NAME_URM_STEPS = "URM_steps"
    _NAME_URM_ALL = "URM_all"
    _NAME_URM_ALL_BINARY = "URM_all"

    _NAME_RANDOM_URM_TRAIN = "URM_random_train"
    _NAME_RANDOM_URM_VALIDATION = "URM_random_validation"
    _NAME_RANDOM_URM_TEST = "URM_random_test"

    _NAME_SEQUENTIAL_URM_TRAIN = "URM_sequential_train"
    _NAME_SEQUENTIAL_URM_VALIDATION = "URM_sequential_validation"
    _NAME_SEQUENTIAL_URM_TEST = "URM_sequential_test"

    _NAME_IMPRESSIONS_ALL = "impressions_all"
    _NAME_IMPRESSIONS_STEPS = "impressions_steps"
    _NAME_SEQUENTIAL_IMPRESSIONS_TRAIN = "impressions_sequential_train"
    _NAME_SEQUENTIAL_IMPRESSIONS_VALIDATION = "impressions_sequential_validation"
    _NAME_SEQUENTIAL_IMPRESSIONS_TEST = "impressions_sequential_test"

    def __init__(
        self,
    ):
        super().__init__()

        self.config = FinnNoSlatesConfig()
        self.ORIGINAL_SPLIT_FOLDER = os.path.join(
            self.config.data_folder,
            self._NAME_URM_ALL_BINARY,
            ""
        )

        self._user_id_to_index_mapper: dict[int, int] = dict()
        self._item_id_to_index_mapper: dict[int, int] = dict()

        self._urms: dict[str, sp.csr_matrix] = dict()
        self._impressions: dict[str, sp.csr_matrix] = dict()

        self._icms = None
        self._icm_mappers = None

        self._ucms = None
        self._ucms_mappers = None

        self.finn_no_data_loader = DaskFinnNoSlateRawData()

        self._interactions_impressions: pd.DataFrame = pd.DataFrame()
        """
        Columns
        -------
            time_step: int
            user_id int
            item_id: Union[int, pd.NA]
            interacted: bool
            item_in_impressions: bool,
            item_position: Union[int, pd.NA]
            impressions: list[int], may contain repeated items
            num_impressions: int, len(impressions)
            impression_origin: IMPRESSION_ORIGIN
       """

        self._dataset: Optional[FINNNoSlateDataset] = None

    @property
    def dataset(self) -> FINNNoSlateDataset:
        if self._dataset is None:
            self._dataset = self.load_data(
                save_folder_path=self.ORIGINAL_SPLIT_FOLDER,
            )
        return self._dataset

    def _get_dataset_name_root(self) -> str:
        return self.DATASET_SUBFOLDER

    def _load_from_original_file(self) -> FINNNoSlateDataset:
        interactions, impressions = self.finn_no_data_loader.load()
        print(interactions, interactions.dtypes, interactions.shape, interactions.memory_usage().sum())
        print(impressions, impressions.dtypes, impressions.shape, impressions.memory_usage().sum())
        quit(255)
        self.finn_no_raw_data.statistics()
        self._create_urm_all_binary()

        return FINNNoSlateDataset(
            urms=self._urms,
            # impressions=self._impressions,
            user_id_to_index_mapper=self._user_id_to_index_mapper,
            item_id_to_index_mapper=self._item_id_to_index_mapper,
            is_implicit=self.IS_IMPLICIT,

        )

    def _create_urm_all_binary(self) -> None:
        """
        This method creates a CSR matrix that contains all interactions found in the dataset as a binary (0: no
        interaction, 1: 1 or more interactions) matrix. Duplicated interactions are removed and only one is kept.

        Filters applied to the dataset:
         - no clicks (item_id=1) are discarded
         - no info (item_id=0) are discarded
        """

        logger.info(
            f"Building URMs:"
        )

        ddf = self.finn_no_raw_data.load()

        interactions = ddf[
            ddf["item_id"] > 1
        ][
            ["user_id", "item_id"]
        ]

        builder_urm_all = IncrementalSparseMatrix_FilterIDs(
            preinitialized_col_mapper=None,
            on_new_col="add",
            preinitialized_row_mapper=None,
            on_new_row="add"
        )

        builder_urm_all.add_data_lists(
            row_list_to_add=interactions['user_id'].values,
            col_list_to_add=interactions['item_id'].values,
            data_list_to_add=np.ones_like(interactions['item_id'].values),
        )

        urm_all = builder_urm_all.get_SparseMatrix()
        urm_all.data = np.ones_like(urm_all.data)

        self._urms[self._NAME_URM_ALL] = urm_all.copy()

        logger.info(
            f"URM ALL size: {urm_all.data.nbytes + urm_all.indptr.nbytes + urm_all.indices.nbytes}"
        )

        # Mappers are: {original_id => new_id}
        self._user_id_to_index_mapper = builder_urm_all.get_row_token_to_id_mapper()
        self._item_id_to_index_mapper = builder_urm_all.get_column_token_to_id_mapper()


def create_mapper(
    values: pd.Series,
    mapper_name: str,
) -> pd.DataFrame:
    original_column_name = f"original_{mapper_name}_indices"
    mapped_column_name = f"mapped_{mapper_name}_indices"

    return pd.DataFrame(
        data={
            original_column_name: values.unique(),
        },
    ).sort_values(
        by=[original_column_name],
        ascending=True,
        inplace=False,
        ignore_index=True,  # Sorting unique values in ascending order.
    ).reset_index(
        drop=False,
        inplace=False,
    ).rename(
        columns={
            "index": mapped_column_name
        },
        inplace=False,
    )


if __name__ == "__main__":
    dask_interface = configure_dask_cluster()

    # FINNNoSlateRawData().load_data()
    # quit(0)

    # DaskFinnNoSlateRawData().to_dask()
    # PandasFinnNoSlateRawData().to_pandas()
    # quit(0)

    finn_no_statistics = StatisticsFinnNoSlate()
    finn_no_statistics.statistics_interactions()
    # finn_no_statistics.statistics_impressions_metadata()
    quit(0)

    data_reader = FINNNoSlateReader()
    # dataset = data_reader.dataset
    # statistics = data_reader.statistics

    # Create a training-validation-test split, for example by leave-1-out
    # This splitter requires the DataReader object and the number of elements to holdout
    data_splitter = DataSplitter_leave_k_out(
        dataReader_object=data_reader,
        k_out_value=1,
        use_validation_set=True,
        leave_random_out=True,
    )

    # The load_data function will split the data and save it in the desired folder.
    # Once the split is saved, further calls to the load_data will load the split data ensuring
    # you always use the same split
    data_splitter.load_data(
        save_folder_path="./result_experiments/FINN-NO-SLATE/data-leave-1-random-out/"
        # save_folder_path="./result_experiments/FINN-NO-SLATE/data-leave-1-random/"
    )

    # We can access the three URMs.
    urm_train, urm_validation, urm_test = data_splitter.get_holdout_split()
