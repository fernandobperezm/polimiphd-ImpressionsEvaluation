""" ContentWiseImpressionsReader.py
This module reads the small or the large version of the Microsoft News Dataset (ContentWiseImpressions).

Notes
-----
TODO: fernando-debugger. 
Columns of the dataset
    Impression ID
        ContentWiseImpressions Dataset identifier for the impression. THEY DO NOT COME IN ORDER, i.e., a higher impression ID
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
from enum import Enum
from typing import Any, cast, Union

import attr
import dask.dataframe as dd
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numba import jit
from recsys_framework.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs
from recsys_framework.Recommenders.DataIO import ExtendedJSONDecoder
from recsys_framework.Utils.conf_logging import get_logger
from recsys_framework.Utils.decorators import timeit
from tqdm import tqdm

from data_splitter import remove_duplicates_in_interactions, remove_users_without_min_number_of_interactions, \
    split_sequential_train_test_by_column_threshold, T_KEEP, split_sequential_train_test_by_num_records_on_test
from mixins import BaseDataset, BaseDataReader, ParquetDataMixin, DaskParquetDataMixin, EvaluationStrategy, \
    DatasetStatisticsMixin
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


@jit(nopython=True, parallel=False)
def compute_interaction_position_in_impression(
    impressions: np.ndarray,
    item_id: int,
) -> Union[float, int, np.ndarray]:
    if np.isscalar(impressions) and np.isnan(impressions):
        return np.NaN

    # This returns a tuple of arrays. First one is row_indices and second position is column indices.
    # given that impressions are 1D arrays, we only need the first position of the tuple.
    row_indices: np.ndarray = np.asarray(impressions == item_id).nonzero()[0]
    if row_indices.size == 0:
        return np.NaN
    else:
        return row_indices.item()


compute_interaction_position_in_impression(
    impressions=np.nan,
    item_id=5,
)
compute_interaction_position_in_impression(
    impressions=np.array([1, 2, 3, 4, 5], dtype=np.int64),
    item_id=4,
)
compute_interaction_position_in_impression(
    impressions=np.array([1, 2, 3, 4, 5], dtype=np.int64),
    item_id=6,
)


class ContentWiseImpressionsVariant(Enum):
    ITEMS = "ITEMS"
    SERIES = "SERIES"


@attr.frozen
class ContentWiseImpressionsConfig:
    data_folder = os.path.join(
        ".", "data", "ContentWiseImpressions",
    )
    variant = ContentWiseImpressionsVariant.SERIES
    num_interaction_records = 10_457_810
    num_impression_records = 307_453

    min_number_of_interactions = 3
    binarize_impressions = True
    binarize_interactions = True
    keep_duplicates: T_KEEP = "first"

    pandas_dtype = {
        "user_id": "category",
        "item_id": "category",
        "series_id": "category",
        "impression_id": "category",
        "episode_number": "category",
        "interaction_type": "category",
        "item_type": "category",
        "explicit_rating": "category",
        "vision_factor": pd.Float32Dtype(),
        "row_position": pd.Int32Dtype(),
        "num_impressions": pd.Int32Dtype(),
        "num_interacted_items": pd.Int32Dtype(),
        "position_interactions": pd.Int32Dtype(),
    }


class ContentWiseImpressionsRawData(DaskParquetDataMixin):
    """Class that reads the 'raw' ContentWiseImpressionsContentWiseImpressions data from disk.

    We refer to raw data as the original representation of the data,
    without cleaning or much processing. The dataset originally has XYZ-fernando-debugger data points.
    """

    def __init__(
        self,
        config: ContentWiseImpressionsConfig,
    ):
        self._config = config

        self._original_dataset_root_folder = os.path.join(
            self._config.data_folder, "original", "CW10M",
        )

        self._original_dataset_interactions_folder = os.path.join(
            self._original_dataset_root_folder, "interactions",
        )

        self._original_dataset_impressions_direct_link_folder = os.path.join(
            self._original_dataset_root_folder, "impressions-direct-link",
        )

        self._original_dataset_impressions_non_direct_link_folder = os.path.join(
            self._original_dataset_root_folder, "impressions-non-direct-link",
        )

        self._original_dataset_metadata_file = os.path.join(
            self._original_dataset_root_folder, "metadata.json",
        )

    @property  # type: ignore
    @typed_cache
    def metadata(self) -> dict[str, Any]:
        with open(self._original_dataset_metadata_file, "r") as metadata_file:
            return json.load(
                metadata_file,
            )

    @property  # type: ignore
    @typed_cache
    def interactions(self) -> dd.DataFrame:
        return self.load_parquet(
            folder_path=self._original_dataset_interactions_folder,
            to_dask_func=None,  # type: ignore
        )

    @property  # type: ignore
    @typed_cache
    def impressions(self) -> dd.DataFrame:
        return self.load_parquet(
            folder_path=self._original_dataset_impressions_direct_link_folder,
            to_dask_func=None,  # type: ignore
        )

    @property  # type: ignore
    @typed_cache
    def impressions_non_direct_link(self) -> pd.DataFrame:
        return self.load_parquet(
            folder_path=self._original_dataset_impressions_non_direct_link_folder,
            to_dask_func=None,  # type: ignore
        )


class PandasContentWiseImpressionsRawData(ParquetDataMixin):
    """A class that reads the ContentWiseImpressions data using Pandas Dataframes."""

    def __init__(
        self,
        config: ContentWiseImpressionsConfig,
        raw_data_loader: ContentWiseImpressionsRawData,
    ):
        self.config = config
        self.raw_data_loader = raw_data_loader

        self._dataset_folder = os.path.join(
            self.config.data_folder, "pandas", "original",
        )
        self.file_data = os.path.join(
            self._dataset_folder, "data.parquet"
        )

        os.makedirs(
            name=self._dataset_folder,
            exist_ok=True,
        )

    @property  # type: ignore
    @typed_cache
    def data(self) -> pd.DataFrame:
        return self.load_parquet(
            to_pandas_func=self._data_to_pandas,
            file_path=self.file_data,
        ).astype(
            dtype=self.config.pandas_dtype,
        )

    @timeit
    def _data_to_pandas(self) -> pd.DataFrame:
        df_data = self.raw_data_loader.interactions
        df_impressions = self.raw_data_loader.impressions

        df_data = cast(
            pd.DataFrame,
            df_data.compute()
        )
        df_impressions = cast(
            pd.DataFrame,
            df_impressions.compute()
        )

        assert df_data.shape[0] == self.config.num_interaction_records
        assert df_impressions.shape[0] == self.config.num_impression_records

        df_data = df_data.reset_index(
            drop=False,
        )
        df_data["timestamp"] = pd.to_datetime(
            df_data["utc_ts_milliseconds"],
            utc=True,
            errors="raise",
            unit="ms",
        )

        df_data["recommendation_id"] = df_data["recommendation_id"].mask(
            cond=df_data["recommendation_id"] == -1,
            other=pd.NA,
        )

        df_data["explicit_rating"] = df_data["explicit_rating"].mask(
            cond=df_data["explicit_rating"] == -1,
            other=pd.NA,
        )

        df_data["vision_factor"] = df_data["vision_factor"].mask(
            cond=df_data["vision_factor"] == -1,
            other=pd.NA,
        )

        df_data = df_data.merge(
            right=df_impressions,
            how="left",
            left_on="recommendation_id",
            left_index=False,
            right_on=None,
            right_index=True,
            sort=False,
            suffixes=("", ""),
        )

        df_data = df_data.rename(
            columns={
                "recommendation_list_length": "num_impressions",
                "recommended_series_list": "impressions",
                "recommendation_id": "impression_id",
            }
        )

        # For this dataset, recommendations are series, not items, therefore, to extract the position of the clicked
        # element in the impressions we must use the `series_id` instead of the `item_id`.
        # Particularly for this dataset, we know that only one item was interacted with each impression.
        # Also, we have the
        df_data["position_interactions"] = df_data.progress_apply(
            lambda df_row: compute_interaction_position_in_impression(
                impressions=df_row["impressions"],
                item_id=df_row["series_id"],
            ),
            axis="columns",
        )

        # The data is laid out by the triplet (user, series, impressions) therefore we always know that the user
        # interacted with one item in the impression given a triplet. We start by assigning the column as 1 and then
        # changing the value of all NA impressions to na.
        df_data["num_interacted_items"] = 1
        df_data["num_interacted_items"] = df_data["num_interacted_items"].mask(
            cond=df_data["impressions"].isna(),
            other=pd.NA,
        )

        df_data = df_data.astype(
            {
                "user_id": "category",
                "item_id": "category",
                "series_id": "category",
                "impression_id": "category",
                "episode_number": "category",
                "interaction_type": "category",
                "item_type": "category",
                "explicit_rating": "category",
                "vision_factor": pd.Float32Dtype(),
                "row_position": pd.Int32Dtype(),
                "num_impressions": pd.Int32Dtype(),
                "num_interacted_items": pd.Int32Dtype(),
                "position_interactions": pd.Int32Dtype(),
            }
        )

        return df_data


class PandasContentWiseImpressionsProcessData(ParquetDataMixin):
    def __init__(
        self,
        config: ContentWiseImpressionsConfig,
        pandas_raw_data: PandasContentWiseImpressionsRawData,
    ):
        self.config = config
        self.pandas_raw_data = pandas_raw_data

        self._dataset_folder = os.path.join(
            self.config.data_folder, "data-processing",
        )

        self.file_leave_last_k_out_folder = os.path.join(
            self._dataset_folder, "leave-last-k-out", ""
        )

        self.file_timestamp_folder = os.path.join(
            self._dataset_folder, "timestamp", ""
        )

        self.file_filter_data_path = os.path.join(
            self._dataset_folder, "filter_data.parquet"
        )

        self.train_filename = "train.parquet"
        self.validation_filename = "validation.parquet"
        self.test_filename = "test.parquet"

        self._keep_duplicates = self.config.keep_duplicates
        self._min_number_of_interactions = self.config.min_number_of_interactions

        self.interactions_item_column = (
            "item_id"
            if self.config.variant.ITEMS
            else "series_id"
        )

        os.makedirs(
            name=self._dataset_folder,
            exist_ok=True,
        )
        os.makedirs(
            name=self.file_leave_last_k_out_folder,
            exist_ok=True,
        )
        os.makedirs(
            name=self.file_timestamp_folder,
            exist_ok=True,
        )

    @property  # type: ignore
    @typed_cache
    def filtered(self) -> pd.DataFrame:
        return self.load_parquet(
            file_path=self.file_filter_data_path,
            to_pandas_func=self._filtered_to_pandas,
        ).astype(
            dtype=self.config.pandas_dtype,
        )

    @property  # type: ignore
    @typed_cache
    def timestamp_splits(
        self
    ) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame,
    ]:
        file_paths = [
            os.path.join(self.file_timestamp_folder, self.train_filename),
            os.path.join(self.file_timestamp_folder, self.validation_filename),
            os.path.join(self.file_timestamp_folder, self.test_filename),
        ]

        df_train, df_validation, df_test = self.load_parquets(
            file_paths=file_paths,
            to_pandas_func=self._timestamp_splits_to_pandas,
        )

        df_train = df_train.astype(dtype=self.config.pandas_dtype)
        df_validation = df_validation.astype(dtype=self.config.pandas_dtype)
        df_test = df_test.astype(dtype=self.config.pandas_dtype)

        return df_train, df_validation, df_test

    @property  # type: ignore
    @typed_cache
    def leave_last_k_out_splits(
        self
    ) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame,
    ]:
        file_paths = [
            os.path.join(self.file_leave_last_k_out_folder, self.train_filename),
            os.path.join(self.file_leave_last_k_out_folder, self.validation_filename),
            os.path.join(self.file_leave_last_k_out_folder, self.test_filename),
        ]

        df_train, df_validation, df_test = self.load_parquets(
            file_paths=file_paths,
            to_pandas_func=self._timestamp_splits_to_pandas,
        )

        df_train = df_train.astype(dtype=self.config.pandas_dtype)
        df_validation = df_validation.astype(dtype=self.config.pandas_dtype)
        df_test = df_test.astype(dtype=self.config.pandas_dtype)

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

        # We place the index as part of the dataframe momentarily. All these functions need unique values in the index.
        # Our dataframes do not have unique values in the indices due that they are used as foreign keys/ids for the
        # dataframes.
        df_data = df_data.sort_values(
            by=["timestamp"],
            ascending=True,
            axis="index",
            inplace=False,
            ignore_index=False,
        )

        df_data, _ = remove_duplicates_in_interactions(
            df=df_data,
            columns_to_compare=["user_id", self.interactions_item_column],
            keep=self._keep_duplicates,
        )

        df_data, _ = remove_users_without_min_number_of_interactions(
            df=df_data,
            users_column="user_id",
            min_number_of_interactions=self._min_number_of_interactions,
        )

        return df_data

    def _timestamp_splits_to_pandas(self) -> list[pd.DataFrame]:
        df_data_filtered = self.filtered

        # The timestamp in the dataset represents the timestamp of the impression, not the interactions, therefore,
        # we must use the condensed version to compute the 80% and the 90% of timestamps in the dataset. Using
        # `df_interactions` may shift the value of the timestamp, specially if there are several popular users.
        described = df_data_filtered["timestamp"].describe(
            datetime_is_numeric=True,
            percentiles=[0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )

        validation_threshold = described["80%"]
        test_threshold = described["90%"]

        df_data_train, df_data_test = split_sequential_train_test_by_column_threshold(
            df=df_data_filtered,
            column="timestamp",
            threshold=test_threshold
        )

        df_data_train, df_data_validation = split_sequential_train_test_by_column_threshold(
            df=df_data_train,
            column="timestamp",
            threshold=validation_threshold
        )

        return [df_data_train, df_data_validation, df_data_test]

    def _leave_last_k_out_splits_to_pandas(self) -> list[pd.DataFrame]:
        df_data_filtered = self.filtered

        df_data_train, df_data_test = split_sequential_train_test_by_num_records_on_test(
            df=df_data_filtered,
            group_by_column="user_id",
            num_records_in_test=1,
        )

        df_data_train, df_data_validation = split_sequential_train_test_by_num_records_on_test(
            df=df_data_train,
            group_by_column="user_id",
            num_records_in_test=1,
        )

        return [df_data_train, df_data_validation, df_data_test]


class ContentWiseImpressionsReader(BaseDataReader):
    def __init__(
        self,
        config: ContentWiseImpressionsConfig,
        processed_data_loader: PandasContentWiseImpressionsProcessData,
    ):
        super().__init__()

        self.config = config

        self.DATA_FOLDER = os.path.join(
            self.config.data_folder, "data_reader", self.config.variant.value, "",
        )

        self.ORIGINAL_SPLIT_FOLDER = self.DATA_FOLDER

        self._DATA_READER_NAME = "ContentWiseImpressionsReader"

        self.DATASET_SUBFOLDER = "ContentWiseImpressionsReader"

        self.IS_IMPLICIT = self.config.binarize_interactions

        self._binarize_impressions = self.config.binarize_impressions
        self._binarize_interactions = self.config.binarize_interactions
        self._num_parts_split_dataset = 10

        self.processed_data_loader = processed_data_loader

        self.users_column = "user_id"
        self.items_column = self.processed_data_loader.interactions_item_column
        self.impressions_column = "impressions"
        self.impressions_id_column = "impression_id"

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

    def _get_dataset_name_root(self) -> str:
        return self.DATASET_SUBFOLDER

    def _load_from_original_file(self) -> BaseDataset:
        # IMPORTANT: calculate first the impressions, so we have all mappers created.
        self._calculate_uim_all()
        self._calculate_urm_all()

        self._calculate_uim_leave_last_k_out_splits()
        self._calculate_urm_leave_last_k_out_splits()

        self._calculate_uim_timestamp_splits()
        self._calculate_urm_timestamp_splits()

        return BaseDataset(
            dataset_name="ContentWiseImpressions",
            impressions=self._impressions,
            interactions=self._interactions,
            mapper_item_original_id_to_index=self._item_id_to_index_mapper,
            mapper_user_original_id_to_index=self._user_id_to_index_mapper,
            is_impressions_implicit=self.config.binarize_impressions,
            is_interactions_implicit=self.config.binarize_interactions,
        )

    def _calculate_urm_all(self):
        logger.info(
            f"Building URM with name {BaseDataset.NAME_URM_ALL}."
        )

        df_data_filtered = self.processed_data_loader.filtered[
            [self.users_column, self.items_column]
        ]

        builder_urm_all = IncrementalSparseMatrix_FilterIDs(
            preinitialized_col_mapper=self._item_id_to_index_mapper,
            on_new_col="add",
            preinitialized_row_mapper=self._user_id_to_index_mapper,
            on_new_row="add"
        )

        users = df_data_filtered[self.users_column].to_numpy()
        items = df_data_filtered[self.items_column].to_numpy()
        data = np.ones_like(users, dtype=np.int32)

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

        df_data_filtered = self.processed_data_loader.filtered[
            [self.users_column, self.impressions_column, self.impressions_id_column]
        ]
        df_data_filtered = df_data_filtered[
            df_data_filtered[self.impressions_id_column].notna()
        ]

        builder_impressions_all = IncrementalSparseMatrix_FilterIDs(
            preinitialized_col_mapper=self._item_id_to_index_mapper,
            on_new_col="add",
            preinitialized_row_mapper=self._user_id_to_index_mapper,
            on_new_row="add",
        )

        df_split_chunk: pd.DataFrame
        for df_split_chunk in tqdm(
            np.array_split(df_data_filtered, indices_or_sections=self._num_parts_split_dataset)
        ):
            # Explosions of empty lists in impressions are transformed into NAs, NA values must be removed before
            # being inserted into the csr_matrix.
            df_split_chunk = df_split_chunk.explode(
                column=self.impressions_column,
                ignore_index=False,
            )
            df_split_chunk = df_split_chunk[
                df_split_chunk[self.impressions_column].notna()
            ]

            users = df_split_chunk[self.users_column].to_numpy()
            impressions = df_split_chunk[self.impressions_column].to_numpy()
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
        df_train, df_validation, df_test = self.processed_data_loader.leave_last_k_out_splits

        df_train = df_train[
            [self.users_column, self.items_column]
        ]
        df_validation = df_validation[
            [self.users_column, self.items_column]
        ]
        df_test = df_test[
            [self.users_column, self.items_column]
        ]

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

            users = df_split[self.users_column].to_numpy()
            items = df_split[self.items_column].to_numpy()
            data = np.ones_like(users, dtype=np.int32)

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
        df_train, df_validation, df_test = self.processed_data_loader.leave_last_k_out_splits

        df_train = df_train[
            [self.users_column, self.impressions_column, self.impressions_id_column]
        ]
        df_train = df_train[
            df_train[self.impressions_id_column].notna()
        ]

        df_validation = df_validation[
            [self.users_column, self.impressions_column, self.impressions_id_column]
        ]
        df_validation = df_validation[
            df_validation[self.impressions_id_column].notna()
        ]

        df_test = df_test[
            [self.users_column, self.impressions_column, self.impressions_id_column]
        ]
        df_test = df_test[
            df_test[self.impressions_id_column].notna()
        ]

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
                uim_split.data = np.ones_like(uim_split.data, dtype=np.int32)

            self._impressions[name] = uim_split.copy()

    def _calculate_urm_timestamp_splits(self) -> None:
        df_train, df_validation, df_test = self.processed_data_loader.timestamp_splits

        df_train = df_train[
            [self.users_column, self.items_column]
        ]
        df_validation = df_validation[
            [self.users_column, self.items_column]
        ]
        df_test = df_test[
            [self.users_column, self.items_column]
        ]

        names = [
            BaseDataset.NAME_URM_TIMESTAMP_TRAIN,
            BaseDataset.NAME_URM_TIMESTAMP_VALIDATION,
            BaseDataset.NAME_URM_TIMESTAMP_TEST,
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

            users = df_split[self.users_column].to_numpy()
            items = df_split[self.items_column].to_numpy()
            data = np.ones_like(users, dtype=np.int32)

            builder_urm_split.add_data_lists(
                row_list_to_add=users,
                col_list_to_add=items,
                data_list_to_add=data,
            )

            urm_split = builder_urm_split.get_SparseMatrix()
            if self._binarize_interactions:
                urm_split.data = np.ones_like(urm_split.data, dtype=np.int32)

            self._interactions[name] = urm_split.copy()

    def _calculate_uim_timestamp_splits(self) -> None:
        df_train, df_validation, df_test = self.processed_data_loader.timestamp_splits

        df_train = df_train[
            [self.users_column, self.impressions_column, self.impressions_id_column]
        ]
        df_train = df_train[
            df_train[self.impressions_id_column].notna()
        ]

        df_validation = df_validation[
            [self.users_column, self.impressions_column, self.impressions_id_column]
        ]
        df_validation = df_validation[
            df_validation[self.impressions_id_column].notna()
        ]

        df_test = df_test[
            [self.users_column, self.impressions_column, self.impressions_id_column]
        ]
        df_test = df_test[
            df_test[self.impressions_id_column].notna()
        ]

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
            print(df_split)
            builder_uim_split = IncrementalSparseMatrix_FilterIDs(
                preinitialized_col_mapper=self._item_id_to_index_mapper,
                on_new_col="ignore",
                preinitialized_row_mapper=self._user_id_to_index_mapper,
                on_new_row="ignore"
            )

            # for _, df_row in tqdm(df_split.iterrows(), total=df_split.shape[0]):
            #     impressions = np.array(df_row["impressions"], dtype="object")
            #     users = np.array([df_row["user_id"]] * len(impressions), dtype="object")
            #     data = np.ones_like(impressions, dtype=np.int32)
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
                uim_split.data = np.ones_like(uim_split.data, dtype=np.int32)

            self._impressions[name] = uim_split.copy()


class ContentWiseImpressionsStatistics(DatasetStatisticsMixin):

    def __init__(
        self,
        reader: ContentWiseImpressionsReader,
        config: ContentWiseImpressionsConfig,
    ):
        self.config = config
        self.reader = reader
        self.processed_data_reader = reader.processed_data_loader
        self.raw_data_reader = reader.processed_data_loader.pandas_raw_data

        self.statistics_folder = os.path.join(
            self.config.data_folder, "statistics", self.config.variant.value, "",
        )
        self.statistics_file_name = "statistics.zip"

        self.columns_for_unique = [
                "user_id",
                "item_id",
                "series_id",
                "episode_number",
                "series_length",
                "item_type",
                "impression_id",
                "interaction_type",
                "vision_factor",
                "explicit_rating",
                # "num_impressions",
                # "position_interactions",
                # "num_interacted_items",
            ]
        self.columns_for_profile_length = [
                "user_id",
                "item_id",
                "series_id",
                "episode_number",
                "series_length",
                "item_type",
                "impression_id",
                "interaction_type",
                "vision_factor",
                "explicit_rating",
            ]
        self.columns_for_gini = [
                "user_id",
                "item_id",
                "series_id",
            ]
        self.columns_to_group_by = [
                "user_id",
                "item_id",
                "series_id",
            ]

        self.statistics: dict[str, Any] = dict()

    @timeit
    def compare_dataframes_statistics(self) -> None:
        pass

    @timeit
    def compare_splits_statistics(self) -> None:
        dataset = self.reader.dataset

        urm_all = dataset.get_URM_all()
        uim_all = dataset.get_uim_all()

        for urm_name, urm_split in dataset.get_loaded_URM_items():
            self.compare_two_sparse_matrices(
                csr_matrix=urm_all,
                csr_matrix_name="URM_all",
                other_csr_matrix=urm_split,
                other_csr_matrix_name=urm_name,
            )

        for uim_name, uim_split in dataset.get_loaded_UIM_items():
            self.compare_two_sparse_matrices(
                csr_matrix=uim_all,
                csr_matrix_name="UIM_all",
                other_csr_matrix=uim_split,
                other_csr_matrix_name=uim_name,
            )

    @timeit
    def raw_data_statistics(self) -> None:
        df_data = self.raw_data_reader.data
        dataset_name = "raw_data"

        self.compute_statistics(
            df=df_data,
            dataset_name=dataset_name,
            columns_for_unique=self.columns_for_unique,
            columns_for_profile_length=self.columns_for_profile_length,
            columns_for_gini=self.columns_for_gini,
            columns_to_group_by=self.columns_to_group_by
        )

        self.compute_statistics_df_on_csr_matrix(
            df=df_data,
            dataset_name=dataset_name,
            user_column="user_id",
            item_column=self.reader.interactions_item_column
        )

    @timeit
    def filtered_data_statistics(self) -> None:
        df_data = self.reader.data_filtered
        dataset_name = "data_filtered"

        self.compute_statistics(df=df_data, dataset_name=dataset_name, columns_for_unique=self.columns_for_unique,
                                columns_for_profile_length=self.columns_for_profile_length,
                                columns_for_gini=self.columns_for_gini, columns_to_group_by=self.columns_to_group_by)
        self.compute_statistics_df_on_csr_matrix(df=df_data, dataset_name=dataset_name, user_column="user_id",
                                                 item_column=self.reader.interactions_item_column)

    @timeit
    def splits_df_statistics(
        self,
        evaluation_strategy: EvaluationStrategy
    ) -> None:
        if evaluation_strategy.LEAVE_LAST_K_OUT:
            df_train, df_validation, df_test = self.reader.data_leave_last_k_out_split
            dataset_name_prefix = "leave_last_k_out"
        else:
            df_train, df_validation, df_test = self.reader.data_timestamp_split
            dataset_name_prefix = "timestamp"

        for df_data, df_name in zip(
            [df_train, df_validation, df_test],
            ["train", "validation", "test"]
        ):
            self.compute_statistics(df=df_data, dataset_name=f"{dataset_name_prefix}_{df_name}",
                                    columns_for_unique=self.columns_for_unique,
                                    columns_for_profile_length=self.columns_for_profile_length,
                                    columns_for_gini=self.columns_for_gini,
                                    columns_to_group_by=self.columns_to_group_by)
            self.compute_statistics_df_on_csr_matrix(df=df_data, dataset_name=f"{dataset_name_prefix}_{df_name}",
                                                     user_column="user_id",
                                                     item_column=self.reader.interactions_item_column)

    @timeit
    def splits_statistics(self) -> None:
        dataset = self.reader.dataset

        for split_name, split_csr_matrix in dataset.get_loaded_URM_items():
            self.compute_statistics_csr_matrix(matrix=split_csr_matrix, dataset_name=split_name)

        for split_name, split_csr_matrix in dataset.get_loaded_UIM_items():
            self.compute_statistics_csr_matrix(matrix=split_csr_matrix, dataset_name=split_name)


if __name__ == "__main__":
    config = ContentWiseImpressionsConfig()

    raw_data = ContentWiseImpressionsRawData(
        config=config,
    )

    pandas_data = PandasContentWiseImpressionsRawData(
        config=config,
        raw_data_loader=raw_data,
    )

    print(pandas_data.data.columns)
    print(pandas_data.data)

    processed_data = PandasContentWiseImpressionsProcessData(
        config=config,
        pandas_raw_data=pandas_data,
    )

    data_reader = ContentWiseImpressionsReader(
        config=config,
        processed_data_loader=processed_data,
    )
    dataset = data_reader.dataset

    print(dataset.get_loaded_UIM_names())
    print(dataset.get_loaded_URM_names())

    statistics = ContentWiseImpressionsStatistics(
        # dask_interface=dask_interface,
        reader=data_reader,
        config=config,
    )

    # statistics.splits_statistics()
    # statistics.raw_data_statistics()
    # statistics.filtered_data_statistics()
    # statistics.splits_df_statistics(
    #     evaluation_strategy=EvaluationStrategy.LEAVE_LAST_K_OUT,
    # )
    # statistics.splits_df_statistics(
    #     evaluation_strategy=EvaluationStrategy.TIMESTAMP,
    # )
    statistics.compare_splits_statistics()
    statistics.save_statistics()

    import json
    logger.info(
        f"Results:\n "
        f"{json.dumps(statistics.statistics, sort_keys=True, indent=4, cls=ExtendedJSONDecoder)}"
    )

    statistics2 = ContentWiseImpressionsStatistics(
        # dask_interface=dask_interface,
        reader=data_reader,
        config=config,
    )
    statistics2.load_statistics()

    import pdb
    pdb.set_trace()

    quit(255)
