""" ContentWiseImpressionsReader.py
This module reads the small or the large version of the Microsoft News ExperimentCase (ContentWiseImpressions).

Notes
-----
TODO: fernando-debugger. 
Columns of the dataset
    Impression ID
        ContentWiseImpressions ExperimentCase identifier for the impression. THEY DO NOT COME IN ORDER, i.e., a higher impression ID
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
import functools
import json
import os
from enum import Enum
from typing import Any, cast, Union, Callable, NamedTuple

import attrs
import dask.dataframe as dd
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numba import jit
from recsys_framework_extensions.data.dataset import BaseDataset
from recsys_framework_extensions.data.features import extract_frequency_user_item, extract_last_seen_user_item, \
    extract_position_user_item, extract_timestamp_user_item
from recsys_framework_extensions.data.mixins import ParquetDataMixin, DaskParquetDataMixin, DatasetStatisticsMixin, \
    SparseDataMixin, DatasetConfigBackupMixin
from recsys_framework_extensions.data.reader import DataReader
from recsys_framework_extensions.data.sparse import create_sparse_matrix_from_dataframe
from recsys_framework_extensions.data.splitter import (
    remove_duplicates_in_interactions,
    remove_users_without_min_number_of_interactions,
    split_sequential_train_test_by_column_threshold,
    split_sequential_train_test_by_num_records_on_test, E_KEEP
)
from recsys_framework_extensions.decorators import typed_cache, timeit
from recsys_framework_extensions.evaluation import EvaluationStrategy
from recsys_framework_extensions.hashing import compute_sha256_hash_from_object_repr
from recsys_framework_extensions.logging import get_logger
from tqdm import tqdm

tqdm.pandas()


logger = get_logger(
    logger_name=__file__,
)


class ContentWiseImpressionsSplits(NamedTuple):
    df_train: pd.DataFrame
    df_validation: pd.DataFrame
    df_train_validation: pd.DataFrame
    df_test: pd.DataFrame


@jit(nopython=True, parallel=False)
def compute_interaction_position_in_impression(
    impressions: np.ndarray,
    item_id: int,
) -> Union[float, int]:
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


@attrs.define(kw_only=True, frozen=True, slots=False)
class ContentWiseImpressionsConfig:
    data_folder = os.path.join(
        ".", "data", "ContentWiseImpressions",
    )

    sha256_hash: str = ""

    num_interaction_records = 10_457_810
    num_impression_records = 307_453

    pandas_dtypes = {
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
    variant: ContentWiseImpressionsVariant = attrs.field(
        default=ContentWiseImpressionsVariant.SERIES,
        validator=[
            attrs.validators.in_(ContentWiseImpressionsVariant),  # type: ignore
        ]
    )
    interactions_item_column: str = attrs.field(
        init=False,
    )

    def __attrs_post_init__(self):
        # We need to use object.__setattr__ because the benchmark_config is an immutable class, this is the attrs way to
        # circumvent assignment in immutable classes.
        if ContentWiseImpressionsVariant.ITEMS == self.variant:
            object.__setattr__(self, "interactions_item_column", "item_id")
        elif ContentWiseImpressionsVariant.SERIES == self.variant:
            object.__setattr__(self, "interactions_item_column", "series_id")
        else:
            raise ValueError(
                f"Received an invalid enum for variant={self.variant}. "
                f"Valid values are {list(ContentWiseImpressionsVariant)}"
            )

        object.__setattr__(self, "sha256_hash", compute_sha256_hash_from_object_repr(obj=self))


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
    ):
        self.config = config
        self.raw_data_loader = ContentWiseImpressionsRawData(
            config=config,
        )

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
            dtype=self.config.pandas_dtypes,
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

        return df_data


class PandasContentWiseImpressionsProcessData(ParquetDataMixin, DatasetConfigBackupMixin):
    def __init__(
        self,
        config: ContentWiseImpressionsConfig,
    ):
        self.config = config

        self.pandas_raw_data = PandasContentWiseImpressionsRawData(
            config=config,
        )

        self._dataset_folder = os.path.join(
            self.config.data_folder, "data-processing", self.config.sha256_hash, ""
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
        self.train_validation_filename = "train_validation.parquet"
        self.test_filename = "test.parquet"

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

        self.save_config(
            config=self.config,
            folder_path=self._dataset_folder,
        )

    @property  # type: ignore
    @typed_cache
    def filtered(self) -> pd.DataFrame:
        return self.load_parquet(
            file_path=self.file_filter_data_path,
            to_pandas_func=self._filtered_to_pandas,
        ).astype(
            dtype=self.config.pandas_dtypes,
        )

    @property  # type: ignore
    @typed_cache
    def timestamp_splits(
        self
    ) -> ContentWiseImpressionsSplits:
        file_paths = [
            os.path.join(self.file_timestamp_folder, self.train_filename),
            os.path.join(self.file_timestamp_folder, self.validation_filename),
            os.path.join(self.file_timestamp_folder, self.train_validation_filename),
            os.path.join(self.file_timestamp_folder, self.test_filename),
        ]

        df_train, df_validation, df_train_validation, df_test = self.load_parquets(
            file_paths=file_paths,
            to_pandas_func=self._timestamp_splits_to_pandas,
        )

        # df_train = df_train.astype(dtype=self.config.pandas_dtypes)
        # df_validation = df_validation.astype(dtype=self.config.pandas_dtypes)
        # df_test = df_test.astype(dtype=self.config.pandas_dtypes)

        df_train = df_train.astype(dtype={"user_id": np.int32, "item_id": np.int32})
        df_validation = df_validation.astype(dtype={"user_id": np.int32, "item_id": np.int32})
        df_train_validation = df_train_validation.astype(dtype={"user_id": np.int32, "item_id": np.int32})
        df_test = df_test.astype(dtype={"user_id": np.int32, "item_id": np.int32})

        assert df_train.shape[0] > 0
        assert df_validation.shape[0] > 0
        assert df_train_validation.shape[0] > 0
        assert df_test.shape[0] > 0

        return ContentWiseImpressionsSplits(
            df_train=df_train,
            df_validation=df_validation,
            df_train_validation=df_train_validation,
            df_test=df_test,
        )

    @property  # type: ignore
    @typed_cache
    def leave_last_k_out_splits(
        self
    ) -> ContentWiseImpressionsSplits:
        file_paths = [
            os.path.join(self.file_leave_last_k_out_folder, self.train_filename),
            os.path.join(self.file_leave_last_k_out_folder, self.validation_filename),
            os.path.join(self.file_leave_last_k_out_folder, self.train_validation_filename),
            os.path.join(self.file_leave_last_k_out_folder, self.test_filename),
        ]

        df_train, df_validation, df_train_validation, df_test = self.load_parquets(
            file_paths=file_paths,
            to_pandas_func=self._leave_last_k_out_splits_to_pandas,
        )

        df_train = df_train.astype(dtype={"user_id": np.int32, "item_id": np.int32})
        df_validation = df_validation.astype(dtype={"user_id": np.int32, "item_id": np.int32})
        df_train_validation = df_train_validation.astype(dtype={"user_id": np.int32, "item_id": np.int32})
        df_test = df_test.astype(dtype={"user_id": np.int32, "item_id": np.int32})

        assert df_train.shape[0] > 0
        assert df_validation.shape[0] > 0
        assert df_train_validation.shape[0] > 0
        assert df_test.shape[0] > 0

        return ContentWiseImpressionsSplits(
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

        df_data = df_data.sort_values(
            by=["timestamp"],
            ascending=True,
            axis="index",
            inplace=False,
            ignore_index=False,
        )

        df_data, _ = remove_duplicates_in_interactions(
            df=df_data,
            columns_to_compare=["user_id", self.config.interactions_item_column],
            keep=self.config.keep_duplicates,
        )

        df_data, _ = remove_users_without_min_number_of_interactions(
            df=df_data,
            users_column="user_id",
            min_number_of_interactions=self.config.min_number_of_interactions,
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

        df_data_train_validation, df_data_test = split_sequential_train_test_by_column_threshold(
            df=df_data_filtered,
            column="timestamp",
            threshold=test_threshold
        )

        df_data_train, df_data_validation = split_sequential_train_test_by_column_threshold(
            df=df_data_train_validation,
            column="timestamp",
            threshold=validation_threshold
        )

        return [df_data_train, df_data_validation, df_data_train_validation, df_data_test]

    def _leave_last_k_out_splits_to_pandas(self) -> list[pd.DataFrame]:
        df_data_filtered = self.filtered

        df_data_train_validation, df_data_test = split_sequential_train_test_by_num_records_on_test(
            df=df_data_filtered,
            group_by_column="user_id",
            num_records_in_test=1,
        )

        df_data_train, df_data_validation = split_sequential_train_test_by_num_records_on_test(
            df=df_data_train_validation,
            group_by_column="user_id",
            num_records_in_test=1,
        )

        return [df_data_train, df_data_validation, df_data_train_validation, df_data_test]


class PandasContentWiseImpressionsImpressionsFeaturesData(ParquetDataMixin, DatasetConfigBackupMixin):
    def __init__(
        self,
        config: ContentWiseImpressionsConfig,
    ):
        self.config = config
        self.loader_processed_data = PandasContentWiseImpressionsProcessData(
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
    ) -> ContentWiseImpressionsSplits:
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

        return ContentWiseImpressionsSplits(
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
    ) -> ContentWiseImpressionsSplits:
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


class SparseContentWiseImpressionData(SparseDataMixin, ParquetDataMixin, DatasetConfigBackupMixin):
    def __init__(
        self,
        config: ContentWiseImpressionsConfig,
    ):
        super().__init__()

        self.config = config

        self.data_loader_processed = PandasContentWiseImpressionsProcessData(
            config=config,
        )
        self.data_loader_impression_features = PandasContentWiseImpressionsImpressionsFeaturesData(
            config=config,
        )

        self.users_column = "user_id"
        self.items_column = self.config.interactions_item_column
        self.impressions_column = "impressions"
        self.impressions_id_column = "impression_id"

        self._folder_data = os.path.join(
            self.config.data_folder, "data-sparse", self.config.variant.value, self.config.sha256_hash, "",
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
            [self.items_column, self.impressions_id_column, self.impressions_column]
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


class ContentWiseImpressionsReader(DatasetConfigBackupMixin, DataReader):
    def __init__(
        self,
        config: ContentWiseImpressionsConfig,
    ):
        super().__init__()

        self.config = config
        self.data_loader_processed = PandasContentWiseImpressionsProcessData(
            config=config,
        )
        self.data_loader_impression_features = PandasContentWiseImpressionsImpressionsFeaturesData(
            config=config,
        )
        self.data_loader_sparse_data = SparseContentWiseImpressionData(
            config=config,
        )

        self.DATA_FOLDER = os.path.join(
            self.config.data_folder, "data_reader", self.config.variant.value, self.config.sha256_hash, "",
        )

        self.ORIGINAL_SPLIT_FOLDER = self.DATA_FOLDER
        self._DATA_READER_NAME = "ContentWiseImpressionsReader"
        self.DATASET_SUBFOLDER = "ContentWiseImpressionsReader"
        self.IS_IMPLICIT = self.config.binarize_interactions

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
        # IMPORTANT: calculate first the impressions, so we have all mappers created.
        dataframes = self.data_loader_processed.dataframes

        mapper_user_original_id_to_index = self.data_loader_sparse_data.mapper_user_id_to_index
        mapper_item_original_id_to_index = self.data_loader_sparse_data.mapper_item_id_to_index

        interactions = self.data_loader_sparse_data.interactions
        impressions = self.data_loader_sparse_data.impressions
        impressions_features_sparse_matrices = self.data_loader_sparse_data.impressions_features
        impressions_features_dataframes = self.data_loader_impression_features.impressions_features

        return BaseDataset(
            dataset_name="ContentWiseImpressions",
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


# class ContentWiseImpressionsStatistics(DatasetStatisticsMixin):
#     def __init__(
#         self,
#         reader: ContentWiseImpressionsReader,
#         config: ContentWiseImpressionsConfig,
#     ):
#         self.config = config
#         self.reader = reader
#         self.processed_data_reader = reader.processed_data_loader
#         self.raw_data_reader = reader.processed_data_loader.pandas_raw_data
#
#         self.statistics_folder = os.path.join(
#             self.config.data_folder, "statistics", self.config.variant.value, self.config.sha256_hash, "",
#         )
#         self.statistics_file_name = "statistics.zip"
#
#         self.users_column = self.reader.users_column
#         self.items_column = self.reader.items_column
#
#         self.columns_for_unique = [
#             "user_id",
#             "item_id",
#             "series_id",
#             "episode_number",
#             "series_length",
#             "item_type",
#             "impression_id",
#             "interaction_type",
#             "vision_factor",
#             "explicit_rating",
#             "num_impressions",
#             "position_interactions",
#             "num_interacted_items",
#         ]
#         self.columns_for_profile_length = [
#             "user_id",
#             "item_id",
#             "series_id",
#             "episode_number",
#             "series_length",
#             "item_type",
#             "impression_id",
#             "interaction_type",
#             "vision_factor",
#             "explicit_rating",
#             "num_impressions",
#             "position_interactions",
#             "num_interacted_items",
#         ]
#         self.columns_for_gini = [
#             "user_id",
#             "item_id",
#             "series_id",
#             "num_impressions",
#             "position_interactions",
#             "num_interacted_items",
#         ]
#         self.columns_to_group_by = [
#             "user_id",
#             "item_id",
#             "series_id",
#             "num_impressions",
#         ]
#
#         self.statistics: dict[str, Any] = dict()
#
#     @timeit
#     def compare_dataframes_statistics(self) -> None:
#         pass
#
#     @timeit
#     def compare_splits_statistics(self) -> None:
#         dataset = self.reader.dataset
#
#         urm_all = dataset.get_URM_all()
#         uim_all = dataset.get_uim_all()
#
#         for urm_name, urm_split in dataset.get_loaded_URM_items():
#             self.compare_two_sparse_matrices(
#                 csr_matrix=urm_all,
#                 csr_matrix_name="URM_all",
#                 other_csr_matrix=urm_split,
#                 other_csr_matrix_name=urm_name,
#             )
#
#         for uim_name, uim_split in dataset.get_loaded_UIM_items():
#             self.compare_two_sparse_matrices(
#                 csr_matrix=uim_all,
#                 csr_matrix_name="UIM_all",
#                 other_csr_matrix=uim_split,
#                 other_csr_matrix_name=uim_name,
#             )
#
#     @timeit
#     def raw_data_statistics(self) -> None:
#         df_data = self.raw_data_reader.data
#         dataset_name = "raw_data"
#
#         self.compute_statistics(
#             df=df_data,
#             dataset_name=dataset_name,
#             columns_for_unique=self.columns_for_unique,
#             columns_for_profile_length=self.columns_for_profile_length,
#             columns_for_gini=self.columns_for_gini,
#             columns_to_group_by=self.columns_to_group_by
#         )
#
#         self.compute_statistics_df_on_csr_matrix(
#             df=df_data,
#             dataset_name=dataset_name,
#             user_column="user_id",
#             item_column=self.reader.items_column
#         )
#
#     @timeit
#     def filtered_data_statistics(self) -> None:
#         df_data = self.processed_data_reader.filtered
#         dataset_name = "data_filtered"
#
#         self.compute_statistics(
#             df=df_data,
#             dataset_name=dataset_name,
#             columns_for_unique=self.columns_for_unique,
#             columns_for_profile_length=self.columns_for_profile_length,
#             columns_for_gini=self.columns_for_gini,
#             columns_to_group_by=self.columns_to_group_by
#         )
#
#         self.compute_statistics_df_on_csr_matrix(
#             df=df_data,
#             dataset_name=dataset_name,
#             user_column=self.users_column,
#             item_column=self.items_column,
#         )
#
#     @timeit
#     def splits_df_statistics(
#         self,
#         evaluation_strategy: EvaluationStrategy
#     ) -> None:
#         if evaluation_strategy.LEAVE_LAST_K_OUT:
#             df_train, df_validation, df_test = self.processed_data_reader.leave_last_k_out_splits
#             dataset_name_prefix = "leave_last_k_out"
#         else:
#             df_train, df_validation, df_test = self.processed_data_reader.timestamp_splits
#             dataset_name_prefix = "timestamp"
#
#         for df_data, df_name in zip(
#             [df_train, df_validation, df_test],
#             ["train", "validation", "test"]
#         ):
#             self.compute_statistics(
#                 df=df_data,
#                 dataset_name=f"{dataset_name_prefix}_{df_name}",
#                 columns_for_unique=self.columns_for_unique,
#                 columns_for_profile_length=self.columns_for_profile_length,
#                 columns_for_gini=self.columns_for_gini,
#                 columns_to_group_by=self.columns_to_group_by
#             )
#
#             self.compute_statistics_df_on_csr_matrix(
#                 df=df_data,
#                 dataset_name=f"{dataset_name_prefix}_{df_name}",
#                 user_column=self.users_column,
#                 item_column=self.items_column
#             )
#
#     @timeit
#     def splits_statistics(self) -> None:
#         dataset = self.reader.dataset
#
#         for split_name, split_csr_matrix in dataset.get_loaded_URM_items():
#             self.compute_statistics_csr_matrix(matrix=split_csr_matrix, dataset_name=split_name)
#
#         for split_name, split_csr_matrix in dataset.get_loaded_UIM_items():
#             self.compute_statistics_csr_matrix(matrix=split_csr_matrix, dataset_name=split_name)


if __name__ == "__main__":
    config = ContentWiseImpressionsConfig()

    dataset = ContentWiseImpressionsReader(config=config).dataset

    # print(dataset.get_loaded_UIM_names())
    # print(dataset.get_loaded_URM_names())
    #
    # statistics = ContentWiseImpressionsStatistics(
    #     reader=data_reader,
    #     config=config,
    # )
    #
    # # statistics.splits_statistics()
    # # statistics.raw_data_statistics()
    # # statistics.filtered_data_statistics()
    # # statistics.splits_df_statistics(
    # #     evaluation_strategy=EvaluationStrategy.LEAVE_LAST_K_OUT,
    # # )
    # # statistics.splits_df_statistics(
    # #     evaluation_strategy=EvaluationStrategy.TIMESTAMP,
    # # )
    # statistics.compare_splits_statistics()
    # statistics.save_statistics()
    #
    # import json
    # from recsys_framework_extensions.data.io import ExtendedJSONEncoderDecoder
    # json_str = json.dumps(
    #     statistics.statistics,
    #     sort_keys=True,
    #     indent=4,
    #     default=ExtendedJSONEncoderDecoder.to_json
    # )
    # logger.info(
    #     f"Results:\n "
    #     f"{json_str}"
    # )
    #
    # statistics2 = ContentWiseImpressionsStatistics(
    #     # dask_interface=dask_interface,
    #     reader=data_reader,
    #     config=config,
    # )
    # statistics2.load_statistics()
    #
    # st1 = statistics.statistics
    # st2 = statistics2.statistics
    #
    # assert len(st1) == len(st2)
    # assert set(st1.keys()) == set(st2.keys())
