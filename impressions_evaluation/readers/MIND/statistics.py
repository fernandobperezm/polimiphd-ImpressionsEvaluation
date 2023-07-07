""" ContentWiseImpressionsReader.py
This module reads, processes, splits, creates impressiosn features, and saves into disk the ContentWiseImpressions
dataset.

"""
import logging
from typing import cast, Union

import numpy as np
import pandas as pd
import sparse
import impressions_evaluation.readers.MINDReader as mind_reader
from tqdm import tqdm


logger = logging.getLogger(__name__)

tqdm.pandas()


def _set_unique_items(
    *,
    df: pd.DataFrame,
    df_previous_interactions: pd.DataFrame,
) -> set[int]:
    df_items_previous_interactions = (
        df_previous_interactions["item_ids"]
        .explode(
            ignore_index=True,
        )
        .dropna(
            inplace=False,
            how="any",
            axis="index",
        )
    )

    df_items = (
        df["item_ids"]
        .explode(
            ignore_index=True,
        )
        .dropna(
            inplace=False,
            how="any",
            axis="index",
        )
    )

    df_items_impressions = (
        df["impressions"]
        .explode(
            ignore_index=True,
        )
        .dropna(
            inplace=False,
            how="any",
            axis="index",
        )
    )

    unique_items = (
        set(df_items_previous_interactions).union(df_items).union(df_items_impressions)
    )

    return unique_items


def _set_unique_users(
    *,
    df: pd.DataFrame,
    df_previous_interactions: pd.DataFrame,
) -> set[int]:
    df_users = df["user_id"].dropna(
        how="any",
        inplace=False,
        axis="index",
    )
    df_previous_interactions = df_previous_interactions["user_id"].dropna(
        how="any",
        inplace=False,
        axis="index",
    )
    return set(df_users).union(df_previous_interactions)


def _create_mapper_id_to_indices(
    set_ids: set,
) -> dict[str, int]:
    return {data_id: assigned_idx for assigned_idx, data_id in enumerate(set_ids)}


def convert_dataframe_to_sparse(
    *,
    df: pd.DataFrame,
    users_column: str,
    items_column: str,
    shape: tuple,
    mapper_user_id_to_idx: dict[str, int],
    mapper_item_id_to_idx: dict[str, int],
) -> sparse.COO:
    df = df[[users_column, items_column]]

    if df[items_column].dtype == "object":
        df = (
            cast(pd.DataFrame, df)
            .explode(
                column=items_column,
                ignore_index=True,
            )
            .dropna(
                how="any",
                axis="index",
                inplace=False,
            )
        )

    rows_str = df[users_column].to_numpy(dtype="object")
    cols_str = df[items_column].to_numpy(dtype="object")

    rows = np.array(
        [mapper_user_id_to_idx[user_id] for user_id in rows_str], dtype=np.int32
    )
    cols = np.array(
        [mapper_item_id_to_idx[item_id] for item_id in cols_str], dtype=np.int32
    )
    data = np.ones_like(rows_str, dtype=np.int32)

    urm = sparse.COO(
        (data, (rows, cols)),
        has_duplicates=True,
        shape=shape,
    )
    print(urm, urm.data.sum(), urm.nnz)

    return urm


def remove_interactions_from_uim(
    *,
    urm: sparse.COO,
    uim: sparse.COO,
) -> sparse.COO:
    uim_dok = sparse.DOK.from_coo(uim)

    for data_idx in range(urm.data.size):
        row_idx = urm.coords[0, data_idx]
        col_idx = urm.coords[1, data_idx]

        uim_dok[row_idx, col_idx] = 0

    return uim_dok.to_coo()


def _compute_mind_statistics_from_config(
    config: Union[mind_reader.MINDSmallConfig, mind_reader.MINDLargeConfig],
) -> dict:
    pandas_raw_data = mind_reader.PandasMINDRawData(
        config=config,
    )

    df_raw_data = pandas_raw_data.data
    df_previous_interactions = pandas_raw_data.previous_interactions

    logger.debug("Computing set unique users")
    unique_users = _set_unique_users(
        df=df_raw_data,
        df_previous_interactions=df_previous_interactions,
    )

    logger.debug("Computing set unique items")
    unique_items = _set_unique_items(
        df=df_raw_data,
        df_previous_interactions=df_previous_interactions,
    )

    mapper_user_id_to_index = _create_mapper_id_to_indices(set_ids=unique_users)
    mapper_item_id_to_index = _create_mapper_id_to_indices(set_ids=unique_items)

    num_users, num_items = len(unique_users), len(unique_items)
    common_sparse_shape = (num_users + 1, num_items + 1)
    logger.debug(f"{num_users=} - {num_items=} - {common_sparse_shape=}")

    logger.debug("Computing URM previous interactions")
    urm_previous_interactions = convert_dataframe_to_sparse(
        df=df_previous_interactions,
        users_column="user_id",
        items_column="item_ids",
        shape=common_sparse_shape,
        mapper_user_id_to_idx=mapper_user_id_to_index,
        mapper_item_id_to_idx=mapper_item_id_to_index,
    )

    logger.debug("Computing URM interactions")
    urm_interactions = convert_dataframe_to_sparse(
        df=df_raw_data,
        users_column="user_id",
        items_column="item_ids",
        shape=common_sparse_shape,
        mapper_user_id_to_idx=mapper_user_id_to_index,
        mapper_item_id_to_idx=mapper_item_id_to_index,
    )

    logger.debug("Computing URM")
    urm: sparse.COO = urm_previous_interactions + urm_interactions

    logger.debug("Computing UIM from Interactions")
    uim_with_interactions = convert_dataframe_to_sparse(
        df=df_raw_data,
        users_column="user_id",
        items_column="impressions",
        shape=common_sparse_shape,
        mapper_user_id_to_idx=mapper_user_id_to_index,
        mapper_item_id_to_idx=mapper_item_id_to_index,
    )

    logger.debug("Computing UIM without Interactions")
    uim = remove_interactions_from_uim(
        urm=urm,
        uim=uim_with_interactions,
    )

    logger.debug("Creating statistics dictionary")
    statistics = {
        "dataset": "MIND",
        "num_users": num_users,
        "num_items": num_items,
        "num_interactions": urm.data.sum(),
        "num_unique_interactions": urm.nnz,
        "num_impressions": uim.data.sum(),
        "num_unique_impressions": uim.nnz,
    }

    return statistics


def mind_small_statistics_full_dataset() -> dict:
    config = mind_reader.MINDSmallConfig()

    statistics = _compute_mind_statistics_from_config(config=config)
    statistics["dataset"] = "MIND Small"

    return statistics

    # pandas_raw_data = mind_reader.PandasMINDRawData(
    #     config=config,
    # )
    #
    # df_raw_data = pandas_raw_data.data
    # df_previous_interactions = pandas_raw_data.previous_interactions
    #
    # logger.debug("Computing set unique users")
    # unique_users = _set_unique_users(
    #     df=df_raw_data,
    #     df_previous_interactions=df_previous_interactions,
    # )
    #
    # logger.debug("Computing set unique items")
    # unique_items = _set_unique_items(
    #     df=df_raw_data,
    #     df_previous_interactions=df_previous_interactions,
    # )
    #
    # mapper_user_id_to_index = _create_mapper_id_to_indices(set_ids=unique_users)
    # mapper_item_id_to_index = _create_mapper_id_to_indices(set_ids=unique_items)
    #
    # num_users, num_items = len(unique_users), len(unique_items)
    # common_sparse_shape = (num_users + 1, num_items + 1)
    #
    # logger.debug(f"{num_users=} - {num_items=} - {common_sparse_shape=}")
    #
    # logger.debug("Computing URM previous interactions")
    # urm_previous_interactions = convert_dataframe_to_sparse(
    #     df=df_previous_interactions,
    #     users_column="user_id",
    #     items_column="item_ids",
    #     shape=common_sparse_shape,
    #     mapper_user_id_to_idx=mapper_user_id_to_index,
    #     mapper_item_id_to_idx=mapper_item_id_to_index,
    # )
    #
    # logger.debug("Computing URM interactions")
    # urm_interactions = convert_dataframe_to_sparse(
    #     df=df_raw_data,
    #     users_column="user_id",
    #     items_column="item_ids",
    #     shape=common_sparse_shape,
    #     mapper_user_id_to_idx=mapper_user_id_to_index,
    #     mapper_item_id_to_idx=mapper_item_id_to_index,
    # )
    #
    # logger.debug("Computing URM")
    # urm: sparse.COO = urm_previous_interactions + urm_interactions
    #
    # logger.debug("Computing UIM from Interactions")
    # uim_with_interactions = convert_dataframe_to_sparse(
    #     df=df_raw_data,
    #     users_column="user_id",
    #     items_column="impressions",
    #     shape=common_sparse_shape,
    #     mapper_user_id_to_idx=mapper_user_id_to_index,
    #     mapper_item_id_to_idx=mapper_item_id_to_index,
    # )
    #
    # logger.debug("Computing UIM without Interactions")
    # uim = remove_interactions_from_uim(
    #     urm=urm,
    #     uim=uim_with_interactions,
    # )
    #
    # logger.debug("Creating statistics dictionary")
    # statistics = {
    #     "dataset": "MIND Small",
    #     "num_users": num_users,
    #     "num_items": num_items,
    #     "num_interactions": urm.data.sum(),
    #     "num_unique_interactions": urm.nnz,
    #     "num_impressions": uim.data.sum(),
    #     "num_unique_impressions": uim.nnz,
    # }
    #
    # return statistics


def mind_large_statistics_full_dataset() -> dict:
    config = mind_reader.MINDLargeConfig(
        use_test_set=True,
        use_historical_interactions=True,
    )

    statistics = _compute_mind_statistics_from_config(config=config)
    statistics["dataset"] = "MIND"

    return statistics
