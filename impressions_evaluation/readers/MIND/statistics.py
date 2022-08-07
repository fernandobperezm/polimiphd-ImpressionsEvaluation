""" ContentWiseImpressionsReader.py
This module reads, processes, splits, creates impressiosn features, and saves into disk the ContentWiseImpressions
dataset.

"""
from typing import cast

import numpy as np
import pandas as pd
import sparse
import impressions_evaluation.readers.MIND.reader as mind_reader
from tqdm import tqdm


import logging
logger = logging.getLogger(__name__)
logger.debug(f"new-{__name__=} - {__file__=}")

# logger_old = getLogger(__name__)
# logger_old.debug(f"old-{__name__=} - {__file__=}")

tqdm.pandas()


def _set_unique_items(
    *,
    df: pd.DataFrame,
    df_previous_interactions: pd.DataFrame,
) -> set[int]:
    df_items_previous = df_previous_interactions["item_ids"].explode(
        ignore_index=True,
    ).dropna(
        inplace=False,
        how="any",
        axis="index",
    )

    df_items = df["item_ids"].explode(
        ignore_index=True,
    ).dropna(
        inplace=False,
        how="any",
        axis="index",
    )

    unique_items = set(
        df_series_interactions
    ).union(
        df_series_interactions_impressions
    ).union(
        df_series_impressions
    ).union(
        df_series_impressions_non_direct_link
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


def convert_dataframe_to_sparse(
    *,
    df: pd.DataFrame,
    users_column: str,
    items_column: str,
    shape: tuple,
) -> sparse.COO:
    df = df[[users_column, items_column]]

    if df[items_column].dtype == "object":
        df = cast(pd.DataFrame, df).explode(
            column=items_column,
            ignore_index=True,
        ).dropna(
            how="any",
            axis="index",
            inplace=False,
        )

    rows = df[users_column].to_numpy(dtype=np.int32)
    cols = df[items_column].to_numpy(dtype=np.int32)
    data = np.ones_like(rows, dtype=np.int32)

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


def mind_small_statistics_full_dataset() -> dict:
    config = mind_reader.MINDSmallConfig()

    pandas_raw_data = mind_reader.PandasMINDRawData(
        config=config,
    )

    df_raw_data = pandas_raw_data.data
    df_previous_interactions = pandas_raw_data.previous_interactions

    # df = mind_reader.add_previous_interactions_to_dataframe(
    #     df=df_raw_data,
    #     df_previous_interactions=df_previous_interactions,
    # )
    #
    # df = mind_reader.convert_user_item_impressions_dataframe(
    #     df=df,
    #     column_item="item_ids",
    #     column_new_item="item_id",
    #     column_dtype=pd.StringDtype(),
    # )

    logger.debug("Computing set unique users")
    unique_users = _set_unique_users(
        df=df_raw_data,
        df_previous_interactions=df_previous_interactions,
    )

    logger.debug("Computing set unique items")
    unique_items = _set_unique_items(
        df_interactions=df_raw_data,
        df_impressions=df_impressions,
        df_impressions_non_direct_link=df_impressions_non_direct_link,
    )

    common_sparse_shape = (max(unique_users) + 1, max(unique_items) + 1)
    num_users, num_items = len(unique_users), len(unique_items)
    logger.debug(f"{num_users=} - {num_items=} - {common_sparse_shape=}")

    logger.debug("Computing URM")
    urm = convert_dataframe_to_sparse(
        df=df_raw_data,
        users_column="user_id",
        items_column="series_id",
        shape=common_sparse_shape,
    )

    logger.debug("Computing UIM from Interactions")
    uim_interactions = convert_dataframe_to_sparse(
        df=df_raw_data,
        users_column="user_id",
        items_column="impressions",
        shape=common_sparse_shape,
    )

    logger.debug("Computing UIM from Non-Direct-Link")
    uim_non_direct_link = convert_dataframe_to_sparse(
        df=df_impressions_non_direct_link,
        users_column="user_id",
        items_column="recommended_series_list",
        shape=common_sparse_shape,
    )

    logger.debug("Computing UIM without Interactions")
    uim = remove_interactions_from_uim(
        urm=urm,
        uim=uim_interactions + uim_non_direct_link,
    )

    logger.debug("Creating statistics dictionary")
    statistics = {
        "dataset": "ContentWise Impressions",
        "num_users": num_users,
        "num_items": num_items,
        "num_interactions": urm.data.sum(),
        "num_unique_interactions": urm.nnz,
        "num_impressions": uim.data.sum(),
        "num_unique_impressions": uim.nnz,
    }

    return statistics


def content_wise_impressions_statistics_non_duplicates_dataset():
    ...
