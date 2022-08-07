""" ContentWiseImpressionsReader.py
This module reads, processes, splits, creates impressiosn features, and saves into disk the ContentWiseImpressions
dataset.

"""
from typing import cast

import numpy as np
import pandas as pd
import sparse
import impressions_evaluation.readers.ContentWiseImpressions.reader as cw_impressions_reader

from tqdm import tqdm
from recsys_framework_extensions.logging import get_logger


tqdm.pandas()


logger = get_logger(
    logger_name=__file__,
)


def _set_unique_items(
    *,
    df_interactions: pd.DataFrame,
    df_impressions: pd.DataFrame,
    df_impressions_non_direct_link: pd.DataFrame,
) -> set[int]:
    df_series_interactions = df_interactions["series_id"].dropna(
        inplace=False,
        how="any",
        axis="index",
    )

    df_series_interactions_impressions = df_interactions["impressions"].explode(
        ignore_index=True,
    ).dropna(
        inplace=False,
        how="any",
        axis="index",
    )

    df_series_impressions = df_impressions["recommended_series_list"].explode(
        ignore_index=True,
    ).dropna(
        inplace=False,
        how="any",
        axis="index",
    )

    df_series_impressions_non_direct_link = df_impressions_non_direct_link["recommended_series_list"].explode(
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
    df_interactions: pd.DataFrame,
    df_impressions_non_direct_link: pd.DataFrame,
) -> set[int]:
    df_interactions = df_interactions["user_id"].dropna(
        inplace=False,
        how="any",
        axis="index",
    )

    df_impressions_non_direct_link = df_impressions_non_direct_link["user_id"].explode(
        ignore_index=True,
    ).dropna(
        inplace=False,
        how="any",
        axis="index",
    )

    unique_users = set(df_interactions).union(df_impressions_non_direct_link)

    return unique_users


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


def content_wise_impressions_statistics_full_dataset() -> dict:
    config = cw_impressions_reader.ContentWiseImpressionsConfig()

    raw_data = cw_impressions_reader.ContentWiseImpressionsRawData(
        config=config,
    )

    pandas_raw_data = cw_impressions_reader.PandasContentWiseImpressionsRawData(
        config=config,
    )

    df_raw_data = pandas_raw_data.data
    df_impressions = raw_data.impressions.compute().reset_index(drop=False)
    df_impressions_non_direct_link = raw_data.impressions_non_direct_link.compute().reset_index(drop=False)

    logger.debug("Computing set unique users")
    unique_users = _set_unique_users(
        df_interactions=df_raw_data,
        df_impressions_non_direct_link=df_impressions_non_direct_link,
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
