import logging
from typing import cast

import numpy as np
import pandas as pd
import sparse
from recsys_framework_extensions.data.splitter import (
    remove_records_by_threshold,
    apply_custom_function,
)

import impressions_evaluation.readers.FINNNoSlates as finn_reader
from tqdm import tqdm


logger = logging.getLogger(__name__)

tqdm.pandas()


def _set_unique_items(
    *,
    df: pd.DataFrame,
) -> set[int]:
    df_items = (
        df["item_id"]
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

    unique_items = set(df_items).union(df_items_impressions)

    unique_items.discard(finn_reader.ITEM_ID_FILL_VALUE)
    unique_items.discard(finn_reader.ITEM_ID_NON_INTERACTED)

    return unique_items


def _set_unique_users(
    *,
    df: pd.DataFrame,
) -> set[int]:
    df_users = df["user_id"].dropna(
        how="any",
        inplace=False,
        axis="index",
    )

    return set(df_users)


def _create_mapper_id_to_indices(
    *,
    set_ids: set,
) -> dict[int, int]:
    return {data_id: assigned_idx for assigned_idx, data_id in enumerate(set_ids)}


def convert_dataframe_to_sparse(
    *,
    df: pd.DataFrame,
    users_column: str,
    items_column: str,
    shape: tuple,
    mapper_user_id_to_idx: dict[int, int],
    mapper_item_id_to_idx: dict[int, int],
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

    rows = df[users_column].to_numpy(dtype=np.int32)
    cols = df[items_column].to_numpy(dtype=np.int32)

    mask_array = cols >= finn_reader.ITEM_ID_DELETED_AT_DATA_COLLECTION

    rows = rows[mask_array]
    cols = cols[mask_array]

    rows = np.array(
        [mapper_user_id_to_idx[user_id] for user_id in rows], dtype=np.int32
    )
    cols = np.array(
        [mapper_item_id_to_idx[item_id] for item_id in cols], dtype=np.int32
    )
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

    uim = uim_dok.to_coo()
    print(uim, uim.data.sum(), uim.nnz)

    return uim


def finn_no_slates_statistics_full_dataset() -> dict:
    config = finn_reader.FinnNoSlatesConfig()

    pandas_raw_data = finn_reader.PandasFinnNoSlateRawData(
        config=config,
    )

    df_raw_data = pandas_raw_data.data

    # We need to remove those interaction points with 0. As these are records with an "undefined" impression origin.
    # These records all look like this: (user_id, item_id=0, impressions=[0, 0, ... 0]) so they do not bring any
    # information and were mostly processing errors.
    df_raw_data, _ = remove_records_by_threshold(
        df=df_raw_data,
        column="item_id",
        threshold=[finn_reader.ITEM_ID_FILL_VALUE],
        how="not_isin",
    )
    # We need to remove from the impressions the 'fill items' (AKA item_id = 0) and 'non-clicked items'
    # (AKA item_id = 1)
    df_raw_data, _ = apply_custom_function(
        df=df_raw_data,
        column="impressions",
        axis="columns",
        func=finn_reader.remove_non_clicks_on_impressions,
        func_name=finn_reader.remove_non_clicks_on_impressions.__name__,
        min_item_id=finn_reader.ITEM_ID_DELETED_AT_DATA_COLLECTION,
    )

    logger.debug("Computing set unique users")
    unique_users = _set_unique_users(
        df=df_raw_data,
    )

    logger.debug("Computing set unique items")
    unique_items = _set_unique_items(
        df=df_raw_data,
    )

    logger.debug("Computing mappers to users and items")
    mapper_user_id_to_index = _create_mapper_id_to_indices(set_ids=unique_users)
    mapper_item_id_to_index = _create_mapper_id_to_indices(set_ids=unique_items)

    # We need to remove the non-click item from the matrices.
    num_users, num_items = len(unique_users), len(unique_items)
    common_sparse_shape = (num_users, num_items)
    logger.debug(f"{num_users=} - {num_items=} - {common_sparse_shape=}")

    logger.debug("Computing URM")
    urm = convert_dataframe_to_sparse(
        df=df_raw_data,
        users_column="user_id",
        items_column="item_id",
        shape=common_sparse_shape,
        mapper_user_id_to_idx=mapper_user_id_to_index,
        mapper_item_id_to_idx=mapper_item_id_to_index,
    )

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
        "dataset": "FINN.no Slates",
        "num_users": num_users,
        "num_items": num_items,
        "num_interactions": urm.data.sum(),
        "num_unique_interactions": urm.nnz,
        "num_impressions": uim.data.sum(),
        "num_unique_impressions": uim.nnz,
    }

    return statistics
