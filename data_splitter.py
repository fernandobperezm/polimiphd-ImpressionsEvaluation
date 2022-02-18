from typing import Any, cast, Literal

import numpy as np
import pandas as pd
from recsys_framework.Utils.conf_logging import get_logger

logger = get_logger(
    logger_name=__file__,
)

T_KEEP = Literal["first", "last", False]


def remove_duplicates_in_interactions(
    df: pd.DataFrame,
    columns_to_compare: list[str],
    keep: T_KEEP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Removes duplicates in a dataframe using the `drop_duplicates` method of Pandas.

    See Also
    --------
    pandas.DataFrame.drop_duplicates : drops duplicates in Pandas DataFrame.

    Returns
    -------
    A tuple. The first position is the dataframe without the duplicated records. The second position is the dataframe of
    the removed records.
    """
    assert (
        keep is None
        or not keep
        or keep in ["first", "last"]
    )

    df_without_duplicates = df.drop_duplicates(
        subset=[columns_to_compare],
        keep=keep,
        inplace=False,
        ignore_index=False,
    )

    df_removed_records = df.drop(
        df_without_duplicates.index,
        inplace=False,
    )

    return df_without_duplicates, df_removed_records


def remove_users_without_min_number_of_interactions(
    df: pd.DataFrame,
    users_column: str,
    min_number_of_interactions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Removes from a dataframe those users without a certain number of interactions.

    Returns
    -------
    A tuple. The first element is the dataframe with the records of selected users removed. The second element is a
    dataframe of the removed records.
    """
    grouped_df = df.groupby(
        by=users_column,
        as_index=False,
    )[users_column].size()

    users_to_keep = grouped_df[
        grouped_df["size"] >= min_number_of_interactions
    ][users_column]

    df_users_to_keep = df[df[users_column].isin(users_to_keep)].copy()
    df_removed_users = df.drop(df_users_to_keep.index)

    num_removed_users = df_removed_users[users_column].nunique()
    num_removed_records = df_removed_users.shape[0]

    logger.warning(
        f"Found {num_removed_users} users without the desired minimum number of "
        f"interactions ({min_number_of_interactions=}). Removed {num_removed_records} records."
    )

    return df_users_to_keep, df_removed_users


def split_sequential_train_test_by_column_threshold(
    df: pd.DataFrame,
    column: str,
    threshold: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partitions the dataset into train and test after a certain threshold.

    This method is specially useful when partitioning a dataset on timestamps is required. In particular, this method
    assigns to the *train set* those records that are less or equal than `threshold`. Therefore, the *test set*
    contains those values that are greater than `threshold`.

    Notes
    -----
    This method preserves the original indices.
    """
    df_filter = cast(
        pd.Series,
        df[column] <= threshold
    )

    df_train = df[df_filter].copy()
    df_test = df[~df_filter].copy()

    return df_train, df_test


def split_sequential_train_test_by_num_records_on_test(
    df: pd.DataFrame,
    group_by_column: str,
    num_records_in_test: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partitions the dataset sequentially by a group key maintaining a certain number of records in the test set.

    Notes
    -----
    This method creates *first* the test set and then the train set, due to the pandas API. Therefore, if any group has
    less than (`num_records_in_test` + 1) interactions, these record will be sent to the test set instead of the train
    set. Moreover, if this happens, the method will raise a ValueError instance.

    This method preserves the original indices.

    Raises
    ------
    ValueError
        If any group has 0 or 1 records.

    """
    assert num_records_in_test > 0

    # There are two ways to generate a leave-last-out strategy.
    # The first uses df.groupby(...).nth[-1].
    # The second uses df.reset_index(drop=False).groupby(...).last().set_index("index").
    # Both approaches take the last record of each group with their respective indices.
    # The training set is the resulting dataframe (named df_test)
    # The training set is then df.drop(test_set.index) (named df_train)
    grouped_df = df.groupby(
        by=group_by_column,
        as_index=False,
    )

    # This variable tells the minimum size of each group so
    min_num_records_by_group = num_records_in_test + 1
    grouped_size_df = grouped_df[group_by_column].size()
    non_valid_groups = grouped_size_df["size"] < min_num_records_by_group
    if any(non_valid_groups):
        logger.error(
            f"Cannot partition the dataset given that the following groups do not have at least 2 interaction records:"
            f"\n{grouped_size_df[non_valid_groups]}"
        )
        raise ValueError(
            f"Cannot partition the dataset given that the following groups do not have at least 2 interaction records:"
            f"\n{grouped_size_df[non_valid_groups]}"
        )

    df_test = grouped_df.nth(
        np.arange(
            start=-1,
            step=-1,
            stop=-(num_records_in_test + 1)
        )
    ).copy()

    df_train = df.drop(
        df_test.index,
        inplace=False
    ).copy()

    return df_train, df_test


if __name__ == "__main__":
    test_df = pd.DataFrame(
        data=dict(
            a=[0, 1, 2, 3, 4, 5],
            b=[3, 4, 5, 6, 7, 8],
            group_id=[99, 100, 101, 101, 101, 100],
        ),
        index=[9, 10, 11, 12, 13, 14],
    )

    expected_df_train = pd.DataFrame(
        data=dict(
            a=[0, 1, 2, 3],
            b=[3, 4, 5, 6],
            group_id=[99, 100, 101, 101],
        ),
        index=[9, 10, 11, 12],
    )
    expected_df_validation = pd.DataFrame(
        data=dict(
            a=[4],
            b=[7],
            group_id=[101],
        ),
        index=[13],
    )
    expected_df_test = pd.DataFrame(
        data=dict(
            a=[5],
            b=[8],
            group_id=[100],
        ),
        index=[14],
    )

    df_keep, df_removed = remove_users_without_min_number_of_interactions(
        df=test_df,
        users_column="group_id",
        min_number_of_interactions=3,
    )

    df_train, df_test = split_sequential_train_test_by_num_records_on_test(
        df=df_keep,
        group_by_column="group_id",
        num_records_in_test=1,
    )

    df_train, df_validation = split_sequential_train_test_by_num_records_on_test(
        df=df_train,
        group_by_column="group_id",
        num_records_in_test=1,
    )

    df_train, df_test = split_sequential_train_test_by_column_threshold(
        df=test_df,
        column="b",
        threshold=7
    )

    df_train, df_validation = split_sequential_train_test_by_column_threshold(
        df=df_train,
        column="b",
        threshold=6
    )

    assert expected_df_train.equals(
        other=df_train,
    )
    assert expected_df_validation.equals(
        other=df_validation,
    )
    assert expected_df_test.equals(
        other=df_test,
    )
