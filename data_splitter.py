from typing import Any, cast, Literal, Optional, Union, Callable

import numpy as np
import pandas as pd
from recsys_framework.Utils.conf_logging import get_logger

logger = get_logger(
    logger_name=__file__,
)

T_KEEP = Literal["first", "last", False]
T_AXIS = Literal["index", "columns"]


def _compute_statistics(
    func_name: str,
    message: str,
    df_orig: pd.DataFrame,
    df_keep: pd.DataFrame,
    df_removed: pd.DataFrame,
    **func_kwargs
) -> None:
    num_total_users = df_orig["user_id"].nunique()
    num_total_items = df_orig["item_id"].nunique()
    num_total_records = df_orig.shape[0]

    num_keep_users = df_keep["user_id"].nunique()
    num_keep_items = df_keep["item_id"].nunique()
    num_keep_records = df_keep.shape[0]

    percentage_keep_users = (num_keep_users / num_total_users) * 100
    percentage_keep_items = (num_keep_items / num_total_items) * 100
    percentage_keep_records = (num_keep_records / num_total_records) * 100

    num_users_with_repeated_interactions = df_removed["user_id"].nunique()
    num_items_with_repeated_interactions = df_removed["item_id"].nunique()
    num_records_with_repeated_interactions = df_removed.shape[0]

    percentage_users_with_repeated_interactions = (num_users_with_repeated_interactions / num_total_users) * 100
    percentage_items_with_repeated_interactions = (num_items_with_repeated_interactions / num_total_items) * 100
    percentage_records_with_repeated_interactions = (num_records_with_repeated_interactions / num_total_records) * 100

    num_removed_users = num_total_users - num_keep_users
    num_removed_items = num_total_items - num_keep_items
    num_removed_records = num_total_records - num_keep_records

    percentage_removed_users = (num_removed_users / num_total_users) * 100
    percentage_removed_items = (num_removed_items / num_total_items) * 100
    percentage_removed_records = (num_removed_records / num_total_records) * 100

    statistics_dict = dict(
        **func_kwargs,
        original_dataset=dict(
            num_users=num_total_users,
            num_items=num_total_items,
            num_interactions=num_total_records,
        ),
        new_dataset=dict(
            num_users=num_keep_users,
            num_items=num_keep_items,
            num_interactions=num_keep_records,
            percentage_users=percentage_keep_users,
            percentage_items=percentage_keep_items,
            percentage_records=percentage_keep_records,
        ),
        other_dataset=dict(
            num_users=num_users_with_repeated_interactions,
            num_items=num_items_with_repeated_interactions,
            num_interactions=num_records_with_repeated_interactions,
            percentage_users=percentage_users_with_repeated_interactions,
            percentage_items=percentage_items_with_repeated_interactions,
            percentage_records=percentage_records_with_repeated_interactions,
        ),
        removed=dict(
            num_users=num_removed_users,
            num_items=num_removed_items,
            num_interactions=num_removed_records,
            percentage_users=percentage_removed_users,
            percentage_items=percentage_removed_items,
            percentage_records=percentage_removed_records,
        ),
    )

    logger.warning(
        f"Function {func_name}. "
        f"Found {num_removed_records}/{num_total_records} ({percentage_removed_records:.2f}%) {message}. "
        f"Function kwargs:"
        f"\n\t* {func_kwargs}"
        f"\nStatistics of the new dataset:"
        f"\n\t* # Interactions: {num_keep_records}/{num_total_records} ({percentage_keep_records:.2f}%)"
        f"\n\t* # Users: {num_keep_users}/{num_total_users} ({percentage_keep_users:.2f}%)"
        f"\n\t* # Items: {num_keep_items}/{num_total_items} ({percentage_keep_items:.2f}%)"
        f"\nStatistics of the other dataset:"
        f"\n\t* # Interactions: {num_records_with_repeated_interactions}/{num_total_records} ({percentage_records_with_repeated_interactions:.2f}%)"
        f"\n\t* # Users: {num_users_with_repeated_interactions}/{num_total_users} ({percentage_users_with_repeated_interactions:.2f}%)"
        f"\n\t* # Items: {num_items_with_repeated_interactions}/{num_total_items} ({percentage_items_with_repeated_interactions:.2f}%)"
        f"\nStatistics of the removed information:"
        f"\n\t* # Interactions: {num_removed_records}/{num_total_records} ({percentage_removed_records:.2f}%)"
        f"\n\t* # Users: {num_removed_users}/{num_total_users} ({percentage_removed_users:.2f}%)"
        f"\n\t* # Items: {num_removed_items}/{num_total_items} ({percentage_removed_items:.2f}%)"
        f"\nStatistics dict: "
        f"\n\t* {statistics_dict}"
    )


def _compute_statistics_impressions(
    func_name: str,
    message: str,
    df_orig: pd.DataFrame,
    df_keep: pd.DataFrame,
    df_removed: pd.DataFrame,
    **func_kwargs
) -> None:
    num_total_records = df_orig.shape[0]

    num_keep_records = df_keep.shape[0]
    percentage_keep_records = (num_keep_records / num_total_records) * 100

    num_records_with_repeated_interactions = df_removed.shape[0]
    percentage_records_with_repeated_interactions = (num_records_with_repeated_interactions / num_total_records) * 100

    num_removed_records = num_total_records - num_keep_records
    percentage_removed_records = (num_removed_records / num_total_records) * 100

    statistics_dict = dict(
        **func_kwargs,
        original_dataset=dict(
            num_records=num_total_records,
        ),
        new_dataset=dict(
            num_records=num_keep_records,
            percentage_records=percentage_keep_records,
        ),
        other_dataset=dict(
            num_records=num_records_with_repeated_interactions,
            percentage_records=percentage_records_with_repeated_interactions,
        ),
        removed=dict(
            num_records=num_removed_records,
            percentage_records=percentage_removed_records,
        ),
    )

    logger.warning(
        f"Function {func_name}. "
        f"Found {num_removed_records}/{num_total_records} ({percentage_removed_records:.2f}%) {message}. "
        f"Function kwargs:"
        f"\n\t* {func_kwargs}"
        f"\nStatistics of the new dataset:"
        f"\n\t* # Records: {num_keep_records}/{num_total_records} ({percentage_keep_records:.2f}%)"
        f"\nStatistics of the other dataset:"
        f"\n\t* # Records: {num_records_with_repeated_interactions}/{num_total_records} ({percentage_records_with_repeated_interactions:.2f}%)"
        f"\nStatistics of the removed information:"
        f"\n\t* # Records: {num_removed_records}/{num_total_records} ({percentage_removed_records:.2f}%)"
        f"\nStatistics dict: "
        f"\n\t* {statistics_dict}"
    )


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

    num_unique_indices = df.index.nunique()
    num_indices = df.index.shape[0]

    if num_unique_indices != num_indices:
        raise ValueError(
            f"The function {remove_duplicates_in_interactions.__name__} needs the dataframe's index to not contain "
            f"duplicates. Number of unique values in the index: {num_unique_indices}. Number of indices: {num_indices}"
        )

    df_without_duplicates = df.drop_duplicates(
        subset=columns_to_compare,
        keep=keep,
        inplace=False,
        ignore_index=False,
    )

    df_duplicates = df.drop(
        df_without_duplicates.index,
        inplace=False,
    )

    _compute_statistics(
        func_name="remove_duplicates_in_interactions",
        message="repeated user-item interactions",
        df_orig=df,
        df_keep=df_without_duplicates,
        df_removed=df_duplicates,
        keep=keep,
    )

    return df_without_duplicates, df_duplicates


def remove_users_without_min_number_of_interactions(
    df: pd.DataFrame,
    users_column: str,
    min_number_of_interactions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Removes from a dataframe those users without a certain number of interactions_exploded.

    Returns
    -------
    A tuple. The first element is the dataframe with the records of selected users removed. The second element is a
    dataframe of the removed records.
    """
    if min_number_of_interactions < 0:
        raise ValueError(
            f"min_number_of_interactions must be greater than zero. Value passed: {min_number_of_interactions}"
        )

    num_unique_indices = df.index.nunique()
    num_indices = df.index.shape[0]

    if num_unique_indices != num_indices:
        raise ValueError(
            f"The function {remove_users_without_min_number_of_interactions.__name__} needs the dataframe's index to "
            f"not contain duplicates. Number of unique values in the index: {num_unique_indices}. "
            f"Number of indices: {num_indices}"
        )

    # There are two ways to do this calculation.
    # 1) df.groupby(by=user_column, as_index=False)[user_column].size()
    # 2) df.value_counts(subset=["user_id"], normalize=False, sort=False, ascending=False, dropna=True).to_frame("size").reset_index(drop=False)
    # In both cases we will have a DataFrame with a 0...N index, a column `user_column` with the unique user ids, and
    # a column `size` containing the frequency of each user id in the dataset.
    # 1) is slightly faster than 2), a jupyter-lab %%timeit yields (5.58 ms ± 144 µs per loop (mean ± std. dev.
    # of 7 runs, 100 loops each) vs 6.55 ms ± 139 µs per loop (mean ± std. dev. of 7 runs, 100 loops each).
    grouped_df = df.groupby(
        by=users_column,
        as_index=False,
    )[users_column].size()

    users_to_keep = grouped_df[
        grouped_df["size"] >= min_number_of_interactions
    ][users_column]

    df_users_to_keep = df[
        df[users_column].isin(users_to_keep)
    ].copy()

    df_removed_users = df.drop(
        df_users_to_keep.index
    )

    _compute_statistics(
        func_name="remove_users_without_min_number_of_interactions",
        message="interactions of users without the minimum # of interactions",
        df_orig=df,
        df_keep=df_users_to_keep,
        df_removed=df_removed_users,
        min_number_of_interactions=min_number_of_interactions,
    )

    return df_users_to_keep, df_removed_users


def filter_impressions_by_interactions_index(
    df_impressions: pd.DataFrame,
    df_interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indices_in_interactions_outside_impressions = set(df_interactions.index).difference(df_impressions.index)
    if len(indices_in_interactions_outside_impressions) > 0:
        raise AssertionError(
            f"This function needs the interaction indices to be a subset of the impression indices, this is because "
            f"the interactions set has been filtered. Found the following indices in the interactions that are not in "
            f"the impressions ones: {indices_in_interactions_outside_impressions}."
        )

    # To create the UIM we need to have the (user_id, impressions) pair.
    # Also, we need to filter the impressions by the same filters of the interactions.
    # We do this by selecting from the impression set the remaining indices in the interactions after we've
    # applied all filters.
    # NOTE: be careful, this can be done with a join too
    # (e.g., df_impressions_filtered.join(other=df_interactions_filtered)), but given that the interactions have
    # several repeated indices (given that we exploded it before), then a join would create repeated impressions
    # records.
    df_impressions_keep = df_impressions[
        df_impressions.index.isin(df_interactions.index)
    ].copy()

    df_impressions_removed = df_impressions.drop(
        df_impressions_keep.index,
        inplace=False,
    )

    _compute_statistics_impressions(
        func_name="filter_impressions_by_interactions_index",
        message="impressions inside the filtered interactions.",
        df_orig=df_impressions,
        df_keep=df_impressions_keep,
        df_removed=df_impressions_removed,
    )

    return df_impressions_keep, df_impressions_removed


def remove_records_by_threshold(
    df: pd.DataFrame,
    column: str,
    threshold: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filters the dataset on the column `column` using a threshold `threshold`.

    In particular, this method keeps those records that are greater or equal than `threshold`. Therefore, discarding
    those that are strictly less than `threshold`.

    Notes
    -----
    This method preserves the original indices.
    """
    df_filter_to_keep = cast(
        pd.Series,
        df[column] >= threshold
    )

    df_keep = df[df_filter_to_keep].copy()
    df_removed = df[~df_filter_to_keep].copy()

    _compute_statistics(
        func_name="remove_records_by_threshold",
        message="Filter by threshold value",
        df_orig=df,
        df_keep=df_keep,
        df_removed=df_removed,
        column=column,
        threshold=threshold,
    )

    return df_keep, df_removed


def apply_custom_function(
    df: pd.DataFrame,
    column: str,
    func: Callable[[], Any],
    func_name: str,
    axis: T_AXIS,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    df_keep = df.copy()
    df_keep[column] = df[column].apply(
        func=func,
    )

    df_removed = df.drop(
        df_keep.index,
        inplace=False,
    )

    _compute_statistics_impressions(
        func_name=func_name,
        message=f"Applied {func_name}",
        df_orig=df,
        df_keep=df_keep,
        df_removed=df_removed,
        column=column,
        custom_func_name=func_name,
        axis=axis,
    )

    return df_keep, df_removed



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

    _compute_statistics(
        func_name="split_sequential_train_test_by_column_threshold",
        message="Split by threshold value",
        df_orig=df,
        df_keep=df_train,
        df_removed=df_test,
        column=column,
        threshold=threshold,
    )

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
    less than (`num_records_in_test` + 1) interactions_exploded, these record will be sent to the test set instead of
    the train set. Moreover, if this happens, the method will raise a ValueError instance.

    This method preserves the original indices.

    Raises
    ------
    ValueError
        If any group has 0 or 1 records.

    """
    assert num_records_in_test > 0

    num_unique_indices = df.index.nunique()
    num_indices = df.index.shape[0]

    if num_unique_indices != num_indices:
        raise ValueError(
            f"The function {split_sequential_train_test_by_num_records_on_test.__name__} needs the dataframe's index to "
            f"not contain duplicates. Number of unique values in the index: {num_unique_indices}. "
            f"Number of indices: {num_indices}"
        )

    # There are two ways to generate a leave-last-out strategy.
    # The first uses df.groupby(...).nth[-1].
    # The second uses df.reset_index(drop=False).groupby(...).last().set_index("index").
    # Both approaches take the last record of each group with their respective indices.
    # The training set is the resulting dataframe (named df_test)
    # The training set is then df.drop(test_set.index) (named df_train)
    # NOTE: a groupby on a categorical column will have rows for categories not inside the dataframe, e.g.,
    # if some users were removed from the interactions, the groupby will show their sizes as 0 because they're part
    # of the category.
    grouped_df = df.groupby(
        by=group_by_column,
        as_index=False,
    )

    # This variable tells the minimum size of each group.
    min_num_records_by_group = num_records_in_test + 1
    grouped_size_df = grouped_df[group_by_column].size()

    # Given that the groupby counts for all user, even if they are not in the interactions, then we must filter the
    # grouped dataframe by the actual users in the interactions dataframe.
    users_in_df = grouped_size_df[
        grouped_size_df[group_by_column].isin(df[group_by_column].unique())
    ]

    non_valid_groups = users_in_df["size"] < min_num_records_by_group

    if any(non_valid_groups):
        message = (
            f"Cannot partition the dataset given that the following groups do not have at least "
            f"{min_num_records_by_group} interaction records:"
            f"\n{users_in_df[non_valid_groups]}"
        )
        logger.error(message)
        raise ValueError(message)

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

    _compute_statistics(
        func_name="split_sequential_train_test_by_num_records_on_test",
        message="Split by leave-last-k-out",
        df_orig=df,
        df_keep=df_train,
        df_removed=df_test,
        group_by_column=group_by_column,
        num_records_in_test=num_records_in_test,
    )

    return df_train, df_test


# if __name__ == "__main__":
#     test_df = pd.DataFrame(
#         data=dict(
#             a=[0, 1, 2, 3, 4, 5],
#             b=[3, 4, 5, 6, 7, 8],
#             group_id=[99, 100, 101, 101, 101, 100],
#         ),
#         index=[9, 10, 11, 12, 13, 14],
#     )
#
#     expected_df_train = pd.DataFrame(
#         data=dict(
#             a=[0, 1, 2, 3],
#             b=[3, 4, 5, 6],
#             group_id=[99, 100, 101, 101],
#         ),
#         index=[9, 10, 11, 12],
#     )
#     expected_df_validation = pd.DataFrame(
#         data=dict(
#             a=[4],
#             b=[7],
#             group_id=[101],
#         ),
#         index=[13],
#     )
#     expected_df_test = pd.DataFrame(
#         data=dict(
#             a=[5],
#             b=[8],
#             group_id=[100],
#         ),
#         index=[14],
#     )
#
#     df_keep, df_removed = remove_users_without_min_number_of_interactions(
#         df=test_df,
#         users_column="group_id",
#         min_number_of_interactions=3,
#     )
#
#     df_train, df_test = split_sequential_train_test_by_num_records_on_test(
#         df=df_keep,
#         group_by_column="group_id",
#         num_records_in_test=1,
#     )
#
#     df_train, df_validation = split_sequential_train_test_by_num_records_on_test(
#         df=df_train,
#         group_by_column="group_id",
#         num_records_in_test=1,
#     )
#
#     df_train, df_test = split_sequential_train_test_by_column_threshold(
#         df=test_df,
#         column="b",
#         threshold=7
#     )
#
#     df_train, df_validation = split_sequential_train_test_by_column_threshold(
#         df=df_train,
#         column="b",
#         threshold=6
#     )
#
#     assert expected_df_train.equals(
#         other=df_train,
#     )
#     assert expected_df_validation.equals(
#         other=df_validation,
#     )
#     assert expected_df_test.equals(
#         other=df_test,
#     )
