""" ContentWiseImpressionsReader.py
This module reads, processes, splits, creates impressiosn features, and saves into disk the ContentWiseImpressions
dataset.

"""
import enum
import os
from typing import cast, Sequence, Union, Literal, Optional, Any

import numpy as np
import pandas as pd
import powerlaw
import scipy.stats as sp_st
import sparse
from matplotlib import dates

import impressions_evaluation.readers.ContentWiseImpressionsReader as cw_impressions_reader

import matplotlib.axes
import matplotlib.pyplot as plt
import tikzplotlib

from recsys_framework_extensions.data.io import DataIO
from tqdm import tqdm
import logging

plt.style.use("ggplot")
tqdm.pandas()

SIZE_INCHES_WIDTH = 5.7  # equals to 14.48 cm
SIZE_INCHES_HEIGHT = 2.5  # equals to 6.35 cm

logger = logging.getLogger(__name__)


class InteractionType(enum.Enum):
    View = 0
    Detail = 1
    Rate = 2
    Purchase = 3


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

    df_series_interactions_impressions = (
        df_interactions["impressions"]
        .explode(
            ignore_index=True,
        )
        .dropna(
            inplace=False,
            how="any",
            axis="index",
        )
    )

    df_series_impressions = (
        df_impressions["recommended_series_list"]
        .explode(
            ignore_index=True,
        )
        .dropna(
            inplace=False,
            how="any",
            axis="index",
        )
    )

    df_series_impressions_non_direct_link = (
        df_impressions_non_direct_link["recommended_series_list"]
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
        set(df_series_interactions)
        .union(df_series_interactions_impressions)
        .union(df_series_impressions)
        .union(df_series_impressions_non_direct_link)
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

    df_impressions_non_direct_link = (
        df_impressions_non_direct_link["user_id"]
        .explode(
            ignore_index=True,
        )
        .dropna(
            inplace=False,
            how="any",
            axis="index",
        )
    )

    unique_users = set(df_interactions).union(df_impressions_non_direct_link)

    return unique_users


def _compute_basic_statistics(
    arr_data: np.ndarray,
    data_discrete: bool,
    name: str,
) -> dict[str, Any]:
    if arr_data.size == 0:
        logger.warning(
            "Cannot compute basic statistics on empty array, early returning an empty dictionary."
        )
        return {}

    list_quantiles = [
        0.001,
        0.01,
        *np.arange(start=0.05, stop=0.95, step=0.05, dtype=np.float16),
        0.99,
        0.999,
    ]

    results_scipy_stats_describe = sp_st.describe(
        a=arr_data,
        axis=None,
        bias=True,
        nan_policy="raise",  # This should not raise, in case it does, then we did something wrong.
    )
    results_mode_mode, results_mode_count = cast(
        tuple[float, float],
        sp_st.mode(
            a=arr_data,
            axis=None,
            nan_policy="raise",  # This should not raise, in case it does, then we did something wrong.
        ),
    )
    results_quantile: np.ndarray = np.quantile(
        a=arr_data,
        q=list_quantiles,
        axis=None,
    )
    results_std_dev: float = cast(
        float,
        np.std(
            a=arr_data,
            axis=None,
        ),
    )
    results_kurtosis: float | np.ndarray = sp_st.kurtosis(
        a=arr_data,
        axis=None,
        fisher=False,
        bias=True,
    )
    results_powerlaw = powerlaw.Fit(
        data=arr_data,
        discrete=data_discrete,
    )
    results_powerlaw_lognormal = cast(
        tuple[float, float],
        results_powerlaw.distribution_compare("power_law", "lognormal"),
    )
    results_powerlaw_exponential = cast(
        tuple[float, float],
        results_powerlaw.distribution_compare("power_law", "exponential"),
    )

    return {
        "name": name,
        "num_obs": results_scipy_stats_describe.nobs,
        "min": results_scipy_stats_describe.minmax[0],
        "max": results_scipy_stats_describe.minmax[1],
        "median": results_quantile[5],
        "mode": results_mode_mode,
        "mode_count": results_mode_count,
        "mean": results_scipy_stats_describe.mean,
        "std": results_std_dev,
        "var": results_scipy_stats_describe.variance,
        "skewness": results_scipy_stats_describe.skewness,
        "kurtosis": results_kurtosis,
        "powerlaw_xmin": results_powerlaw.xmin,
        "powerlaw_alpha": results_powerlaw.alpha,
        "powerlaw_lognormal_likelihood": results_powerlaw_lognormal[0],
        "powerlaw_lognormal_pvalue": results_powerlaw_lognormal[1],
        "powerlaw_exponential_likelihood": results_powerlaw_exponential[0],
        "powerlaw_exponential_pvalue": results_powerlaw_exponential[1],
        **{
            f"quantile_percent_{q:.2f}": r
            for q, r in zip(list_quantiles, results_quantile)
        },
    }


def convert_dataframe_to_sparse(
    *,
    df: pd.DataFrame,
    users_column: str,
    items_column: str,
    shape: tuple,
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
    df_impressions_non_direct_link = (
        raw_data.impressions_non_direct_link.compute().reset_index(drop=False)
    )

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


def compute_popularity(
    df: pd.DataFrame,
    column: Union[str, Sequence[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # df = df.sort_values(
    #     by=column,
    #     ascending=True,
    # )
    col_popularity = (
        df[column]
        .value_counts(
            normalize=False,
            sort=True,
            ascending=False,
        )
        .to_frame(name="count")  # Creates column "count"
        .reset_index(drop=False)  # Creates column with name `column`
        .reset_index(drop=False)  # Creates column `index`
    )
    col_popularity_perc = (
        df[column]
        .value_counts(
            normalize=True,
            sort=True,
            ascending=False,
        )
        .to_frame(name="count")  # Creates column "count"
        .reset_index(drop=False)  # Creates column with name `column`
        .reset_index(drop=False)  # Creates column `index`
    )

    return col_popularity, col_popularity_perc


def plot_popularity(
    *,
    df: pd.DataFrame,
    dir_results: str,
    x_data: str,
    y_data: str,
    x_label: str,
    y_label: str,
    name: str,
    x_scale: Literal["linear", "log"] = "linear",
    y_scale: Literal["linear", "log"] = "log",
    x_err: Optional[str] = None,
    y_err: Optional[str] = None,
) -> None:
    # TODO: Add hot, middle, and long tail lines as in "Multistakeholder Recommender Systems" page 663, year 2022 Figure 2.
    #  This is, two lines, one at 80% and another at 60% interactions.

    fig: plt.Figure
    ax: plt.Axes

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(
            SIZE_INCHES_WIDTH,
            SIZE_INCHES_HEIGHT,
        ),  # Must be (width, height) by the docs.
        layout="compressed",
    )

    # TODO: Replace dots by lines? it may work.
    # ax.plot(x_data, y_data, data=df)
    ax.errorbar(x=x_data, y=y_data, xerr=x_err, yerr=y_err, data=df)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)

    plot_name = "zipf_law" if x_scale == "log" and y_scale == "log" else "pop"
    plot_name += "" if x_err is None and y_err is None else "_error"

    tikzplotlib.clean_figure(fig=fig)
    tikzplotlib.save(
        os.path.join(dir_results, f"plot-{plot_name}-{name}.tikz"),  # cannot be kwarg!
        fig,  # cannot be kwarg!
        encoding="utf-8",
        textsize=9,
    )

    fig.show()

    # plt.plot(
    #     x_data,
    #     y_data,
    #     data=df,
    # )
    # plt.xlabel(f"{x_label} Rank")
    # plt.ylabel(y_label)
    # plt.xscale(x_scale)
    # plt.yscale(y_scale)
    #
    # tikzplotlib.clean_figure()
    # tikzplotlib.save(
    #     os.path.join(dir_results, f"plot-pop-{name}.tikz"),
    #     encoding="utf-8",
    #     textsize=9,
    # )
    #
    # plt.show()


# def plot_dates(
#     *,
#     dir_results: str,
#     df: pd.DataFrame,
#     x_data: str,
#     y_data: str,
#     x_date: bool,
#     y_date: bool,
#     x_label: str,
#     y_label: str,
#     name: str,
# ) -> None:
#     fig: plt.Figure
#     ax: plt.Axes
#
#     fig, ax = plt.subplots(
#         nrows=1,
#         ncols=1,
#         figsize=(
#             SIZE_INCHES_WIDTH,
#             SIZE_INCHES_HEIGHT,
#         ),  # Must be (width, height) by the docs.
#         layout="compressed",
#     )
#
#     # NOTE FOR THE FUTURE: despite this method being deprecated, the migration to the newer method is annoying and may break. One thing to remember is to NEVER set the `scale` of the axis that is of type `date`, i.e., DO NOT CALL `ax.set_xscale("linear")` when the X axis contains dates -- calling it causes the plot to print dates as integers.
#     # TODO: do not plot all dates in the x-ticks. Plot every 15 days or so.
#     # TODO: Replace dots by lines? it may work.
#     ax.plot_date(x_data, y_data, data=df, xdate=x_date, ydate=y_date)
#     ax.set_xlabel(x_label)
#     ax.set_ylabel(y_label)
#     # If one axis is of type `date` then the other must be linear, log, etc.
#     if x_date:
#         ax.set_yscale("linear")
#     if y_date:
#         ax.set_xscale("linear")
#
#     tikzplotlib.save(
#         os.path.join(dir_results, f"plot-date-{name}.tikz"),  # cannot be kwarg!
#         fig,  # cannot be kwarg!
#         encoding="utf-8",
#         textsize=9,
#     )
#
#     fig.show()


def plot_dates(
    *,
    dir_results: str,
    df: pd.DataFrame,
    x_data: str,
    y_data: str,
    x_date: bool,
    y_date: bool,
    x_label: str,
    y_label: str,
    name: str,
    x_err: Optional[str] = None,
    y_err: Optional[str] = None,
) -> None:
    fig: plt.Figure
    ax: plt.Axes

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(
            SIZE_INCHES_WIDTH,
            SIZE_INCHES_HEIGHT,
        ),  # Must be (width, height) by the docs.
        layout="compressed",
    )

    # NOTE FOR THE FUTURE: despite this method being deprecated, the migration to the newer method is annoying and may break. One thing to remember is to NEVER set the `scale` of the axis that is of type `date`, i.e., DO NOT CALL `ax.set_xscale("linear")` when the X axis contains dates -- calling it causes the plot to print dates as integers.
    if x_date:
        ax.xaxis_date(None)
    if y_date:
        ax.yaxis_date(None)

    # TODO: do not plot all dates in the x-ticks. Plot every 15 days or so.
    # TODO: Replace dots by lines? it may work.
    ax.errorbar(x=x_data, y=y_data, yerr=y_err, x_err=x_err, data=df)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # If one axis is of type `date` then the other must be linear, log, etc.
    if x_date:
        ax.set_yscale("linear")
    if y_date:
        ax.set_xscale("linear")

    plot_name = "date" if x_err is None and y_err is None else "date_errors"

    tikzplotlib.save(
        os.path.join(dir_results, f"plot-{plot_name}-{name}.tikz"),  # cannot be kwarg!
        fig,  # cannot be kwarg!
        encoding="utf-8",
        textsize=9,
    )

    fig.show()


def plot_barplot(
    *,
    df: Union[pd.DataFrame, Sequence[tuple[pd.DataFrame, str]]],
    dir_results: str,
    x_data: str,
    y_data: str,
    x_label: str,
    y_label: str,
    ticks_labels: Optional[Sequence[str]],
    name: str,
    log: bool = False,
    align: Literal["center", "edge"] = "center",
) -> None:
    fig: plt.Figure
    ax: plt.Axes

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(
            SIZE_INCHES_WIDTH,
            SIZE_INCHES_HEIGHT,
        ),  # Must be (width, height) by the docs.
        layout="compressed",
    )

    ax.bar(
        x_data,
        y_data,
        width=0.25,
        data=df,
        log=log,
        tick_label=ticks_labels,
        align=align,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plot_name = "bar"

    # if isinstance(df, pd.DataFrame):
    #     ax.bar(
    #         x_data,
    #         y_data,
    #         width=0.1,
    #         data=df,
    #         log=log,
    #         tick_label=ticks_labels,
    #         align=align,
    #     )
    #     ax.set_xlabel(x_label)
    #     ax.set_ylabel(y_label)
    #
    #     plot_name = "bar"
    #
    # else:
    #     width = 0.25
    #     multiplier = 0
    #     for df_single, label in df:
    #         offset = width * multiplier
    #         rects = ax.bar(
    #             df_single[x_data] + offset,
    #             df_single[y_data],
    #             width=width,
    #             label=label,
    #             log=log,
    #             # tick_label=ticks_labels,
    #             # align=align,
    #         )
    #         multiplier += 1
    #         # plt.bar_label(rects, padding=3)
    #
    #     plot_name = "bar_group"

    tikzplotlib.clean_figure(fig=fig)
    tikzplotlib.save(
        os.path.join(dir_results, f"plot-{plot_name}-{name}.tikz"),  # cannot be kwarg!
        fig,  # cannot be kwarg!
        encoding="utf-8",
        textsize=9,
    )

    fig.show()


def plot_histogram(
    *,
    df: pd.DataFrame,
    dir_results: str,
    x_data: str,
    x_label: str,
    y_label: str,
    name: str,
    bins: Optional[int] = None,
    hist_type: Literal["bar", "step"] = "bar",
    log: bool = False,
) -> None:
    fig: plt.Figure
    ax: plt.Axes

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(
            SIZE_INCHES_WIDTH,
            SIZE_INCHES_HEIGHT,
        ),  # Must be (width, height) by the docs.
        layout="compressed",
    )

    ax.hist(
        x_data,
        data=df,
        bins=bins,
        histtype=hist_type,
        log=log,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    tikzplotlib.clean_figure(fig=fig)
    tikzplotlib.save(
        os.path.join(dir_results, f"plot-hist-{name}.tikz"),  # cannot be kwarg
        fig,  # cannot be kwarg
        encoding="utf-8",
        textsize=9,
    )

    fig.show()


def plot_boxplot(
    *,
    df: Optional[pd.DataFrame],
    dir_results: str,
    x_data: Union[str, list[np.ndarray]],
    x_labels: Sequence[str],
    y_label: str,
    name: str,
    vert: bool = True,
) -> None:
    fig: plt.Figure
    ax: plt.Axes

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(
            SIZE_INCHES_WIDTH,
            SIZE_INCHES_HEIGHT,
        ),  # Must be (width, height) by the docs.
        layout="compressed",
    )

    ax.boxplot(
        x_data,
        data=df,
        vert=vert,
        labels=x_labels,
    )
    ax.set_ylabel(y_label)

    tikzplotlib.clean_figure(fig=fig)
    tikzplotlib.save(
        os.path.join(dir_results, f"plot-box-{name}.tikz"),  # cannot be kwargs
        fig,  # cannot be kwargs.
        encoding="utf-8",
        textsize=9,
    )

    fig.show()


def plot_violinplot(
    *,
    df: Optional[pd.DataFrame],
    dir_results: str,
    x_data: Union[str, list[np.ndarray]],
    x_labels: Sequence[str],
    y_label: str,
    name: str,
    vert: bool = True,
    show_means: bool = False,
    show_extrema: bool = True,
    show_medians: bool = True,
) -> None:
    fig: plt.Figure
    ax: plt.Axes

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(
            SIZE_INCHES_WIDTH,
            SIZE_INCHES_HEIGHT,
        ),  # Must be (width, height) by the docs.
        layout="compressed",
    )

    ax.violinplot(
        x_data,
        data=df,
        vert=vert,
        showmeans=show_means,
        showextrema=show_extrema,
        showmedians=show_medians,
    )
    ax.set_xticks(np.arange(1, len(x_labels) + 1), labels=x_labels)
    ax.set_xlim(0.25, len(x_labels) + 0.75)
    ax.set_ylabel(y_label)

    tikzplotlib.clean_figure(fig=fig)
    tikzplotlib.save(
        os.path.join(dir_results, f"plot-violin-{name}.tikz"),  # cannot be kwargs.
        fig,  # cannot be kwargs.
        encoding="utf-8",
        textsize=9,
    )

    fig.show()


def compute_popularity_interactions(
    dir_results: str,
    dict_results: dict[str, pd.DataFrame],
    df_interactions_all: pd.DataFrame,
    df_interactions_only_null_impressions: pd.DataFrame,
    df_interactions_only_non_null_impressions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if all(
        filename in dict_results
        for filename in [
            "user_pop_interactions_all",
            "user_pop_interactions_null_impressions",
            "user_pop_interactions_non_null_impressions",
            "item_pop_interactions_all",
            "item_pop_interactions_null_impressions",
            "item_pop_interactions_non_null_impressions",
            "series_pop_interactions_all",
            "series_pop_interactions_null_impressions",
            "series_pop_interactions_non_null_impressions",
        ]
    ):
        logger.warning(
            "Skipping function %(function)s because all keys in the dictionary already exist.",
            {"function": compute_popularity_interactions.__name__},
        )
        return {}

    user_pop_all, user_pop_perc_all = compute_popularity(
        df=df_interactions_all,
        column="user_id",
    )
    user_pop_null, user_pop_perc_null = compute_popularity(
        df=df_interactions_only_null_impressions,
        column="user_id",
    )
    user_pop_non_null, user_pop_perc_non_null = compute_popularity(
        df=df_interactions_only_non_null_impressions,
        column="user_id",
    )
    for df, name in zip(
        [user_pop_all, user_pop_null, user_pop_non_null],
        [
            "user_popularity_interactions_all",
            "user_popularity_interactions_null_impressions",
            "user_popularity_interactions_non_null_impressions",
        ],
    ):
        plot_popularity(
            df=df,
            dir_results=dir_results,
            x_data="index",
            y_data="count",
            x_label=r"Users",
            y_label=r"\# of interactions",
            name=name,
        )

    item_pop_all, item_pop_perc_all = compute_popularity(
        df=df_interactions_all,
        column="item_id",
    )
    item_pop_null, item_pop_perc_null = compute_popularity(
        df=df_interactions_only_null_impressions,
        column="item_id",
    )
    item_pop_non_null, item_pop_perc_non_null = compute_popularity(
        df=df_interactions_only_non_null_impressions,
        column="item_id",
    )
    for df, name in zip(
        [item_pop_all, item_pop_null, item_pop_non_null],
        [
            "item_popularity_interactions_all",
            "item_popularity_interactions_null_impressions",
            "item_popularity_interactions_non_null_impressions",
        ],
    ):
        plot_popularity(
            df=df,
            dir_results=dir_results,
            x_data="index",
            y_data="count",
            x_label=r"Items",
            y_label=r"\# of interactions",
            name=name,
        )

    series_pop_all, series_pop_perc_all = compute_popularity(
        df=df_interactions_all,
        column="series_id",
    )
    series_pop_null, series_pop_perc_null = compute_popularity(
        df=df_interactions_only_null_impressions,
        column="series_id",
    )
    series_pop_non_null, series_pop_perc_non_null = compute_popularity(
        df=df_interactions_only_non_null_impressions,
        column="series_id",
    )
    for df, name in zip(
        [series_pop_all, series_pop_null, series_pop_non_null],
        [
            "series_popularity_interactions_all",
            "series_popularity_interactions_null_impressions",
            "series_popularity_interactions_non_null_impressions",
        ],
    ):
        plot_popularity(
            df=df,
            dir_results=dir_results,
            x_data="index",
            y_data="count",
            x_label=r"Series",
            y_label=r"\# of interactions",
            name=name,
        )

    return {
        "user_pop_interactions_all": user_pop_all,
        "user_pop_interactions_null_impressions": user_pop_null,
        "user_pop_interactions_non_null_impressions": user_pop_non_null,
        "item_pop_interactions_all": item_pop_all,
        "item_pop_interactions_null_impressions": item_pop_null,
        "item_pop_interactions_non_null_impressions": item_pop_non_null,
        "series_pop_interactions_all": series_pop_all,
        "series_pop_interactions_null_impressions": series_pop_null,
        "series_pop_interactions_non_null_impressions": series_pop_non_null,
    }


def compute_popularity_impressions_contextual(
    dir_results: str,
    dict_results: dict[str, pd.DataFrame],
    df_interactions_only_non_null_impressions: pd.DataFrame,
    df_impressions_contextual: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if all(
        filename in dict_results
        for filename in [
            "series_pop_impressions_contextual_with_interactions",
            "series_pop_impressions_contextual",
        ]
    ):
        logger.warning(
            "Skipping function %(function)s because all keys in the dictionary already exist.",
            {"function": compute_popularity_impressions_contextual.__name__},
        )
        return {}

    df_interactions_with_contextual_impressions = (
        df_interactions_only_non_null_impressions[
            ["user_id", "recommendation_id"]
        ].merge(
            right=df_impressions_contextual,
            how="inner",
            left_on="recommendation_id",
            right_index=True,
        )
    )

    df_interactions_with_contextual_impressions = (
        df_interactions_with_contextual_impressions[
            ["recommended_series_list"]
        ].explode(
            column="recommended_series_list",
            ignore_index=True,
        )
    )

    df_only_impressions_contextual = df_impressions_contextual[
        ["recommended_series_list"]
    ].explode(
        column="recommended_series_list",
        ignore_index=True,
    )

    (
        series_pop_impressions_contextual_with_interactions,
        series_pop_perc_impressions_contextual_with_interactions,
    ) = compute_popularity(
        df=df_interactions_with_contextual_impressions,
        column="recommended_series_list",
    )
    (
        series_pop_impressions_contextual,
        series_pop_perc_impressions_contextual,
    ) = compute_popularity(
        df=df_only_impressions_contextual,
        column="recommended_series_list",
    )

    for df, name in zip(
        [
            series_pop_impressions_contextual_with_interactions,
            series_pop_impressions_contextual,
        ],
        [
            "series_popularity_impressions_contextual_with_interactions",
            "series_popularity_impressions_contextual",
        ],
    ):
        plot_popularity(
            df=df,
            dir_results=dir_results,
            x_data="index",
            y_data="count",
            x_label=r"Series",
            y_label=r"\# of impressions",
            name=name,
        )

    return {
        "series_pop_impressions_contextual_with_interactions": series_pop_impressions_contextual_with_interactions,
        "series_pop_impressions_contextual": series_pop_impressions_contextual,
    }


def compute_popularity_impressions_global(
    dir_results: str,
    dict_results: dict[str, pd.DataFrame],
    df_impressions_global: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if all(
        filename in dict_results
        for filename in [
            "series_pop_impressions_global",
        ]
    ):
        logger.warning(
            "Skipping function %(function)s because all keys in the dictionary already exist.",
            {"function": compute_popularity_impressions_global.__name__},
        )
        return {}

    df_impressions_global = df_impressions_global[["recommended_series_list"]].explode(
        column="recommended_series_list",
        ignore_index=True,
    )

    (
        series_pop_impressions_global,
        series_pop_perc_impressions_global,
    ) = compute_popularity(
        df=df_impressions_global,
        column="recommended_series_list",
    )

    for df, name in zip(
        [
            series_pop_impressions_global,
        ],
        [
            "series_popularity_impressions_global",
        ],
    ):
        plot_popularity(
            df=df,
            dir_results=dir_results,
            x_data="index",
            y_data="count",
            x_label=r"Series",
            y_label=r"\# of impressions",
            name=name,
        )

    return {
        "series_pop_impressions_global": series_pop_impressions_global,
    }


def compute_correlation_interactions_impressions(
    dir_results: str,
    dict_results: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    if all(
        filename in dict_results
        for filename in [
            "series_pop_corr_pearson",
            "series_pop_corr_kendall",
            "series_pop_corr_spearman",
        ]
    ):
        logger.warning(
            "Skipping function %(function)s because all keys in the dictionary already exist.",
            {"function": compute_correlation_interactions_impressions.__name__},
        )
        return {}

    assert "series_pop_interactions_all" in dict_results
    assert "series_pop_impressions_contextual" in dict_results
    assert "series_pop_impressions_global" in dict_results

    df_series_pop_interactions_all = (
        dict_results["series_pop_interactions_all"]
        .set_index("series_id")
        .rename(
            columns={
                "count": "count_interactions",
                "index": "index_interactions",
            }
        )
    )
    df_series_pop_impressions_contextual = (
        dict_results["series_pop_impressions_contextual"]
        .set_index("recommended_series_list")
        .rename(
            columns={
                "count": "count_impressions_contextual",
                "index": "index_impressions_contextual",
            }
        )
    )
    df_series_pop_impressions_global = (
        dict_results["series_pop_impressions_global"]
        .set_index("recommended_series_list")
        .rename(
            columns={
                "count": "count_impressions_global",
                "index": "index_impressions_global",
            }
        )
    )

    df_series_pop = (
        df_series_pop_interactions_all.merge(
            right=df_series_pop_impressions_contextual,
            left_index=True,
            right_index=True,
            how="outer",
            suffixes=("", ""),
        )
        .merge(
            right=df_series_pop_impressions_global,
            left_index=True,
            right_index=True,
            how="outer",
            suffixes=("", ""),
        )
        .fillna(
            {
                "count_interactions": 0,
                "count_impressions_contextual": 0,
                "count_impressions_global": 0,
            }
        )
        .astype(
            {
                "count_interactions": np.int32,
                "count_impressions_contextual": np.int32,
                "count_impressions_global": np.int32,
            }
        )
    )

    df_series_pop_corr_pearson = df_series_pop[
        [
            "count_interactions",
            "count_impressions_contextual",
            "count_impressions_global",
        ]
    ].corr(
        method="pearson",
    )
    df_series_pop_corr_kendall = df_series_pop[
        [
            "count_interactions",
            "count_impressions_contextual",
            "count_impressions_global",
        ]
    ].corr(
        method="kendall",
    )

    df_series_pop_corr_spearman = df_series_pop[
        [
            "count_interactions",
            "count_impressions_contextual",
            "count_impressions_global",
        ]
    ].corr(
        method="spearman",
    )

    df_series_pop_corr_pearson.to_csv(
        path_or_buf=os.path.join(dir_results, "table-series_pop_corr_pearson.csv"),
        sep=";",
        float_format="%.4f",
        header=True,
        index=True,
        encoding="utf-8",
        decimal=",",
    )

    df_series_pop_corr_kendall.to_csv(
        path_or_buf=os.path.join(dir_results, "table-series_pop_corr_kendall.csv"),
        sep=";",
        float_format="%.4f",
        header=True,
        index=True,
        encoding="utf-8",
        decimal=",",
    )

    df_series_pop_corr_spearman.to_csv(
        path_or_buf=os.path.join(dir_results, "table-series_pop_corr_spearman.csv"),
        sep=";",
        float_format="%.4f",
        header=True,
        index=True,
        encoding="utf-8",
        decimal=",",
    )

    return {
        "series_pop_corr_pearson": df_series_pop_corr_pearson,
        "series_pop_corr_kendall": df_series_pop_corr_kendall,
        "series_pop_corr_spearman": df_series_pop_corr_spearman,
    }


def compute_daily_hourly_number_of_interactions(
    dir_results: str,
    dict_results: dict[str, pd.DataFrame],
    df_interactions_all: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    # if all(
    #     filename in dict_results
    #     for filename in ["num_interactions_by_date", "num_interactions_by_hour"]
    # ):
    #     return {}

    df_num_interactions_by_date = (
        df_interactions_all.groupby(
            by="date",
        )["series_id"]
        .count()
        .to_frame()
        .reset_index(drop=False)
        .rename(
            columns={
                "index": "date",
                "series_id": "count",
            }
        )
        .sort_values(
            by="date",
            ascending=True,
            ignore_index=True,
            inplace=False,
        )
    )
    df_num_interactions_by_date["date"] = df_num_interactions_by_date["date"].astype(
        str
    )

    df_num_interactions_by_hour = (
        df_interactions_all.groupby(
            by="hour",
        )["series_id"]
        .count()
        .to_frame()
        .reset_index(drop=False)
        .rename(
            columns={
                "index": "hour",
                "series_id": "count",
            }
        )
        .sort_values(
            by="hour",
            ascending=True,
            ignore_index=True,
            inplace=False,
        )
    )

    for df, x_data, x_label, x_date, name in [
        (
            df_num_interactions_by_date,
            "date",
            r"Date",
            True,
            "num_interactions_by_date",
        ),
        (
            df_num_interactions_by_hour,
            "hour",
            r"Hour",
            False,
            "num_interactions_by_hours",
        ),
    ]:
        plot_dates(
            dir_results=dir_results,
            df=df,
            name=name,
            x_data=x_data,
            y_data="count",
            x_label=x_label,
            y_label=r"\# of interactions",
            x_date=x_date,
            y_date=False,
        )

    return {
        "num_interactions_by_date": df_num_interactions_by_date,
        "num_interactions_by_hour": df_num_interactions_by_hour,
    }


def compute_daily_hourly_number_of_impressions(
    dir_results: str,
    dict_results: dict[str, pd.DataFrame],
    df_interactions_only_non_null_impressions: pd.DataFrame,
    df_impressions_contextual: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    # if all(
    #     filename in dict_results
    #     for filename in ["num_impressions_by_date", "num_impressions_by_hour"]
    # ):
    #     return {}

    df_interactions_with_impressions_contextual = (
        df_interactions_only_non_null_impressions.merge(
            right=df_impressions_contextual,
            on="recommendation_id",
            how="inner",
            suffixes=("", ""),
        )
    )

    df_num_impressions_by_date = (
        df_interactions_with_impressions_contextual.groupby(
            by="date",
        )["recommendation_list_length"]
        .sum()
        .to_frame()
        .reset_index(drop=False)
        .rename(
            columns={
                "index": "date",
                "recommendation_list_length": "count",
            }
        )
        .sort_values(
            by="date",
            ascending=True,
            ignore_index=True,
            inplace=False,
        )
    )
    df_num_impressions_by_date["date"] = df_num_impressions_by_date["date"].astype(str)

    df_num_impressions_by_hour = (
        df_interactions_with_impressions_contextual.groupby(
            by="hour",
        )["recommendation_list_length"]
        .sum()
        .to_frame()
        .reset_index(drop=False)
        .rename(
            columns={
                "index": "hour",
                "recommendation_list_length": "count",
            }
        )
        .sort_values(
            by="hour",
            ascending=True,
            ignore_index=True,
            inplace=False,
        )
    )

    for df, x_data, x_label, x_date, name in [
        (
            df_num_impressions_by_date,
            "date",
            r"Date",
            True,
            "num_impressions_by_date",
        ),
        (
            df_num_impressions_by_hour,
            "hour",
            r"Hour",
            False,
            "num_impressions_by_hours",
        ),
    ]:
        plot_dates(
            dir_results=dir_results,
            df=df,
            name=name,
            x_data=x_data,
            y_data="count",
            x_label=x_label,
            y_label=r"\# of impressions",
            x_date=x_date,
            y_date=False,
        )

    return {
        "num_impressions_by_date": df_num_impressions_by_date,
        "num_impressions_by_hour": df_num_impressions_by_hour,
    }


def compute_vision_factor(
    dir_results: str,
    dict_results: dict[str, pd.DataFrame],
    df_interactions_all: pd.DataFrame,
    df_interactions_only_null_impressions: pd.DataFrame,
    df_interactions_only_non_null_impressions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if all(
        filename in dict_results
        for filename in [
            "table_vision_factor",
            "vision_factor_all",
            "vision_factor_only_null_impressions",
            "vision_factor_only_non_null_impressions",
        ]
    ):
        return {}

    df_interactions_all = df_interactions_all[
        df_interactions_all["interaction_type"] == InteractionType.View.value
    ]
    df_interactions_only_null_impressions = df_interactions_only_null_impressions[
        df_interactions_only_null_impressions["interaction_type"]
        == InteractionType.View.value
    ]
    df_interactions_only_non_null_impressions = (
        df_interactions_only_non_null_impressions[
            df_interactions_only_non_null_impressions["interaction_type"]
            == InteractionType.View.value
        ]
    )

    list_basic_statistics = []

    for df, name in [
        (
            df_interactions_all,
            "vision_factor_all",
        ),
        (
            df_interactions_only_null_impressions,
            "vision_factor_only_null_impressions",
        ),
        (
            df_interactions_only_non_null_impressions,
            "vision_factor_only_non_null_impressions",
        ),
    ]:
        logger.debug(f"Computing {name}")

        arr_vision_factor = df["vision_factor"].to_numpy()
        results_basic_statistics = _compute_basic_statistics(
            arr_data=arr_vision_factor,
            data_discrete=False,
            name=name,
        )
        list_basic_statistics.append(results_basic_statistics)

        plot_histogram(
            df=df,
            dir_results=dir_results,
            x_data="vision_factor",
            x_label="Vision factor",
            y_label="Frequency",
            name=name,
            hist_type="bar",
            log=True,
            bins=20,
        )

    plot_boxplot(
        df=None,
        dir_results=dir_results,
        x_data=[
            df_interactions_all["vision_factor"].to_numpy(),
            df_interactions_only_null_impressions["vision_factor"].to_numpy(),
            df_interactions_only_non_null_impressions["vision_factor"].to_numpy(),
        ],
        x_labels=[
            "All \ninteractions",
            "Outside \ncontextual impressions",
            "Inside \ncontextual impressions",
        ],
        y_label="Vision factor",
        name="vision_factor_datasets",
        vert=True,
    )

    plot_violinplot(
        df=None,
        dir_results=dir_results,
        x_data=[
            df_interactions_all["vision_factor"].to_numpy(),
            df_interactions_only_null_impressions["vision_factor"].to_numpy(),
            df_interactions_only_non_null_impressions["vision_factor"].to_numpy(),
        ],
        x_labels=[
            "All \ninteractions",
            "Outside \ncontextual impressions",
            "Inside \ncontextual impressions",
        ],
        y_label="Vision factor",
        name="vision_factor_datasets",
        vert=True,
        show_means=True,
        show_extrema=True,
        show_medians=True,
    )

    df_results = pd.DataFrame.from_records(data=list_basic_statistics)
    df_results.to_csv(
        path_or_buf=os.path.join(dir_results, "table_statistics_vision_factor.csv"),
        index=True,
        header=True,
        sep=";",
        encoding="utf-8",
        decimal=",",
        float_format="%.4f",
    )

    return {
        "table_vision_factor": df_results,
        "vision_factor_all": df_interactions_all["vision_factor"].to_frame(),
        "vision_factor_only_null_impressions": df_interactions_only_null_impressions[
            "vision_factor"
        ].to_frame(),
        "vision_factor_only_non_null_impressions": df_interactions_only_non_null_impressions[
            "vision_factor"
        ].to_frame(),
    }


def compute_ratings(
    dir_results: str,
    dict_results: dict[str, pd.DataFrame],
    df_interactions_all: pd.DataFrame,
    df_interactions_only_null_impressions: pd.DataFrame,
    df_interactions_only_non_null_impressions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if all(
        filename in dict_results
        for filename in [
            "table_vision_factor",
            "explicit_rating_all",
            "explicit_rating_only_null_impressions",
            "explicit_rating_only_non_null_impressions",
        ]
    ):
        return {}

    # df_interactions_all = (
    #     df_interactions_all[
    #         (df_interactions_all["interaction_type"] == InteractionType.Rate.value)
    #         & (df_interactions_all["explicit_rating"] >= 0.0)
    #     ]["explicit_rating"]
    #     .value_counts(ascending=False)
    #     .sort_index()
    #     .reset_index(drop=False)
    #     .assign(
    #         label=lambda df: df["explicit_rating"].apply(
    #             lambda val: f"{val:.1f}",
    #             convert_dtype=True,
    #         )
    #     )
    # )
    #
    # df_interactions_only_null_impressions = (
    #     df_interactions_only_null_impressions[
    #         (
    #             df_interactions_only_null_impressions["interaction_type"]
    #             == InteractionType.Rate.value
    #         )
    #         & (df_interactions_only_null_impressions["explicit_rating"] >= 0.0)
    #     ]["explicit_rating"]
    #     .value_counts(ascending=False)
    #     .sort_index()
    #     .reset_index(drop=False)
    #     .assign(
    #         label=lambda df: df["explicit_rating"].apply(
    #             lambda val: f"{val:.1f}",
    #             convert_dtype=True,
    #         )
    #     )
    # )
    #
    # df_interactions_only_non_null_impressions = (
    #     df_interactions_only_non_null_impressions[
    #         (
    #             df_interactions_only_non_null_impressions["interaction_type"]
    #             == InteractionType.Rate.value
    #         )
    #         & (df_interactions_only_non_null_impressions["explicit_rating"] >= 0.0)
    #     ]["explicit_rating"]
    #     .value_counts(ascending=False)
    #     .sort_index()
    #     .reset_index(drop=False)
    #     .assign(
    #         label=lambda df: df["explicit_rating"].apply(
    #             lambda val: f"{val:.1f}",
    #             convert_dtype=True,
    #         )
    #     )
    # )
    # df_interactions_only_non_null_impressions[
    #     "label"
    # ] = df_interactions_only_non_null_impressions["explicit_rating"].apply(
    #     lambda val: f"{val:.1f}",
    #     convert_dtype=True,
    # )

    df_interactions_all = df_interactions_all[
        df_interactions_all["interaction_type"] == InteractionType.Rate.value
    ]

    df_interactions_only_null_impressions = df_interactions_only_null_impressions[
        df_interactions_only_null_impressions["interaction_type"]
        == InteractionType.Rate.value
    ]

    df_interactions_only_non_null_impressions = (
        df_interactions_only_non_null_impressions[
            df_interactions_only_non_null_impressions["interaction_type"]
            == InteractionType.Rate.value
        ]
    )

    list_basic_statistics = []

    for df, name in [
        (
            df_interactions_all,
            "explicit_ratings_all",
        ),
        (
            df_interactions_only_null_impressions,
            "explicit_ratings_only_null_impressions",
        ),
        (
            df_interactions_only_non_null_impressions,
            "explicit_ratings_only_non_null_impressions",
        ),
    ]:
        logger.debug(f"Computing {name}")

        arr_explicit_ratings = df["explicit_rating"].to_numpy()
        results_basic_statistics = _compute_basic_statistics(
            arr_data=arr_explicit_ratings,
            data_discrete=False,
            name=name,
        )
        list_basic_statistics.append(results_basic_statistics)

        df_pop = (
            df["explicit_rating"]
            .value_counts(ascending=False)
            .sort_index()
            .reset_index(drop=False)
            .assign(
                label=lambda df_: df_["explicit_rating"].apply(
                    lambda val: f"{val:.1f}",
                    convert_dtype=True,
                )
            )
        )

        plot_barplot(
            df=df_pop,
            dir_results=dir_results,
            x_data="explicit_rating",
            y_data="count",
            x_label="Rating",
            y_label="Count",
            ticks_labels=df_pop["label"],
            name=name,
            log=True,
        )

    # plot_barplot(
    #     df=[
    #         (df_interactions_all, "All interactions"),
    #         (df_interactions_only_null_impressions, "Outside\nimpressions"),
    #         (df_interactions_only_non_null_impressions, "Inside\nimpressions"),
    #     ],
    #     dir_results=dir_results,
    #     x_data="explicit_rating",
    #     y_data="count",
    #     x_label="Rating",
    #     y_label="Count",
    #     ticks_labels=df_interactions_all["label"],
    #     name="explicit_ratings_datasets",
    #     log=True,
    # )

    df_results = pd.DataFrame.from_records(data=list_basic_statistics)
    df_results.to_csv(
        path_or_buf=os.path.join(dir_results, "table_statistics_explicit_ratings.csv"),
        index=True,
        header=True,
        sep=";",
        encoding="utf-8",
        decimal=",",
        float_format="%.4f",
    )

    return {
        "table_vision_factor": df_results,
        "explicit_rating_all": df_interactions_all,
        "explicit_rating_only_null_impressions": df_interactions_only_null_impressions,
        "explicit_rating_only_non_null_impressions": df_interactions_only_non_null_impressions,
    }


def _compute_num_int_num_imp_ctr(
    *,
    cols_group_by: Union[str, Sequence[str]],
    df_imp: pd.DataFrame,
    df_int: pd.DataFrame,
) -> pd.DataFrame:
    df_num_int = (
        df_int.groupby(
            by=cols_group_by,
            as_index=False,
        )["item_id"]
        .count()
        .rename(columns={"item_id": "num_interactions"})
    )
    df_num_imp = (
        df_imp.groupby(
            by=cols_group_by,
            as_index=False,
        )["item_id"]
        .count()
        .rename(columns={"item_id": "num_impressions"})
    )
    df_num_int_num_imp = (
        pd.merge(
            left=df_num_int,
            right=df_num_imp,
            on=cols_group_by,
            how="outer",
            left_on=None,
            right_on=None,
            left_index=False,
            right_index=False,
            suffixes=("", ""),
            sort=False,
        )
        .fillna({"num_interactions": 0, "num_impressions": 0})
        .astype({"num_interactions": np.int32, "num_impressions": np.int32})
    )
    df_num_int_num_imp["ctr"] = (
        df_num_int_num_imp["num_interactions"] / df_num_int_num_imp["num_impressions"]
    )
    return df_num_int_num_imp


def compute_table_ctr(
    dir_results: str,
    dict_results: dict[str, pd.DataFrame],
    df_interactions_all: pd.DataFrame,
    df_interactions_only_non_null_impressions: pd.DataFrame,
    df_impressions_contextual_all: pd.DataFrame,
    df_impressions_contextual_only_non_null_impressions: pd.DataFrame,
    df_impressions_global: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if all(
        filename in dict_results
        for filename in [
            "table_ctr_statistics",
        ]
    ):
        logger.warning(
            "Skipping function %(function)s because all keys in the dictionary already exist.",
            {"function": compute_table_ctr.__name__},
        )
        return {}

    df_impressions_contextual_global_all = pd.concat(
        objs=[
            df_impressions_contextual_all,
            df_impressions_global,
        ],
        axis=0,
        ignore_index=True,
        sort=False,
    )

    df_impressions_contextual_global_only_non_null_impressions = pd.concat(
        objs=[
            df_impressions_contextual_only_non_null_impressions,
            df_impressions_global,
        ],
        axis=0,
        ignore_index=True,
        sort=False,
    )

    cases_contextual_all = [
        (
            df_interactions_all,
            df_impressions_contextual_all,
            "ctr_1d_global_contextual_all",
            "global",
        ),
        (
            df_interactions_all,
            df_impressions_contextual_all,
            "ctr_1d_user_contextual_all",
            "user_id",
        ),
        (
            df_interactions_all,
            df_impressions_contextual_all,
            "ctr_1d_date_contextual_all",
            "date",
        ),
        (
            df_interactions_all,
            df_impressions_contextual_all,
            "ctr_1d_series_contextual_all",
            "series_id",
        ),
        (
            df_interactions_all,
            df_impressions_contextual_all,
            "ctr_2d_user_series_contextual_all",
            ["user_id", "series_id"],
        ),
    ]
    cases_contextual_only_non_null_impressions = [
        (
            df_interactions_only_non_null_impressions,
            df_impressions_contextual_only_non_null_impressions,
            "ctr_1d_global_contextual_only_non_null",
            "global",
        ),
        (
            df_interactions_only_non_null_impressions,
            df_impressions_contextual_only_non_null_impressions,
            "ctr_1d_user_contextual_only_non_null",
            "user_id",
        ),
        (
            df_interactions_only_non_null_impressions,
            df_impressions_contextual_only_non_null_impressions,
            "ctr_1d_date_contextual_only_non_null",
            "date",
        ),
        (
            df_interactions_only_non_null_impressions,
            df_impressions_contextual_only_non_null_impressions,
            "ctr_1d_series_contextual_only_non_null",
            "series_id",
        ),
        (
            df_interactions_only_non_null_impressions,
            df_impressions_contextual_only_non_null_impressions,
            "ctr_2d_user_series_contextual_only_non_null",
            ["user_id", "series_id"],
        ),
    ]
    cases_global_all = [
        (
            df_interactions_all,
            df_impressions_contextual_global_all,
            "ctr_1d_global_global_all",
            "global",
        ),
        (
            df_interactions_all,
            df_impressions_contextual_global_all,
            "ctr_1d_user_global_all",
            "user_id",
        ),
        (
            df_interactions_all,
            df_impressions_contextual_global_all,
            "ctr_1d_date_global_all",
            "date",
        ),
        (
            df_interactions_all,
            df_impressions_contextual_global_all,
            "ctr_1d_series_global_all",
            "series_id",
        ),
        (
            df_interactions_all,
            df_impressions_contextual_global_all,
            "ctr_2d_user_series_global_all",
            ["user_id", "series_id"],
        ),
    ]
    cases_global_only_non_null_impressions = [
        (
            df_interactions_only_non_null_impressions,
            df_impressions_contextual_global_only_non_null_impressions,
            "ctr_1d_global_global_only_non_null",
            "global",
        ),
        (
            df_interactions_only_non_null_impressions,
            df_impressions_contextual_global_only_non_null_impressions,
            "ctr_1d_user_global_only_non_null",
            "user_id",
        ),
        (
            df_interactions_only_non_null_impressions,
            df_impressions_contextual_global_only_non_null_impressions,
            "ctr_1d_date_global_only_non_null",
            "date",
        ),
        (
            df_interactions_only_non_null_impressions,
            df_impressions_contextual_global_only_non_null_impressions,
            "ctr_1d_series_global_only_non_null",
            "series_id",
        ),
        (
            df_interactions_only_non_null_impressions,
            df_impressions_contextual_global_only_non_null_impressions,
            "ctr_2d_user_series_global_only_non_null",
            ["user_id", "series_id"],
        ),
    ]

    cases = (
        cases_contextual_all
        + cases_contextual_only_non_null_impressions
        + cases_global_all
        + cases_global_only_non_null_impressions
    )

    list_results = []

    df_int: pd.DataFrame
    df_imp: pd.DataFrame
    name: str
    col_group_by: str | Sequence[str]
    for df_int, df_imp, name, col_group_by in cases:
        logger.debug(f"Computing {name}")

        if col_group_by == "global":
            global_num_impressions = df_imp["item_id"].count()
            global_num_interactions = df_int["item_id"].count()

            arr_ctr = np.asarray(
                [global_num_interactions / global_num_impressions],
                dtype=np.float32,
            )
        else:
            df_num_int_num_imp = _compute_num_int_num_imp_ctr(
                cols_group_by=col_group_by,
                df_imp=df_imp,
                df_int=df_int,
            )
            arr_ctr = df_num_int_num_imp["ctr"].to_numpy()

        results = _compute_basic_statistics(
            arr_data=arr_ctr,
            data_discrete=False,
            name=name,
        )

        list_results.append(results)

    df_results = pd.DataFrame.from_records(data=list_results)
    df_results.to_csv(
        path_or_buf=os.path.join(dir_results, "table_ctr_statistics.csv"),
        index=True,
        header=True,
        sep=";",
        encoding="utf-8",
        decimal=",",
        float_format="%.4f",
    )

    return {"table_ctr_statistics": df_results}


def compute_ctr_1d(
    dir_results: str,
    dict_results: dict[str, pd.DataFrame],
    df_interactions_all: pd.DataFrame,
    df_interactions_only_non_null_impressions: pd.DataFrame,
    df_impressions_contextual_all: pd.DataFrame,
    df_impressions_contextual_only_non_null_impressions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    # TODO: Generate plots for day of week and hour of day.
    if all(
        filename in dict_results
        for filename in [
            "ctr_1d_user_all",
            "ctr_1d_date_all",
            "ctr_1d_series_all",
            "ctr_1d_user_only_non_null",
            "ctr_1d_date_only_non_null",
            "ctr_1d_series_only_non_null",
        ]
    ):
        logger.warning(
            "Skipping function %(function)s because all keys in the dictionary already exist.",
            {"function": compute_ctr_1d.__name__},
        )
        return {}

    cases = [
        (
            df_interactions_all,
            df_impressions_contextual_all,
            "ctr_1d_user_all",
            "Users",
            "user_id",
        ),
        (
            df_interactions_all,
            df_impressions_contextual_all,
            "ctr_1d_date_all",
            "Dates",
            "date",
        ),
        (
            df_interactions_all,
            df_impressions_contextual_all,
            "ctr_1d_series_all",
            "Series",
            "series_id",
        ),
        (
            df_interactions_only_non_null_impressions,
            df_impressions_contextual_only_non_null_impressions,
            "ctr_1d_user_only_non_null",
            "Users",
            "user_id",
        ),
        (
            df_interactions_only_non_null_impressions,
            df_impressions_contextual_only_non_null_impressions,
            "ctr_1d_date_only_non_null",
            "Dates",
            "date",
        ),
        (
            df_interactions_only_non_null_impressions,
            df_impressions_contextual_only_non_null_impressions,
            "ctr_1d_series_only_non_null",
            "Series",
            "series_id",
        ),
    ]

    results = {}
    for df_int, df_imp, name, label, col_group_by in cases:
        df_num_int_num_imp = _compute_num_int_num_imp_ctr(
            cols_group_by=col_group_by,
            df_imp=df_imp,
            df_int=df_int,
        )

        x_data = col_group_by
        y_data = "ctr"
        x_err = None
        y_err = None
        x_label = f"{label} rank"
        y_label = "Click-through rate (CTR)"
        x_scale: Literal["linear"] = "linear"
        y_scale: Literal["linear"] = "linear"
        if "date" == x_data:
            plot_dates(
                dir_results=dir_results,
                df=df_num_int_num_imp,
                x_data=x_data,
                y_data=y_data,
                x_date=True,
                y_date=False,
                x_label=x_label,
                y_label=y_label,
                name=name,
                x_err=x_err,
                y_err=y_err,
            )
        elif "date" != y_data:
            df_pop = df_num_int_num_imp.sort_values(
                by=y_data,
                axis="rows",
                ascending=False,
                inplace=False,
                ignore_index=True,
            )
            df_pop[x_data] = np.arange(
                start=0,
                stop=df_pop.shape[0],
                step=1,
            )

            plot_popularity(
                df=df_pop,
                dir_results=dir_results,
                x_data=x_data,
                y_data=y_data,
                x_label=x_label,
                y_label=y_label,
                x_scale=x_scale,
                y_scale=y_scale,
                name=name,
            )
        else:
            pass

        results[name] = df_num_int_num_imp

    return results


def compute_ctr_2d(
    dir_results: str,
    dict_results: dict[str, pd.DataFrame],
    df_interactions_all: pd.DataFrame,
    df_interactions_only_non_null_impressions: pd.DataFrame,
    df_impressions_contextual_all: pd.DataFrame,
    df_impressions_contextual_only_non_null_impressions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    # TODO: Generate plots for day of week and hour of day.
    if all(
        filename in dict_results
        for filename in [
            # TODO: We must compute the average of the right-most (e.g., on user_series, compute the series mean) and
            #  then use those means+std_dev to create error bars. May be more informative than the uni-dimensional.
            "ctr_2d_user_series_all",
            "ctr_2d_user_date_all",
            #
            "ctr_2d_series_user_all",
            "ctr_2d_series_date_all",
            #
            "ctr_2d_date_user_all",
            "ctr_2d_date_series_all",
            #
            "ctr_2d_user_date_only_non_null",
            "ctr_2d_user_series_only_non_null",
            #
            "ctr_2d_series_user_only_non_null",
            "ctr_2d_series_date_only_non_null",
            #
            "ctr_2d_date_user_only_non_null",
            "ctr_2d_date_series_only_non_null",
        ]
    ):
        logger.warning(
            "Skipping function %(function)s because all keys in the dictionary already exist.",
            {"function": compute_ctr_2d.__name__},
        )
        return {}

    cases = [
        (
            df_interactions_all,
            df_impressions_contextual_all,
            "ctr_2d_user_series_all",
            ["Users", "Series"],
            ["user_id", "series_id"],
        ),
        (
            df_interactions_all,
            df_impressions_contextual_all,
            "ctr_2d_user_date_all",
            ["Users", "Dates"],
            ["user_id", "date"],
        ),
        #
        (
            df_interactions_all,
            df_impressions_contextual_all,
            "ctr_2d_series_user_all",
            ["Series", "Users"],
            ["series_id", "user_id"],
        ),
        (
            df_interactions_all,
            df_impressions_contextual_all,
            "ctr_2d_series_date_all",
            ["Series", "Dates"],
            ["series_id", "date"],
        ),
        #
        (
            df_interactions_all,
            df_impressions_contextual_all,
            "ctr_2d_date_user_all",
            ["Dates", "Users"],
            ["date", "user_id"],
        ),
        (
            df_interactions_all,
            df_impressions_contextual_all,
            "ctr_2d_date_series_all",
            ["Dates", "Series"],
            ["date", "series_id"],
        ),
        #
        #
        (
            df_interactions_only_non_null_impressions,
            df_impressions_contextual_only_non_null_impressions,
            "ctr_2d_user_series_only_non_null",
            ["Users", "Series"],
            ["user_id", "series_id"],
        ),
        (
            df_interactions_only_non_null_impressions,
            df_impressions_contextual_only_non_null_impressions,
            "ctr_2d_user_date_only_non_null",
            ["Users", "Dates"],
            ["user_id", "date"],
        ),
        #
        (
            df_interactions_only_non_null_impressions,
            df_impressions_contextual_only_non_null_impressions,
            "ctr_2d_series_user_only_non_null",
            ["Series", "Users"],
            ["series_id", "user_id"],
        ),
        (
            df_interactions_only_non_null_impressions,
            df_impressions_contextual_only_non_null_impressions,
            "ctr_2d_series_date_only_non_null",
            ["Series", "Dates"],
            ["series_id", "date"],
        ),
        #
        (
            df_interactions_only_non_null_impressions,
            df_impressions_contextual_only_non_null_impressions,
            "ctr_2d_date_user_only_non_null",
            ["Dates", "Users"],
            ["date", "user_id"],
        ),
        (
            df_interactions_only_non_null_impressions,
            df_impressions_contextual_only_non_null_impressions,
            "ctr_2d_date_series_only_non_null",
            ["Dates", "Series"],
            ["date", "series_id"],
        ),
    ]

    results = {}

    df_int: pd.DataFrame
    df_imp: pd.DataFrame
    name: str
    labels: list[str]
    cols_group_by: list[str]
    for df_int, df_imp, name, labels, cols_group_by in cases:
        df_num_int_num_imp = _compute_num_int_num_imp_ctr(
            cols_group_by=cols_group_by,
            df_imp=df_imp,
            df_int=df_int,
        )

        col_first: str
        col_second: str
        col_first, col_second = cols_group_by

        label_first: str
        label_second: str
        label_first, label_second = labels

        df_ctr_2d_mean_std = (
            df_num_int_num_imp.groupby(
                by=col_first,
                as_index=False,
            )["ctr"]
            .agg(
                ["mean", "std"],
            )
            .fillna({"std": 0.0})  # some std is nan.
        )

        x_data = col_first
        y_data = "mean"
        x_err = None
        y_err = None  # "std"
        x_label = label_first
        y_label = f"Mean click-through rate\n{col_second}"
        if "date" == x_data:
            plot_dates(
                dir_results=dir_results,
                df=df_ctr_2d_mean_std,
                x_data=x_data,
                y_data=y_data,
                x_date=True,
                y_date=False,
                x_label=x_label,
                y_label=y_label,
                name=name,
                x_err=x_err,
                y_err=y_err,
            )
        elif "date" != x_data:
            df_pop = df_ctr_2d_mean_std.sort_values(
                by=y_data,
                axis="rows",
                ascending=False,
                inplace=False,
                ignore_index=True,
            )
            df_pop[x_data] = np.arange(start=0, stop=df_pop.shape[0], step=1)

            plot_popularity(
                df=df_pop,
                dir_results=dir_results,
                x_data=x_data,
                y_data=y_data,
                x_label=x_label,
                y_label=y_label,
                x_scale="linear",
                y_scale="linear",
                x_err=x_err,
                y_err=y_err,
                name=name,
            )
        else:
            pass

        results[name] = df_ctr_2d_mean_std

    return results


def compute_ctr_3d(
    dir_results: str,
    dict_results: dict[str, pd.DataFrame],
    df_interactions_all: pd.DataFrame,
    df_interactions_only_non_null_impressions: pd.DataFrame,
    df_impressions_contextual_all: pd.DataFrame,
    df_impressions_contextual_only_non_null_impressions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    # TODO: Generate plots for day of week and hour of day.
    if all(
        filename in dict_results
        for filename in [
            "ctr_3d_user_series_date_all",
            "ctr_3d_user_series_date_only_non_null_impressions",
            "ctr_3d_user_date_series_all",
            "ctr_3d_user_date_series_only_non_null_impressions",
            #
            "ctr_3d_series_user_date_all",
            "ctr_3d_series_user_date_only_non_null_impressions",
            "ctr_3d_series_date_user_all",
            "ctr_3d_series_date_user_only_non_null_impressions",
            #
            "ctr_3d_date_user_series_all",
            "ctr_3d_date_user_series_only_non_null_impressions",
            "ctr_3d_date_series_user_all",
            "ctr_3d_date_series_user_only_non_null_impressions",
        ]
    ):
        logger.warning(
            "Skipping function %(function)s because all keys in the dictionary already exist.",
            {"function": compute_ctr_3d.__name__},
        )
        return {}

    cases = [
        (
            df_interactions_all,
            df_impressions_contextual_all,
            "ctr_3d_user_series_date_all",
            ["Users", "Series", "Date"],
            ["user_id", "series_id", "date"],
        ),
        (
            df_interactions_only_non_null_impressions,
            df_impressions_contextual_only_non_null_impressions,
            "ctr_3d_user_series_date_only_non_null_impressions",
            ["Users", "Series", "Date"],
            ["user_id", "series_id", "date"],
        ),
        (
            df_interactions_all,
            df_impressions_contextual_all,
            "ctr_3d_user_date_series_all",
            ["Users", "Date", "Series"],
            ["user_id", "date", "series_id"],
        ),
        (
            df_interactions_only_non_null_impressions,
            df_impressions_contextual_only_non_null_impressions,
            "ctr_3d_user_date_series_only_non_null_impressions",
            ["Users", "Date", "Series"],
            ["user_id", "date", "series_id"],
        ),
        #
        (
            df_interactions_all,
            df_impressions_contextual_all,
            "ctr_3d_series_user_date_all",
            ["Series", "Users", "Date"],
            ["series_id", "user_id", "date"],
        ),
        (
            df_interactions_only_non_null_impressions,
            df_impressions_contextual_only_non_null_impressions,
            "ctr_3d_series_user_date_only_non_null_impressions",
            ["Series", "Users", "Date"],
            ["series_id", "user_id", "date"],
        ),
        (
            df_interactions_all,
            df_impressions_contextual_all,
            "ctr_3d_series_date_user_all",
            ["Series", "Date", "Users"],
            ["series_id", "date", "user_id"],
        ),
        (
            df_interactions_only_non_null_impressions,
            df_impressions_contextual_only_non_null_impressions,
            "ctr_3d_series_date_user_only_non_null_impressions",
            ["Series", "Date", "Users"],
            ["series_id", "date", "user_id"],
        ),
        #
        #
        (
            df_interactions_all,
            df_impressions_contextual_all,
            "ctr_3d_date_user_series_all",
            ["Date", "Users", "Series"],
            ["date", "user_id", "series_id"],
        ),
        (
            df_interactions_only_non_null_impressions,
            df_impressions_contextual_only_non_null_impressions,
            "ctr_3d_date_user_series_only_non_null_impressions",
            ["Date", "Users", "Series"],
            ["date", "user_id", "series_id"],
        ),
        (
            df_interactions_all,
            df_impressions_contextual_all,
            "ctr_3d_date_series_user_all",
            ["Date", "Series", "Users"],
            ["date", "series_id", "user_id"],
        ),
        (
            df_interactions_only_non_null_impressions,
            df_impressions_contextual_only_non_null_impressions,
            "ctr_3d_date_series_user_only_non_null_impressions",
            ["Date", "Series", "Users"],
            ["date", "series_id", "user_id"],
        ),
    ]

    results = {}
    for df_int, df_imp, name, labels, cols_group_by in cases:
        df_num_int_num_imp = _compute_num_int_num_imp_ctr(
            cols_group_by=cols_group_by,
            df_int=df_int,
            df_imp=df_imp,
        )

        col_first: str
        col_second: str
        col_third: str
        col_first, col_second, col_third = cols_group_by

        label_first: str
        label_second: str
        label_third: str
        label_first, label_second, label_third = labels

        df_ctr_mean_std = (
            df_num_int_num_imp.groupby(
                by=[col_first, col_second],
                as_index=False,
            )["ctr"]
            .agg(["mean"])
            .rename(columns={"mean": "mean_ctr"})
        )

        df_ctr_3d_mean_stddev = df_ctr_mean_std.groupby(
            by=col_first,
            as_index=False,
        )[
            "mean_ctr"
        ].agg(["mean", "std"])

        x_data = col_first
        y_data = "mean"
        x_err = None
        y_err = None  # "std"
        x_label = label_first
        y_label = f"Mean click-through rate\n{col_second} - {col_third}"

        if "date" == x_data:
            plot_dates(
                dir_results=dir_results,
                df=df_ctr_3d_mean_stddev,
                x_data=x_data,
                y_data=y_data,
                x_date=True,
                y_date=False,
                x_label=x_label,
                y_label=y_label,
                name=name,
                x_err=x_err,
                y_err=y_err,
            )
        elif "date" != x_data:
            df_pop = df_ctr_3d_mean_stddev.sort_values(
                by=y_data,
                axis="rows",
                ascending=False,
                inplace=False,
                ignore_index=True,
            )
            df_pop[x_data] = np.arange(
                start=0,
                stop=df_pop.shape[0],
                step=1,
            )

            plot_popularity(
                dir_results=dir_results,
                df=df_pop,
                x_data=x_data,
                y_data=y_data,
                x_err=x_err,
                y_err=y_err,
                x_label=x_label,
                y_label=y_label,
                x_scale="linear",
                y_scale="linear",
                name=f"{name} rank",
            )
        else:
            pass

        results[name] = df_ctr_3d_mean_stddev

    return results


def transform_dataframes_add_date_time_hour_to_interactions(
    *,
    dfs: list[pd.DataFrame],
) -> list[pd.DataFrame]:
    new_dfs = []
    for df in dfs:
        df = df.reset_index(
            drop=False,
        )
        df["datetime"] = pd.to_datetime(
            df["utc_ts_milliseconds"],
            errors="raise",
            utc=True,
            unit="ms",
        )
        df["date"] = df["datetime"].dt.date
        df["time"] = df["datetime"].dt.time
        df["hour"] = df["datetime"].dt.hour
        df["month"] = df["datetime"].dt.month

        new_dfs.append(df)

    return new_dfs


def transform_dataframes_for_ctr_computation(
    *,
    df_interactions_all: pd.DataFrame,
    df_interactions_only_null_impressions: pd.DataFrame,
    df_interactions_only_non_null_impressions: pd.DataFrame,
    df_impressions_contextual: pd.DataFrame,
    df_impressions_global: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_impressions_contextual = df_impressions_contextual.reset_index(
        drop=False,
    )

    # This dataframe holds all recorded impressions without interactions (useful when comparing with only interactions from contextual impressions)
    # at the same time, the dataframe holds repeated impressions if a user interacted more than once with the same impression.
    df_impressions_contextual_only_non_null_impressions = (
        df_interactions_all.merge(
            right=df_impressions_contextual,
            on="recommendation_id",
            how="left",
            suffixes=("", ""),
        )
        .explode(column="recommended_series_list")
        .dropna(inplace=False, ignore_index=True)
        .drop(columns=["series_id"])
        .rename(columns={"recommended_series_list": "series_id"})[
            # the `item_id` column is to make aggregations, not used at all in computations.
            ["date", "hour", "user_id", "series_id", "item_id"]
        ]
    )

    # This dataframe holds impressions on interactions outside and inside contextual impressions.
    # As for those interactions outside contextual impressions we do not have the corresponding impression,
    # we at least count the interaction as one impression.
    # This must not be done on the other dataframe because the other has the impression on interacted items.
    # In this case, the only dataframe we must use is `df_interactions_only_null_impressions`.
    # If not, we're counting impressions twice (one on the contextual impression and one on the interaction)
    df_impressions_contextual_all = pd.concat(
        objs=[
            df_interactions_only_null_impressions,
            df_impressions_contextual_only_non_null_impressions,
        ],
        axis="index",
        ignore_index=True,
    )[
        # the `item_id` column is to make aggregations, not used at all in computations.
        ["date", "hour", "user_id", "series_id", "item_id"]
    ]

    # Remove unnecessary columns on interactions
    # This dataframe holds all interactions
    df_interactions_all = df_interactions_all[
        # the `recommendation_id` column is used in the merge.
        # the `item_id` column is to make aggregations, not used at all in computations.
        ["date", "hour", "user_id", "series_id", "item_id"]
    ]

    # This dataframe holds ony interactions on contextual impressions.
    df_interactions_only_non_null_impressions = df_interactions_only_non_null_impressions[
        # the `recommendation_id` column is used in the merge.
        # the `item_id` column is to make aggregations, not used at all in computations.
        ["date", "hour", "user_id", "series_id", "item_id"]
    ]

    # This dataframe holds impressions that received no interaction from the users.
    df_impressions_global = (
        df_impressions_global.reset_index(drop=False)
        .explode(column="recommended_series_list")
        .dropna(inplace=False, ignore_index=True)
        .assign(
            item_id=-1,
            date=pd.to_datetime("2019-12-01 00:00:00", utc=True).date(),
            hour=pd.to_datetime("2019-12-01 00:00:00", utc=True).hour,
        )
        .rename(columns={"recommended_series_list": "series_id"})[
            # the `item_id` column is to make aggregations, not used at all in computations.
            ["date", "hour", "user_id", "series_id", "item_id"]
        ]
    )

    return (
        df_interactions_all,
        df_interactions_only_non_null_impressions,
        df_impressions_contextual_all,
        df_impressions_contextual_only_non_null_impressions,
        df_impressions_global,
    )


def contentwise_impressions_compute_statistics_thesis(
    dir_results: str,
) -> dict:
    filename = "statistics_thesis_content_wise_impressions.zip"

    try:
        dict_results: dict[str, pd.DataFrame] = DataIO.s_load_data(
            folder_path=dir_results,
            file_name=filename,
        )
    except FileNotFoundError:
        dict_results = {}

    logger.info(f"starting script after loading previously exported info.")

    config = cw_impressions_reader.ContentWiseImpressionsConfig()

    cw_impressions_raw_data = cw_impressions_reader.ContentWiseImpressionsRawData(
        config=config,
    )

    df_interactions_all: pd.DataFrame = cw_impressions_raw_data.interactions.compute()
    df_interactions_only_null_impressions: pd.DataFrame = df_interactions_all[
        df_interactions_all["recommendation_id"] == -1
    ]
    df_interactions_only_non_null_impressions: pd.DataFrame = df_interactions_all[
        df_interactions_all["recommendation_id"] >= 0
    ]

    df_impressions_contextual = cw_impressions_raw_data.impressions.compute()
    df_impressions_global = (
        cw_impressions_raw_data.impressions_non_direct_link.compute()
    )

    (
        df_interactions_all,
        df_interactions_only_null_impressions,
        df_interactions_only_non_null_impressions,
    ) = transform_dataframes_add_date_time_hour_to_interactions(
        dfs=[
            df_interactions_all,
            df_interactions_only_null_impressions,
            df_interactions_only_non_null_impressions,
        ]
    )

    # results = compute_popularity_interactions(
    #     dir_results=dir_results,
    #     dict_results=dict_results,
    #     df_interactions_all=df_interactions_all,
    #     df_interactions_only_null_impressions=df_interactions_only_null_impressions,
    #     df_interactions_only_non_null_impressions=df_interactions_only_non_null_impressions,
    # )
    # dict_results.update(results)
    #
    # results = compute_popularity_impressions_contextual(
    #     dir_results=dir_results,
    #     dict_results=dict_results,
    #     df_interactions_only_non_null_impressions=df_interactions_only_non_null_impressions,
    #     df_impressions_contextual=df_impressions_contextual,
    # )
    # dict_results.update(results)
    #
    # results = compute_popularity_impressions_global(
    #     dir_results=dir_results,
    #     dict_results=dict_results,
    #     df_impressions_global=df_impressions_global,
    # )
    # dict_results.update(results)
    #
    # results = compute_correlation_interactions_impressions(
    #     dir_results=dir_results,
    #     dict_results=dict_results,
    # )
    # dict_results.update(results)
    #
    # results = compute_daily_hourly_number_of_interactions(
    #     dir_results=dir_results,
    #     dict_results=dict_results,
    #     df_interactions_all=df_interactions_all,
    # )
    # dict_results.update(results)
    #
    # results = compute_daily_hourly_number_of_impressions(
    #     dir_results=dir_results,
    #     dict_results=dict_results,
    #     df_interactions_only_non_null_impressions=df_interactions_only_non_null_impressions,
    #     df_impressions_contextual=df_impressions_contextual,
    # )
    # dict_results.update(results)
    #
    # results = compute_vision_factor(
    #     dir_results=dir_results,
    #     dict_results=dict_results,
    #     df_interactions_all=df_interactions_all,
    #     df_interactions_only_null_impressions=df_interactions_only_null_impressions,
    #     df_interactions_only_non_null_impressions=df_interactions_only_non_null_impressions,
    # )
    # dict_results.update(results)
    #
    results = compute_ratings(
        dir_results=dir_results,
        dict_results=dict_results,
        df_interactions_all=df_interactions_all,
        df_interactions_only_null_impressions=df_interactions_only_null_impressions,
        df_interactions_only_non_null_impressions=df_interactions_only_non_null_impressions,
    )
    dict_results.update(results)

    # (
    #     df_interactions_all,
    #     df_interactions_only_non_null_impressions,
    #     df_impressions_contextual_all,
    #     df_impressions_contextual_only_non_null_impressions,
    #     df_impressions_global,
    # ) = transform_dataframes_for_ctr_computation(
    #     df_interactions_all=df_interactions_all,
    #     df_interactions_only_null_impressions=df_interactions_only_null_impressions,
    #     df_interactions_only_non_null_impressions=df_interactions_only_non_null_impressions,
    #     df_impressions_contextual=df_impressions_contextual,
    #     df_impressions_global=df_impressions_global,
    # )
    #
    # results = compute_table_ctr(
    #     dir_results=dir_results,
    #     dict_results=dict_results,
    #     df_interactions_all=df_interactions_all,
    #     df_interactions_only_non_null_impressions=df_interactions_only_non_null_impressions,
    #     df_impressions_contextual_all=df_impressions_contextual_all,
    #     df_impressions_contextual_only_non_null_impressions=df_impressions_contextual_only_non_null_impressions,
    #     df_impressions_global=df_impressions_global,
    # )
    # dict_results.update(results)

    # results = compute_ctr_1d(
    #     dir_results=dir_results,
    #     dict_results=dict_results,
    #     df_interactions_all=df_interactions_all,
    #     df_interactions_only_non_null_impressions=df_interactions_only_non_null_impressions,
    #     df_impressions_contextual_all=df_impressions_contextual_all,
    #     df_impressions_contextual_only_non_null_impressions=df_impressions_contextual_only_non_null_impressions,
    # )
    # dict_results.update(results)
    #
    # results = compute_ctr_2d(
    #     dir_results=dir_results,
    #     dict_results=dict_results,
    #     df_interactions_all=df_interactions_all,
    #     df_interactions_only_non_null_impressions=df_interactions_only_non_null_impressions,
    #     df_impressions_contextual_all=df_impressions_contextual_all,
    #     df_impressions_contextual_only_non_null_impressions=df_impressions_contextual_only_non_null_impressions,
    # )
    # dict_results.update(results)
    #
    # results = compute_ctr_3d(
    #     dir_results=dir_results,
    #     dict_results=dict_results,
    #     df_interactions_all=df_interactions_all,
    #     df_interactions_only_non_null_impressions=df_interactions_only_non_null_impressions,
    #     df_impressions_contextual_all=df_impressions_contextual_all,
    #     df_impressions_contextual_only_non_null_impressions=df_impressions_contextual_only_non_null_impressions,
    # )
    # dict_results.update(results)

    DataIO.s_save_data(
        folder_path=dir_results,
        file_name=filename,
        data_dict_to_save=dict_results,
    )

    return dict_results
