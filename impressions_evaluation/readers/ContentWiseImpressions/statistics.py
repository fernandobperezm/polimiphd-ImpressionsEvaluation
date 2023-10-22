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
    Unknown = -1


class ItemType(enum.Enum):
    EpisodesTVSeries = 3
    TVMoviesShows = 2
    MoviesAndClipsInSeries = 1
    Movies = 0
    Unknown = -1


DICT_ITEM_TYPE_TO_STR = {
    ItemType.EpisodesTVSeries.value: "Episodes of TV series",
    ItemType.TVMoviesShows.value: "Movies",
    ItemType.Movies.value: "TV movies and shows",
    ItemType.MoviesAndClipsInSeries.value: "Movies and clips in series",
    ItemType.Unknown.value: "Unknown",
}


DICT_INTERACTION_TYPE_TO_STR = {
    InteractionType.View.value: "View",
    InteractionType.Detail.value: "Detail",
    InteractionType.Rate.value: "Rate",
    InteractionType.Purchase.value: "Purchase",
    InteractionType.Unknown.value: "Unknown",
}

DICT_DAY_OF_WEEK_TO_STR = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}


DICT_MONTH_TO_STR = {
    0: "January",
    1: "February",
    2: "March",
    3: "April",
    4: "May",
    5: "June",
    6: "July",
    7: "August",
    8: "September",
    9: "October",
    10: "November",
    11: "December",
}


def _set_unique_items(
    *,
    df_interactions: pd.DataFrame,
) -> set[int]:
    df_series_interactions = df_interactions["item_id"].dropna(
        inplace=False,
        how="any",
        axis="index",
    )

    return set(df_series_interactions)


def _set_unique_series(
    *,
    df_interactions: pd.DataFrame,
    df_impressions: pd.DataFrame,
    df_impressions_non_direct_link: pd.DataFrame,
) -> set[int]:
    df_series_interactions = (
        df_interactions["series_id"]
        .dropna(
            inplace=False,
            how="any",
            axis="index",
        )
        .to_numpy(dtype=np.int32)
    )

    df_series_impressions_contextual = (
        df_impressions["recommended_series_list"]
        .explode(
            ignore_index=True,
        )
        .dropna(
            inplace=False,
            how="any",
            axis="index",
        )
        .to_numpy(dtype=np.int32)
    )

    df_series_impressions_global = (
        df_impressions_non_direct_link["recommended_series_list"]
        .explode(
            ignore_index=True,
        )
        .dropna(
            inplace=False,
            how="any",
            axis="index",
        )
        .to_numpy(dtype=np.int32)
    )

    return (
        set(df_series_interactions)
        .union(df_series_impressions_contextual)
        .union(df_series_impressions_global)
    )


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
    results_powerlaw_truncated_powerlaw = cast(
        tuple[float, float],
        results_powerlaw.distribution_compare("power_law", "truncated_power_law"),
    )
    results_powerlaw_stretched_exponential = cast(
        tuple[float, float],
        results_powerlaw.distribution_compare("power_law", "stretched_exponential"),
    )
    results_powerlaw_lognormal_positive = cast(
        tuple[float, float],
        results_powerlaw.distribution_compare("power_law", "lognormal_positive"),
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
        "powerlaw_sigma": results_powerlaw.sigma,
        "powerlaw_lognormal_likelihood": results_powerlaw_lognormal[0],
        "powerlaw_lognormal_pvalue": results_powerlaw_lognormal[1],
        "powerlaw_exponential_likelihood": results_powerlaw_exponential[0],
        "powerlaw_exponential_pvalue": results_powerlaw_exponential[1],
        "powerlaw_truncated_powerlaw_likelihood": results_powerlaw_truncated_powerlaw[
            0
        ],
        "powerlaw_truncated_powerlaw_pvalue": results_powerlaw_truncated_powerlaw[1],
        "powerlaw_stretched_exponential_likelihood": results_powerlaw_stretched_exponential[
            0
        ],
        "powerlaw_stretched_exponential_pvalue": results_powerlaw_stretched_exponential[
            1
        ],
        "powerlaw_lognormal_positive_likelihood": results_powerlaw_lognormal_positive[
            0
        ],
        "powerlaw_lognormal_positive_pvalue": results_powerlaw_lognormal_positive[1],
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
    unique_items = _set_unique_series(
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
        .assign(cumsum=lambda df_: df_["count"].cumsum())
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
        .assign(cumsum=lambda df_: df_["count"].cumsum())
    )

    return col_popularity, col_popularity_perc


def compute_position_of_interaction_on_impression(
    df_row: pd.DataFrame,
) -> int:
    series_id = df_row["series_id"]
    impression = df_row["recommended_series_list"]

    for position, rec_series_id in enumerate(impression):
        if rec_series_id == series_id:
            return position

    return -1


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
    x_ticks: Optional[np.ndarray] = None,
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
    ax.errorbar(x=x_data, y=y_data, x_err=x_err, yerr=y_err, data=df)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # If one axis is of type `date` then the other must be linear, log, etc.
    if x_date:
        ax.set_yscale("linear")
    if y_date:
        ax.set_xscale("linear")
    if x_ticks is not None:
        ax.set_xticks(x_ticks)

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

    # We must convert data into strings due to an error in the library. See: https://github.com/nschloe/tikzplotlib/issues/440
    # THIS DOES NOT WORK; PLEASE DO NOT UNCOMMENT.
    # if isinstance(df, pd.DataFrame):
    #     df = df.astype({x_data: pd.StringDtype(), y_data: pd.StringDtype()})
    # else:
    #     df = [
    #         (df_.astype({x_data: pd.StringDtype(), y_data: pd.StringDtype()}), group)
    #         for df_, group in df
    #     ]

    ax.bar(
        x_data,
        y_data,
        data=df,
        width=0.5,
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

    folder_to_save_tikz = os.path.join(dir_results, "tikz", "")
    folder_to_save_png = os.path.join(dir_results, "png", "")

    os.makedirs(folder_to_save_tikz, exist_ok=True)
    os.makedirs(folder_to_save_png, exist_ok=True)

    # Not supported yet as per warning: "UserWarning: Cleaning Bar Container (bar plot) is not supported yet"
    # tikzplotlib.clean_figure(fig=fig)
    tikzplotlib.save(
        os.path.join(
            folder_to_save_tikz, f"plot-{plot_name}-{name}.tikz"
        ),  # cannot be kwarg!
        fig,  # cannot be kwarg!
        encoding="utf-8",
        textsize=9,
    )

    fig.savefig(os.path.join(folder_to_save_png, f"plot-{plot_name}-{name}.png"))
    # fig.show()
    plt.close(fig)  # ensure proper disposal/destruction of fig object.


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
    if bins is None:
        bins = 10

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

    hist, bin_edges = np.histogram(
        df[x_data],
        bins=bins,
    )

    # Try to see if this works.
    ax.set_xticks(bin_edges)
    ax.set_yticks(hist)

    # Try to see if this works.
    # ax.set_xticks(
    #     np.linspace(
    #         start=df[x_data].min() - 1,
    #         stop=df[x_data].max() + 1,
    #         num=20 if bins is None else bins * 2,
    #     )
    # )
    # ax.set_yticks()

    folder_to_save_tikz = os.path.join(dir_results, "tikz", "")
    folder_to_save_png = os.path.join(dir_results, "png", "")

    os.makedirs(folder_to_save_tikz, exist_ok=True)
    os.makedirs(folder_to_save_png, exist_ok=True)

    # It says it is not supported yet
    # tikzplotlib.clean_figure(fig=fig)
    tikzplotlib.save(
        os.path.join(folder_to_save_tikz, f"plot-hist-{name}.tikz"),  # cannot be kwarg
        fig,  # cannot be kwarg
        encoding="utf-8",
        textsize=9,
    )

    # fig.savefig(os.path.join(folder_to_save_png, f"plot-hist-{name}.png"))  # see if xticks are not being printed because of this?
    # fig.show()
    plt.close(fig)


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


def plot_cdf_pdf_ccdf(
    *,
    dir_results: str,
    df: pd.DataFrame,
    x_data: str,
    x_label: str,
    name: str,
) -> None:
    fig: plt.Figure
    ax: plt.Axes

    arr_data = df[x_data].to_numpy()

    cases = [
        (
            powerlaw.plot_pdf,
            "pdf",
            "PDF",
            {"linear_bins": False},
        ),
        (
            powerlaw.plot_cdf,
            "cdf",
            r"CDF - $\Pr{\left(X < x\right)}$",
            {"survival": False},
        ),
        (
            powerlaw.plot_ccdf,
            "ccdf",
            r"CCDF - $\Pr{\left(X \geq x\right)}$",
            {"survival": True},
        ),
    ]

    for func, plot_name, y_label, kwargs in cases:
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(
                SIZE_INCHES_WIDTH,
                SIZE_INCHES_HEIGHT,
            ),  # Must be (width, height) by the docs.
            layout="compressed",
        )

        ax = func(data=arr_data.copy(), ax=ax, **kwargs)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        tikzplotlib.clean_figure(fig=fig)
        tikzplotlib.save(
            os.path.join(
                dir_results, f"plot-{plot_name}-{name}.tikz"
            ),  # cannot be kwargs.
            fig,  # cannot be kwargs.
            encoding="utf-8",
            textsize=9,
        )

        fig.show()


def compute_metadata_dataset(
    metadata: dict[str, Any],
    df_interactions: pd.DataFrame,
    df_impressions_contextual: pd.DataFrame,
    df_impressions_global: pd.DataFrame,
) -> dict:
    df_interactions = df_interactions.reset_index(drop=False)
    df_impressions_contextual = df_impressions_contextual.reset_index(drop=False)
    df_impressions_global = df_impressions_global.reset_index(drop=False)

    num_users = len(
        _set_unique_users(
            df_interactions=df_interactions,
            df_impressions_non_direct_link=df_impressions_global,
        )
    )

    num_items = len(
        _set_unique_items(
            df_interactions=df_interactions,
        )
    )

    num_series = len(
        _set_unique_series(
            df_interactions=df_interactions,
            df_impressions=df_impressions_contextual,
            df_impressions_non_direct_link=df_impressions_global,
        )
    )

    if num_users != metadata["num_users"]:
        logger.warning(
            f"The number of users in the metadata and computed in the dataset are different. Computed value is %(computed_value)s while value in metadata is %(metadata_value)s. Using computed value.",
            {"computed_value": num_users, "metadata_value": metadata["num_users"]},
        )
        metadata["num_users"] = num_users

    if num_series != metadata["num_series"]:
        logger.warning(
            f"The number of series in the metadata and computed in the dataset are different. Computed value is %(computed_value)s while value in metadata is %(metadata_value)s. Using computed value.",
            {"computed_value": num_series, "metadata_value": metadata["num_series"]},
        )
        metadata["num_series"] = num_series

    if num_items != metadata["num_items"]:
        logger.warning(
            f"The number of items in the metadata and computed in the dataset are different. Computed value is %(computed_value)s while value in metadata is %(metadata_value)s. Using computed value.",
            {"computed_value": num_items, "metadata_value": metadata["num_items"]},
        )
        metadata["num_items"] = num_items

    return metadata


def compute_number_unique_items_by_item_type(
    *,
    dir_results: str,
    dict_results: dict[str, pd.DataFrame],
    df_interactions_with_impressions_all: pd.DataFrame,
    df_interactions_outside_impressions: pd.DataFrame,
    df_interactions_inside_impressions: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    if all(
        filename in dict_results
        for filename in [
            "num_unique_items_by_item_type",
            "num_unique_items_by_item_type-series_all",
            "num_unique_items_by_item_type-items_all",
            "num_unique_items_by_item_type-series_inside_impressions",
            "num_unique_items_by_item_type-items_inside_impressions",
            "num_unique_items_by_item_type-series_outside_impressions",
            "num_unique_items_by_item_type-items_outside_impressions",
        ]
    ):
        logger.warning(
            "Skipping function %(function)s because all keys in the dictionary already exist.",
            {"function": compute_number_unique_items_by_item_type.__name__},
        )
        return {}

    num_unique_series: int = metadata["num_series"]
    num_unique_items: int = metadata["num_items"]

    cases_all = [
        (
            df_interactions_with_impressions_all,
            "All",
            "Items",
            "item_id",
            num_unique_items,
            "items_all",
        ),
        (
            df_interactions_with_impressions_all,
            "All",
            "Series",
            "series_id",
            num_unique_series,
            "series_all",
        ),
    ]
    cases_inside_impressions = [
        (
            df_interactions_inside_impressions,
            "Inside contextual impressions",
            "Items",
            "item_id",
            num_unique_items,
            "items_inside_impressions",
        ),
        (
            df_interactions_inside_impressions,
            "Inside contextual impressions",
            "Series",
            "series_id",
            num_unique_series,
            "series_inside_impressions",
        ),
    ]
    cases_outside_impressions = [
        (
            df_interactions_outside_impressions,
            "Outside contextual impressions",
            "Items",
            "item_id",
            num_unique_items,
            "items_outside_impressions",
        ),
        (
            df_interactions_outside_impressions,
            "Outside contextual impressions",
            "Series",
            "series_id",
            num_unique_series,
            "series_outside_impressions",
        ),
    ]

    cases = cases_all + cases_inside_impressions + cases_outside_impressions

    results = {}
    dfs = []
    for df, subset, attr, col, num_unique, name in cases:
        df_known: pd.DataFrame = (
            df.drop_duplicates(
                subset=[col],
                keep="first",
                inplace=False,
                ignore_index=True,
            )
            .groupby(
                by=["item_type"],
                as_index=False,
            )[col]
            .agg(["count"])
            .assign(
                perc=lambda df_: df_["count"] / num_unique,
                item_type_str=lambda df_: df_["item_type"].apply(
                    lambda df_row_item_type: DICT_ITEM_TYPE_TO_STR[df_row_item_type],
                    convert_dtype=True,
                ),
            )
            .astype({"perc": np.float32, "item_type_str": pd.StringDtype()})
        )

        df_unknown = pd.DataFrame.from_records(
            data=[
                {
                    "item_type": ItemType.Unknown.value,
                    "item_type_str": DICT_ITEM_TYPE_TO_STR[ItemType.Unknown.value],
                    "count": num_unique - df_known["count"].sum(),
                    "perc": 1 - df_known["perc"].sum(),
                }
            ]
        ).astype(
            dtype={
                "item_type": df_known["item_type"].dtype,
                "item_type_str": pd.StringDtype(),
                "count": df_known["count"].dtype,
                "perc": df_known["perc"].dtype,
            }
        )

        df_known_and_unknown = (
            pd.concat(
                objs=[df_known, df_unknown],
                axis=0,
                ignore_index=True,
            )
            .sort_values(
                by=["count"],
                axis=0,
                ascending=False,
            )
            .assign(subset=subset, attribute=attr)
            .astype({"subset": pd.StringDtype(), "attribute": pd.StringDtype()})[
                ["subset", "attribute", "item_type_str", "count", "perc"]
            ]
        )

        dfs.append(df_known_and_unknown)

        results[f"num_unique_items_by_item_type-{name}"] = df_known_and_unknown

    df_results = pd.concat(
        objs=dfs,
        axis=0,
        ignore_index=True,
    )
    df_results.to_csv(
        path_or_buf=os.path.join(
            dir_results, f"table-num_unique_items_by_item_type.csv"
        ),
        sep=";",
        float_format="%.4f",
        header=True,
        index=True,
        encoding="utf-8",
        decimal=",",
    )
    results["num_unique_items_by_item_type"] = df_results

    return results


def compute_number_unique_items_by_interaction_type(
    *,
    dir_results: str,
    dict_results: dict[str, pd.DataFrame],
    df_interactions_with_impressions_all: pd.DataFrame,
    df_interactions_outside_impressions: pd.DataFrame,
    df_interactions_inside_impressions: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    if all(
        filename in dict_results
        for filename in [
            "num_unique_items_by_interaction_type",
            "num_unique_items_by_interaction_type-series_all",
            "num_unique_items_by_interaction_type-items_all",
            "num_unique_items_by_interaction_type-series_inside_impressions",
            "num_unique_items_by_interaction_type-items_inside_impressions",
            "num_unique_items_by_interaction_type-series_outside_impressions",
            "num_unique_items_by_interaction_type-items_outside_impressions",
        ]
    ):
        logger.warning(
            "Skipping function %(function)s because all keys in the dictionary already exist.",
            {"function": compute_number_unique_items_by_interaction_type.__name__},
        )
        return {}

    num_unique_series: int = metadata["num_series"]
    num_unique_items: int = metadata["num_items"]

    column_perc = "perc"
    column_count = "nunique"
    column_subset = "subset"
    column_attribute = "attribute"
    column_interaction_type = "interaction_type"
    column_interaction_type_str = "interaction_type_str"

    column_export_count = "Count"
    column_export_perc = "Percentage"
    column_export_attribute = "Attribute"
    column_export_subset = "Dataset subset"
    column_export_interaction_type_str = "Interaction type"

    cases_all = [
        (
            df_interactions_with_impressions_all,
            "All",
            "Items",
            "item_id",
            num_unique_items,
            "items_all",
        ),
        (
            df_interactions_with_impressions_all,
            "All",
            "Series",
            "series_id",
            num_unique_series,
            "series_all",
        ),
    ]
    cases_inside_impressions = [
        (
            df_interactions_inside_impressions,
            "Inside contextual impressions",
            "Items",
            "item_id",
            num_unique_items,
            "items_inside_impressions",
        ),
        (
            df_interactions_inside_impressions,
            "Inside contextual impressions",
            "Series",
            "series_id",
            num_unique_series,
            "series_inside_impressions",
        ),
    ]
    cases_outside_impressions = [
        (
            df_interactions_outside_impressions,
            "Outside contextual impressions",
            "Items",
            "item_id",
            num_unique_items,
            "items_outside_impressions",
        ),
        (
            df_interactions_outside_impressions,
            "Outside contextual impressions",
            "Series",
            "series_id",
            num_unique_series,
            "series_outside_impressions",
        ),
    ]

    cases = cases_all + cases_inside_impressions + cases_outside_impressions

    results = {}
    dfs = []
    for df, subset, attr, col, num_unique, name in cases:
        df_num_unique_items_by_interaction_type: pd.DataFrame = (
            # We don't remove duplicates of interaction types because an item that has been rated and watched must be
            # included twice.
            # The percentage is with respect to the number of unique items not the number of interactions as this is a partial sum on interactions.
            df.groupby(
                by=[column_interaction_type],
                as_index=False,
            )[col]
            .agg([column_count])
            .assign(
                **{
                    column_subset: subset,
                    column_attribute: attr,
                    column_perc: lambda df_: df_[column_count] / num_unique,
                    column_interaction_type_str: lambda df_: df_[
                        column_interaction_type
                    ].apply(
                        lambda df_row_item_type: DICT_INTERACTION_TYPE_TO_STR[
                            df_row_item_type
                        ],
                        convert_dtype=True,
                    ),
                },
            )
            .astype(
                {
                    column_perc: np.float32,
                    column_interaction_type_str: pd.StringDtype(),
                    column_subset: pd.StringDtype(),
                    column_attribute: pd.StringDtype(),
                }
            )
            .sort_values(
                by=[column_count],
                axis=0,
                ascending=False,
            )[
                [
                    column_subset,
                    column_attribute,
                    column_interaction_type_str,
                    column_count,
                    column_perc,
                ]
            ]
            .rename(
                columns={
                    column_subset: column_export_subset,
                    column_attribute: column_export_attribute,
                    column_interaction_type_str: column_export_interaction_type_str,
                    column_count: column_export_count,
                    column_perc: column_export_perc,
                },
            )
        )

        dfs.append(df_num_unique_items_by_interaction_type)

        results[
            f"num_unique_items_by_interaction_type-{name}"
        ] = df_num_unique_items_by_interaction_type

    df_results = pd.concat(
        objs=dfs,
        axis=0,
        ignore_index=True,
    )
    df_results.to_csv(
        path_or_buf=os.path.join(
            dir_results, f"table-num_unique_items_by_interaction_type.csv"
        ),
        sep=";",
        float_format="%.4f",
        header=True,
        index=True,
        encoding="utf-8",
        decimal=",",
    )
    results["num_unique_items_by_interaction_type"] = df_results

    return results


def compute_number_interactions_by_item_type(
    *,
    dir_results: str,
    dict_results: dict[str, pd.DataFrame],
    df_interactions_with_impressions_all: pd.DataFrame,
    df_interactions_outside_impressions: pd.DataFrame,
    df_interactions_inside_impressions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if all(
        filename in dict_results
        for filename in [
            "num_interactions_by_item_type",
            "num_interactions_by_item_type-series_all",
            "num_interactions_by_item_type-items_all",
            "num_interactions_by_item_type-series_inside_impressions",
            "num_interactions_by_item_type-items_inside_impressions",
            "num_interactions_by_item_type-series_outside_impressions",
            "num_interactions_by_item_type-items_outside_impressions",
        ]
    ):
        logger.warning(
            "Skipping function %(function)s because all keys in the dictionary already exist.",
            {"function": compute_number_interactions_by_item_type.__name__},
        )
        return {}

    column_perc = "perc"
    column_count = "count"
    column_subset = "subset"
    column_attribute = "attribute"
    column_item_type = "item_type"
    column_item_type_str = "item_type_str"

    column_export_count = "Count"
    column_export_perc = "Percentage"
    column_export_attribute = "Attribute"
    column_export_subset = "Dataset subset"
    column_export_item_type_str = "Interaction type"

    cases_all = [
        (
            df_interactions_with_impressions_all,
            "All",
            "Items",
            "item_id",
            "items_all",
        ),
    ]
    cases_inside_impressions = [
        (
            df_interactions_inside_impressions,
            "Inside contextual impressions",
            "Items",
            "item_id",
            "items_inside_impressions",
        ),
    ]
    cases_outside_impressions = [
        (
            df_interactions_outside_impressions,
            "Outside contextual impressions",
            "Items",
            "item_id",
            "items_outside_impressions",
        ),
    ]

    cases = cases_all + cases_inside_impressions + cases_outside_impressions

    results = {}
    dfs = []
    for df, subset, attr, col, name in cases:
        num_interactions = df.shape[0]

        df_known: pd.DataFrame = (
            df.groupby(
                by=[column_item_type],
                as_index=False,
            )[col]
            .agg([column_count])
            .assign(
                **{
                    column_subset: subset,
                    column_attribute: attr,
                    column_perc: lambda df_: df_[column_count] / num_interactions,
                    column_item_type_str: lambda df_: df_[column_item_type].apply(
                        lambda df_row_item_type: DICT_ITEM_TYPE_TO_STR[
                            df_row_item_type
                        ],
                        convert_dtype=True,
                    ),
                },
            )
            .astype(
                {
                    column_perc: np.float32,
                    column_item_type_str: pd.StringDtype(),
                    column_subset: pd.StringDtype(),
                    column_attribute: pd.StringDtype(),
                }
            )
            .sort_values(
                by=[column_count],
                axis=0,
                ascending=False,
            )[
                [
                    column_subset,
                    column_attribute,
                    column_item_type_str,
                    column_count,
                    column_perc,
                ]
            ]
            .rename(
                columns={
                    column_subset: column_export_subset,
                    column_attribute: column_export_attribute,
                    column_item_type_str: column_export_item_type_str,
                    column_count: column_export_count,
                    column_perc: column_export_perc,
                },
            )
        )

        dfs.append(df_known)

        results[f"num_interactions_by_item_type-{name}"] = df_known

    df_results = pd.concat(
        objs=dfs,
        axis=0,
        ignore_index=True,
    )
    df_results.to_csv(
        path_or_buf=os.path.join(
            dir_results, f"table-num_interactions_by_item_type.csv"
        ),
        sep=";",
        float_format="%.4f",
        header=True,
        index=True,
        encoding="utf-8",
        decimal=",",
    )
    results["num_interactions_by_item_type"] = df_results

    return results


def compute_number_of_interactions_by_interaction_type(
    *,
    dir_results: str,
    dict_results: dict[str, pd.DataFrame],
    df_interactions_with_impressions_all: pd.DataFrame,
    df_interactions_outside_impressions: pd.DataFrame,
    df_interactions_inside_impressions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if all(
        filename in dict_results
        for filename in [
            "num_interactions_by_interaction_type",
            "num_interactions_by_interaction_type-series_all",
            "num_interactions_by_interaction_type-items_all",
            "num_interactions_by_interaction_type-series_inside_impressions",
            "num_interactions_by_interaction_type-items_inside_impressions",
            "num_interactions_by_interaction_type-series_outside_impressions",
            "num_interactions_by_interaction_type-items_outside_impressions",
        ]
    ):
        logger.warning(
            "Skipping function %(function)s because all keys in the dictionary already exist.",
            {"function": compute_number_of_interactions_by_interaction_type.__name__},
        )
        return {}

    column_perc = "perc"
    column_count = "count"
    column_subset = "subset"
    column_attribute = "attribute"
    column_interaction_type = "interaction_type"
    column_interaction_type_str = "interaction_type_str"

    column_export_count = "Count"
    column_export_perc = "Percentage"
    column_export_attribute = "Attribute"
    column_export_subset = "Dataset subset"
    column_export_interaction_type_str = "Interaction type"

    cases_all = [
        (
            df_interactions_with_impressions_all,
            "All",
            "Items",
            "item_id",
            "items_all",
        ),
    ]
    cases_inside_impressions = [
        (
            df_interactions_inside_impressions,
            "Inside contextual impressions",
            "Items",
            "item_id",
            "items_inside_impressions",
        ),
    ]
    cases_outside_impressions = [
        (
            df_interactions_outside_impressions,
            "Outside contextual impressions",
            "Items",
            "item_id",
            "items_outside_impressions",
        ),
    ]

    cases = cases_all + cases_inside_impressions + cases_outside_impressions

    results = {}
    dfs = []
    for df, subset, attr, col, name in cases:
        num_interactions = df.shape[0]

        df_known: pd.DataFrame = (
            df.groupby(
                by=[column_interaction_type],
                as_index=False,
            )[col]
            .agg([column_count])
            .assign(
                **{
                    column_subset: subset,
                    column_attribute: attr,
                    column_perc: lambda df_: df_[column_count] / num_interactions,
                    column_interaction_type_str: lambda df_: df_[
                        column_interaction_type
                    ].apply(
                        lambda df_row_interaction_type: DICT_INTERACTION_TYPE_TO_STR[
                            df_row_interaction_type
                        ],
                        convert_dtype=True,
                    ),
                },
            )
            .astype(
                {
                    column_perc: np.float32,
                    column_interaction_type_str: pd.StringDtype(),
                    column_subset: pd.StringDtype(),
                    column_attribute: pd.StringDtype(),
                }
            )
            .sort_values(
                by=[column_count],
                axis=0,
                ascending=False,
            )[
                [
                    column_subset,
                    column_attribute,
                    column_interaction_type_str,
                    column_count,
                    column_perc,
                ]
            ]
            .rename(
                columns={
                    column_subset: column_export_subset,
                    column_attribute: column_export_attribute,
                    column_interaction_type_str: column_export_interaction_type_str,
                    column_count: column_export_count,
                    column_perc: column_export_perc,
                },
            )
        )

        dfs.append(df_known)

        results[f"num_interactions_by_interaction_type-{name}"] = df_known

    df_results = pd.concat(
        objs=dfs,
        axis=0,
        ignore_index=True,
    )
    df_results.to_csv(
        path_or_buf=os.path.join(
            dir_results, f"table-num_interactions_by_interaction_type.csv"
        ),
        sep=";",
        float_format="%.4f",
        header=True,
        index=True,
        encoding="utf-8",
        decimal=",",
    )
    results["num_interactions_by_interaction_type"] = df_results

    return results


def compute_popularity_interactions(
    *,
    dir_results: str,
    dict_results: dict[str, pd.DataFrame],
    df_interactions_with_impressions_all: pd.DataFrame,
    df_interactions_outside_impressions: pd.DataFrame,
    df_interactions_inside_impressions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if all(
        filename in dict_results
        for filename in [
            "table_statistics_popularity_interactions",
            "popularity_interactions_user_all",
            "popularity_interactions_item_all",
            "popularity_interactions_series_all",
            "popularity_interactions_user_inside_impressions",
            "popularity_interactions_item_inside_impressions",
            "popularity_interactions_series_inside_impressions",
            "popularity_interactions_user_outside_impressions",
            "popularity_interactions_item_outside_impressions",
            "popularity_interactions_series_outside_impressions",
        ]
    ):
        logger.warning(
            "Skipping function %(function)s because all keys in the dictionary already exist.",
            {"function": compute_popularity_interactions.__name__},
        )
        return {}

    cases_users = [
        (
            df_interactions_with_impressions_all,
            "user_id",
            "popularity_interactions_user_all",
            r"Rank of users",
            r"\# of interactions",
        ),
        (
            df_interactions_outside_impressions,
            "user_id",
            "popularity_interactions_user_outside_impressions",
            r"Rank of users",
            r"\# of interactions",
        ),
        (
            df_interactions_inside_impressions,
            "user_id",
            "popularity_interactions_user_inside_impressions",
            r"Rank of users",
            r"\# of interactions",
        ),
    ]

    cases_items = [
        (
            df_interactions_with_impressions_all,
            "item_id",
            "popularity_interactions_item_all",
            r"Rank of items",
            r"\# of interactions",
        ),
        (
            df_interactions_outside_impressions,
            "item_id",
            "popularity_interactions_item_outside_impressions",
            r"Rank of items",
            r"\# of interactions",
        ),
        (
            df_interactions_inside_impressions,
            "item_id",
            "popularity_interactions_item_inside_impressions",
            r"Rank of items",
            r"\# of interactions",
        ),
    ]

    cases_series = [
        (
            df_interactions_with_impressions_all,
            "series_id",
            "popularity_interactions_series_all",
            r"Rank of series",
            r"\# of interactions",
        ),
        (
            df_interactions_outside_impressions,
            "series_id",
            "popularity_interactions_series_outside_impressions",
            r"Rank of series",
            r"\# of interactions",
        ),
        (
            df_interactions_inside_impressions,
            "series_id",
            "popularity_interactions_series_inside_impressions",
            "Rank of series",
            r"\# of interactions",
        ),
    ]

    cases = cases_users + cases_items + cases_series
    list_basic_statistics = []
    results = {}
    for df, col, name, x_label, x_label_pdf in cases:
        df_pop, df_pop_perc = compute_popularity(
            df=df,
            column=col,
        )

        arr_popularity = df_pop["count"].to_numpy()
        results_basic_statistics = _compute_basic_statistics(
            arr_data=arr_popularity,
            data_discrete=True,
            name=name,
        )
        list_basic_statistics.append(results_basic_statistics)

        results[name] = df_pop

        plot_popularity(
            df=df_pop,
            dir_results=dir_results,
            x_data="index",
            y_data="count",
            x_label=x_label,
            y_label=r"\# of interactions",
            name=name,
            x_scale="linear",
            y_scale="linear",
        )

        plot_cdf_pdf_ccdf(
            df=df_pop,
            dir_results=dir_results,
            x_data="count",
            x_label=x_label_pdf,
            name=name,
        )

    df_results = pd.DataFrame.from_records(data=list_basic_statistics)
    df_results.to_csv(
        path_or_buf=os.path.join(
            dir_results, "table_statistics_popularity_interactions.csv"
        ),
        index=True,
        header=True,
        sep=";",
        encoding="utf-8",
        decimal=",",
        float_format="%.4f",
    )

    return {"table_statistics_popularity_interactions": df_results, **results}


def compute_temporal_distribution_of_interactions(
    dir_results: str,
    dict_results: dict[str, pd.DataFrame],
    df_interactions_with_impressions_all: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if all(
        filename in dict_results
        for filename in [
            "num_interactions_by_date_all",
            "num_interactions_by_hours_all",
            "num_interactions_by_minute_all",
            "num_interactions_by_days_in_month_all",
            "num_interactions_by_day_of_year_all",
            "num_interactions_by_day_of_week_all",
            "num_interactions_by_month_all",
        ]
    ):
        logger.warning(
            "Skipping function %(function)s because all keys in the dictionary already exist.",
            {"function": compute_temporal_distribution_of_interactions.__name__},
        )
        return {}

    cases = [
        (
            df_interactions_with_impressions_all,
            "date",
            "date",
            r"Date",
            True,
            "num_interactions_by_date_all",
        ),
        (
            df_interactions_with_impressions_all,
            "hour",
            "hour",
            r"Hour",
            False,
            "num_interactions_by_hours_all",
        ),
        (
            df_interactions_with_impressions_all,
            "minute",
            "minute",
            r"Minutes",
            False,
            "num_interactions_by_minute_all",
        ),
        (
            df_interactions_with_impressions_all,
            "days_in_month",
            "days_in_month",
            r"Day in month",
            False,
            "num_interactions_by_days_in_month_all",
        ),
        (
            df_interactions_with_impressions_all,
            "day_of_year",
            "day_of_year",
            r"Day of year",
            False,
            "num_interactions_by_day_of_year_all",
        ),
        (
            df_interactions_with_impressions_all,
            "day_of_week",
            "day_name",
            r"Day of week",
            False,
            "num_interactions_by_day_of_week_all",
        ),
        (
            df_interactions_with_impressions_all,
            "month",
            "month_name",
            r"Month",
            False,
            "num_interactions_by_month_all",
        ),
    ]

    results = {}

    list_basic_statistics = []

    for df, col, x_data, x_label, x_date, name in cases:
        df_dates = (
            df.groupby(
                by=[col],
                as_index=False,
            )["series_id"]
            .agg(["count"])
            .sort_values(
                by=col,
                ascending=True,
                ignore_index=True,
                inplace=False,
            )
        )

        if col == "day_of_week":
            df_dates[x_data] = (
                df_dates[col]
                .apply(
                    lambda df_row: DICT_DAY_OF_WEEK_TO_STR[df_row],
                    convert_dtype=True,
                )
                .astype(pd.StringDtype())
            )
        if col == "month":
            df_dates[x_data] = (
                df_dates[col]
                .apply(
                    lambda df_row: DICT_MONTH_TO_STR[df_row],
                    convert_dtype=True,
                )
                .astype(pd.StringDtype())
            )

        results[name] = df_dates

        if col in {"month", "hour", "minute"}:
            logger.debug(f"Processing column {col}")

            arr_data = df_dates[col].to_numpy()

            dict_statistics = _compute_basic_statistics(
                arr_data=arr_data,
                data_discrete=True,
                name=name,
            )
            list_basic_statistics.append(dict_statistics)

        else:
            arr_data = (
                pd.to_datetime(df_dates[col])
                .dt.strftime(date_format="%Y-%m-%d")
                .to_numpy()
            )

        plot_dates(
            dir_results=dir_results,
            df=df_dates,
            name=name,
            x_data=x_data,
            y_data="count",
            x_label=x_label,
            y_label=r"\# of interactions",
            x_date=x_date,
            y_date=False,
        )

    df_results = pd.DataFrame.from_records(data=list_basic_statistics)
    df_results.to_csv(
        path_or_buf=os.path.join(
            dir_results, "table_statistics_temporal_distribution_interactions.csv"
        ),
        index=True,
        header=True,
        sep=";",
        encoding="utf-8",
        decimal=",",
        float_format="%.4f",
    )

    return {
        "table_statistics_temporal_distribution_interactions": df_results,
        **results,
    }


def compute_popularity_impressions_contextual(
    *,
    dir_results: str,
    dict_results: dict[str, pd.DataFrame],
    df_interactions_inside_impressions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if all(
        filename in dict_results
        for filename in [
            "table_popularity_impressions_contextual",
            "popularity_impressions_contextual_user_with_interactions",
            "popularity_impressions_contextual_series_with_interactions",
            "popularity_impressions_contextual_series",
        ]
    ):
        logger.warning(
            "Skipping function %(function)s because all keys in the dictionary already exist.",
            {"function": compute_popularity_impressions_contextual.__name__},
        )
        return {}

    df_impressions_on_interactions = df_interactions_inside_impressions[
        ["user_id", "recommended_series_list"]
    ].explode(
        column="recommended_series_list",
        ignore_index=True,
    )

    cases_user = [
        (
            df_impressions_on_interactions,
            "user_id",
            "Rank of users",
            "popularity_impressions_contextual_user_with_interactions",
        ),
    ]
    cases_series = [
        (
            df_impressions_on_interactions,
            "recommended_series_list",
            "Rank of series",
            "popularity_impressions_contextual_series_with_interactions",
        ),
    ]

    cases = cases_user + cases_series

    list_basic_statistics = []
    results = {}

    for df, col, x_label, name in cases:
        df_pop, df_pop_perc = compute_popularity(
            df=df,
            column=col,
        )

        arr_popularity = df_pop["count"].to_numpy()
        results_basic_statistics = _compute_basic_statistics(
            arr_data=arr_popularity,
            data_discrete=True,
            name=name,
        )
        list_basic_statistics.append(results_basic_statistics)

        results[name] = df_pop

        plot_popularity(
            df=df_pop,
            dir_results=dir_results,
            x_data="index",
            y_data="count",
            x_label=x_label,
            y_label=r"\# of impressions",
            name=name,
            x_scale="linear",
            y_scale="linear",
        )

        plot_cdf_pdf_ccdf(
            df=df_pop,
            dir_results=dir_results,
            x_data="count",
            x_label=r"\# of impressions",
            name=name,
        )

    df_results = pd.DataFrame.from_records(data=list_basic_statistics)
    df_results.to_csv(
        path_or_buf=os.path.join(
            dir_results, "table_statistics_popularity_impressions_contextual.csv"
        ),
        index=True,
        header=True,
        sep=";",
        encoding="utf-8",
        decimal=",",
        float_format="%.4f",
    )

    return {"table_popularity_impressions_contextual": df_results, **results}


def compute_temporal_distribution_of_impressions_contextual(
    *,
    dir_results: str,
    dict_results: dict[str, pd.DataFrame],
    df_interactions_inside_impressions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if all(
        filename in dict_results
        for filename in [
            "num_impressions_contextual_by_date_with_interactions",
            "num_impressions_contextual_by_hours_with_interactions",
            "num_impressions_contextual_by_minute_with_interactions",
            "num_impressions_contextual_by_days_in_month_with_interactions",
            "num_impressions_contextual_by_day_of_year_with_interactions",
            "num_impressions_contextual_by_day_of_week_with_interactions",
            "num_impressions_contextual_by_month_with_interactions",
        ]
    ):
        logger.warning(
            "Skipping function %(function)s because all keys in the dictionary already exist.",
            {
                "function": compute_temporal_distribution_of_impressions_contextual.__name__
            },
        )
        return {}

    df_impressions_contextual_in_interactions = df_interactions_inside_impressions[
        [
            "date",
            "hour",
            "minute",
            "days_in_month",
            "day_of_year",
            "day_of_week",
            "day_name",
            "month",
            "month_name",
            "recommended_series_list",
        ]
    ].explode(
        column="recommended_series_list",
        ignore_index=True,
    )

    cases = [
        (
            df_impressions_contextual_in_interactions,
            "date",
            "date",
            r"Date",
            True,
            "num_impressions_contextual_by_date_with_interactions",
        ),
        (
            df_impressions_contextual_in_interactions,
            "hour",
            "hour",
            r"Hour",
            False,
            "num_impressions_contextual_by_hours_with_interactions",
        ),
        (
            df_impressions_contextual_in_interactions,
            "minute",
            "minute",
            r"Minutes",
            False,
            "num_impressions_contextual_by_minute_with_interactions",
        ),
        (
            df_impressions_contextual_in_interactions,
            "days_in_month",
            "days_in_month",
            r"Day in month",
            False,
            "num_impressions_contextual_by_days_in_month_with_interactions",
        ),
        (
            df_impressions_contextual_in_interactions,
            "day_of_year",
            "day_of_year",
            r"Day of year",
            False,
            "num_impressions_contextual_by_day_of_year_with_interactions",
        ),
        (
            df_impressions_contextual_in_interactions,
            "day_of_week",
            "day_name",
            r"Day of week",
            False,
            "num_impressions_contextual_by_day_of_week_with_interactions",
        ),
        (
            df_impressions_contextual_in_interactions,
            "month",
            "month_name",
            r"Month",
            False,
            "num_impressions_contextual_by_month_with_interactions",
        ),
    ]

    results = {}

    list_basic_statistics = []

    for df, col, x_data, x_label, x_date, name in cases:
        df_dates = (
            df.groupby(
                by=[col],
                as_index=False,
            )["recommended_series_list"]
            .agg(["count"])
            .sort_values(
                by=col,
                ascending=True,
                ignore_index=True,
                inplace=False,
            )
        )

        if col == "day_of_week":
            df_dates[x_data] = (
                df_dates[col]
                .apply(
                    lambda df_row: DICT_DAY_OF_WEEK_TO_STR[df_row],
                    convert_dtype=True,
                )
                .astype(pd.StringDtype())
            )
        if col == "month":
            df_dates[x_data] = (
                df_dates[col]
                .apply(
                    lambda df_row: DICT_MONTH_TO_STR[df_row],
                    convert_dtype=True,
                )
                .astype(pd.StringDtype())
            )

        results[name] = df_dates

        if col in {"month", "hour", "minute"}:
            logger.debug(f"Processing column {col}")

            arr_data = df_dates[col].to_numpy()

            dict_statistics = _compute_basic_statistics(
                arr_data=arr_data,
                data_discrete=True,
                name=name,
            )
            list_basic_statistics.append(dict_statistics)

        else:
            arr_data = (
                pd.to_datetime(df_dates[col])
                .dt.strftime(date_format="%Y-%m-%d")
                .to_numpy()
            )

        plot_dates(
            dir_results=dir_results,
            df=df_dates,
            name=name,
            x_data=x_data,
            y_data="count",
            x_label=x_label,
            y_label=r"\# of impressions",
            x_date=x_date,
            y_date=False,
        )

    df_results = pd.DataFrame.from_records(data=list_basic_statistics)
    df_results.to_csv(
        path_or_buf=os.path.join(
            dir_results,
            "table_statistics_temporal_distribution_impressions_contextual.csv",
        ),
        index=True,
        header=True,
        sep=";",
        encoding="utf-8",
        decimal=",",
        float_format="%.4f",
    )

    return {
        "table_statistics_temporal_distribution_impressions_contextual": df_results,
        **results,
    }


def compute_popularity_impressions_global(
    dir_results: str,
    dict_results: dict[str, pd.DataFrame],
    df_impressions_global: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if all(
        filename in dict_results
        for filename in [
            "popularity_impressions_global_user",
            "popularity_impressions_global_series",
        ]
    ):
        logger.warning(
            "Skipping function %(function)s because all keys in the dictionary already exist.",
            {"function": compute_popularity_impressions_global.__name__},
        )
        return {}

    df_impressions_global = df_impressions_global.reset_index(
        drop=False,
        inplace=False,
    )[["user_id", "recommended_series_list"]].explode(
        column="recommended_series_list",
        ignore_index=True,
    )

    cases_user = [
        (
            df_impressions_global,
            "user_id",
            "Rank of users",
            "popularity_impressions_global_user",
        ),
    ]
    cases_series = [
        (
            df_impressions_global,
            "recommended_series_list",
            "Rank of series",
            "popularity_impressions_global_series",
        ),
    ]

    cases = cases_user + cases_series

    list_basic_statistics = []
    results = {}

    for df, col, x_label, name in cases:
        df_pop, df_pop_perc = compute_popularity(
            df=df,
            column=col,
        )

        arr_popularity = df_pop["count"].to_numpy()
        results_basic_statistics = _compute_basic_statistics(
            arr_data=arr_popularity,
            data_discrete=True,
            name=name,
        )
        list_basic_statistics.append(results_basic_statistics)

        results[name] = df_pop

        plot_popularity(
            df=df_pop,
            dir_results=dir_results,
            x_data="index",
            y_data="count",
            x_label=x_label,
            y_label=r"\# of global impressions",
            name=name,
            x_scale="linear",
            y_scale="linear",
        )

        plot_cdf_pdf_ccdf(
            df=df_pop,
            dir_results=dir_results,
            x_data="count",
            x_label=r"\# of global impressions",
            name=name,
        )

    df_results = pd.DataFrame.from_records(data=list_basic_statistics)
    df_results.to_csv(
        path_or_buf=os.path.join(
            dir_results, "table_statistics_popularity_impressions_global.csv"
        ),
        index=True,
        header=True,
        sep=";",
        encoding="utf-8",
        decimal=",",
        float_format="%.4f",
    )

    return {"table_popularity_impressions_global": df_results, **results}


def compute_correlation_interactions_impressions(
    dir_results: str,
    dict_results: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    if all(
        filename in dict_results
        for filename in [
            #
            "corr_pearson_popularity_correlation_interactions_impressions_user_all",
            "corr_kendall_popularity_correlation_interactions_impressions_user_all",
            "corr_spearman_popularity_correlation_interactions_impressions_user_all",
            #
            "corr_pearson_popularity_correlation_interactions_impressions_user_inside_impressions",
            "corr_kendall_popularity_correlation_interactions_impressions_user_inside_impressions",
            "corr_spearman_popularity_correlation_interactions_impressions_user_inside_impressions",
            #
            "corr_pearson_popularity_correlation_interactions_impressions_series_all",
            "corr_kendall_popularity_correlation_interactions_impressions_series_all",
            "corr_spearman_popularity_correlation_interactions_impressions_series_all",
            #
            "corr_pearson_popularity_correlation_interactions_impressions_series_inside_impressions",
            "corr_kendall_popularity_correlation_interactions_impressions_series_inside_impressions",
            "corr_spearman_popularity_correlation_interactions_impressions_series_inside_impressions",
        ]
    ):
        logger.warning(
            "Skipping function %(function)s because all keys in the dictionary already exist.",
            {"function": compute_correlation_interactions_impressions.__name__},
        )
        return {}

    assert "popularity_interactions_user_all" in dict_results
    assert "popularity_interactions_series_all" in dict_results

    assert "popularity_interactions_user_inside_impressions" in dict_results
    assert "popularity_interactions_series_inside_impressions" in dict_results

    assert "popularity_impressions_contextual_user_with_interactions" in dict_results
    assert "popularity_impressions_contextual_series_with_interactions" in dict_results

    assert "popularity_impressions_global_user" in dict_results
    assert "popularity_impressions_global_series" in dict_results

    cases_user = [
        (
            dict_results["popularity_interactions_user_all"],
            dict_results["popularity_impressions_contextual_user_with_interactions"],
            dict_results["popularity_impressions_global_user"],
            "user_id",
            "user_id",
            "Users - all",
            "popularity_correlation_interactions_impressions_user_all",
        ),
        (
            dict_results["popularity_interactions_user_inside_impressions"],
            dict_results["popularity_impressions_contextual_user_with_interactions"],
            dict_results["popularity_impressions_global_user"],
            "user_id",
            "user_id",
            "Users - inside contextual impressions",
            "popularity_correlation_interactions_impressions_user_inside_impressions",
        ),
    ]

    cases_series = [
        (
            dict_results["popularity_interactions_series_all"],
            dict_results["popularity_impressions_contextual_series_with_interactions"],
            dict_results["popularity_impressions_global_series"],
            "series_id",
            "recommended_series_list",
            "Series - all",
            "popularity_correlation_interactions_impressions_series_all",
        ),
        (
            dict_results["popularity_interactions_series_inside_impressions"],
            dict_results["popularity_impressions_contextual_series_with_interactions"],
            dict_results["popularity_impressions_global_series"],
            "series_id",
            "recommended_series_list",
            "Series - inside contextual impressions",
            "popularity_correlation_interactions_impressions_series_inside_impressions",
        ),
    ]

    cases = cases_user + cases_series

    results = {}
    list_dfs = []

    for df_int, df_imp_cont, df_imp_global, col_int, col_imp, attr, name in cases:
        df_int = df_int.set_index(col_int).rename(
            columns={"count": "count_interactions"}
        )[["count_interactions"]]

        df_imp_cont = df_imp_cont.set_index(col_imp).rename(
            columns={"count": "count_impressions_contextual"}
        )[["count_impressions_contextual"]]

        df_imp_global = df_imp_global.set_index(col_imp).rename(
            columns={"count": "count_impressions_global"}
        )[["count_impressions_global"]]

        df_pop = (
            df_int.merge(
                right=df_imp_cont,
                left_index=True,
                right_index=True,
                how="outer",
                suffixes=("", ""),
            )
            .merge(
                right=df_imp_global,
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
        )[
            [
                "count_interactions",
                "count_impressions_contextual",
                "count_impressions_global",
            ]
        ]

        df_pop_corr_pearson = (
            df_pop.corr(method="pearson")
            .assign(attr=attr, corr_method="pearson")
            .astype({"attr": pd.StringDtype(), "corr_method": pd.StringDtype()})
        )
        df_pop_corr_kendall = (
            df_pop.corr(method="kendall")
            .assign(attr=attr, corr_method="kendall")
            .astype({"attr": pd.StringDtype(), "corr_method": pd.StringDtype()})
        )
        df_pop_corr_spearman = (
            df_pop.corr(method="spearman")
            .assign(attr=attr, corr_method="spearman")
            .astype({"attr": pd.StringDtype(), "corr_method": pd.StringDtype()})
        )

        results[f"corr_pearson_{name}"] = df_pop_corr_pearson
        results[f"corr_kendall_{name}"] = df_pop_corr_kendall
        results[f"corr_spearman_{name}"] = df_pop_corr_spearman

        list_dfs.append(df_pop_corr_pearson)
        list_dfs.append(df_pop_corr_kendall)
        list_dfs.append(df_pop_corr_spearman)

    df_results = pd.concat(
        objs=list_dfs,
        axis=0,
        ignore_index=True,
    )
    df_results.to_csv(
        path_or_buf=os.path.join(
            dir_results,
            "table_corr_popularity_interactions_impressions.csv",
        ),
        index=True,
        header=True,
        sep=";",
        encoding="utf-8",
        decimal=",",
        float_format="%.4f",
    )

    # df_series_pop_corr_pearson.to_csv(
    #     path_or_buf=os.path.join(dir_results, f"table-corr_pearson-{name}.csv"),
    #     sep=";",
    #     float_format="%.4f",
    #     header=True,
    #     index=True,
    #     encoding="utf-8",
    #     decimal=",",
    # )
    #
    # df_series_pop_corr_kendall.to_csv(
    #     path_or_buf=os.path.join(dir_results, f"table-corr_kendall-{name}.csv"),
    #     sep=";",
    #     float_format="%.4f",
    #     header=True,
    #     index=True,
    #     encoding="utf-8",
    #     decimal=",",
    # )
    #
    # df_series_pop_corr_spearman.to_csv(
    #     path_or_buf=os.path.join(dir_results, f"table-corr_spearman-{name}.csv"),
    #     sep=";",
    #     float_format="%.4f",
    #     header=True,
    #     index=True,
    #     encoding="utf-8",
    #     decimal=",",
    # )

    # df_series_pop_interactions_all = (
    #     dict_results["series_pop_interactions_all"]
    #     .set_index("series_id")
    #     .rename(
    #         columns={
    #             "count": "count_interactions",
    #             "index": "index_interactions",
    #         }
    #     )
    # )
    # df_series_pop_impressions_contextual = (
    #     dict_results["series_pop_impressions_contextual"]
    #     .set_index("recommended_series_list")
    #     .rename(
    #         columns={
    #             "count": "count_impressions_contextual",
    #             "index": "index_impressions_contextual",
    #         }
    #     )
    # )
    # df_series_pop_impressions_global = (
    #     dict_results["series_pop_impressions_global"]
    #     .set_index("recommended_series_list")
    #     .rename(
    #         columns={
    #             "count": "count_impressions_global",
    #             "index": "index_impressions_global",
    #         }
    #     )
    # )

    # df_series_pop = (
    #     df_series_pop_interactions_all.merge(
    #         right=df_series_pop_impressions_contextual,
    #         left_index=True,
    #         right_index=True,
    #         how="outer",
    #         suffixes=("", ""),
    #     )
    #     .merge(
    #         right=df_series_pop_impressions_global,
    #         left_index=True,
    #         right_index=True,
    #         how="outer",
    #         suffixes=("", ""),
    #     )
    #     .fillna(
    #         {
    #             "count_interactions": 0,
    #             "count_impressions_contextual": 0,
    #             "count_impressions_global": 0,
    #         }
    #     )
    #     .astype(
    #         {
    #             "count_interactions": np.int32,
    #             "count_impressions_contextual": np.int32,
    #             "count_impressions_global": np.int32,
    #         }
    #     )
    # )
    #
    # df_series_pop_corr_pearson = df_series_pop[
    #     [
    #         "count_interactions",
    #         "count_impressions_contextual",
    #         "count_impressions_global",
    #     ]
    # ].corr(
    #     method="pearson",
    # )
    # df_series_pop_corr_kendall = df_series_pop[
    #     [
    #         "count_interactions",
    #         "count_impressions_contextual",
    #         "count_impressions_global",
    #     ]
    # ].corr(
    #     method="kendall",
    # )
    #
    # df_series_pop_corr_spearman = df_series_pop[
    #     [
    #         "count_interactions",
    #         "count_impressions_contextual",
    #         "count_impressions_global",
    #     ]
    # ].corr(
    #     method="spearman",
    # )
    #
    # df_series_pop_corr_pearson.to_csv(
    #     path_or_buf=os.path.join(dir_results, "table-series_pop_corr_pearson.csv"),
    #     sep=";",
    #     float_format="%.4f",
    #     header=True,
    #     index=True,
    #     encoding="utf-8",
    #     decimal=",",
    # )
    #
    # df_series_pop_corr_kendall.to_csv(
    #     path_or_buf=os.path.join(dir_results, "table-series_pop_corr_kendall.csv"),
    #     sep=";",
    #     float_format="%.4f",
    #     header=True,
    #     index=True,
    #     encoding="utf-8",
    #     decimal=",",
    # )
    #
    # df_series_pop_corr_spearman.to_csv(
    #     path_or_buf=os.path.join(dir_results, "table-series_pop_corr_spearman.csv"),
    #     sep=";",
    #     float_format="%.4f",
    #     header=True,
    #     index=True,
    #     encoding="utf-8",
    #     decimal=",",
    # )

    return {"table_corr_popularity_interactions_impressions": df_results, **results}


def compute_position_impressions(
    dir_results: str,
    dict_results: dict[str, pd.DataFrame],
    df_interactions_with_impressions_all: pd.DataFrame,
    df_impressions_global: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if all(
        filename in dict_results
        for filename in [
            "table_position_impressions",
            "position_impressions",
        ]
    ):
        return {}

    # No need to compute on `df_interactions_all` because we're merging with interactions on impressions anyway.
    df_interactions_with_impressions_all = (
        df_interactions_with_impressions_all.dropna(inplace=False, ignore_index=True)
        .rename(
            columns={
                "row_position": "position_impression_on_screen",
                "recommendation_list_length": "num_items_in_impression",
            }
        )
        .astype(
            dtype={
                "position_impression_on_screen": np.int32,
                "num_items_in_impression": np.int32,
            }
        )[["position_impression_on_screen", "num_items_in_impression"]]
    )

    df_impressions_global = (
        df_impressions_global.dropna(inplace=False, ignore_index=True)
        .rename(
            columns={
                "row_position": "position_impression_on_screen",
                "recommendation_list_length": "num_items_in_impression",
            }
        )
        .astype(
            dtype={
                "position_impression_on_screen": np.int32,
                "num_items_in_impression": np.int32,
            }
        )[["position_impression_on_screen", "num_items_in_impression"]]
    )

    df_impressions_all = pd.concat(
        objs=[df_interactions_with_impressions_all, df_impressions_global],
        axis="index",
        ignore_index=True,
    )

    cases = [
        (
            df_impressions_all,
            "position_impression_on_screen",
            "Position on screen",
            "position_impression_on_screen_all",
        ),
        (
            df_impressions_all,
            "num_items_in_impression",
            "Items in impressions",
            "num_items_in_impression_all",
        ),
        (
            df_interactions_with_impressions_all,
            "position_impression_on_screen",
            "Position on screen",
            "position_impression_on_screen_contextual",
        ),
        (
            df_interactions_with_impressions_all,
            "num_items_in_impression",
            "Items in impressions",
            "num_items_in_impression_contextual",
        ),
        (
            df_impressions_global,
            "position_impression_on_screen",
            "Position on screen",
            "position_impression_on_screen_global",
        ),
        (
            df_impressions_global,
            "num_items_in_impression",
            "Items in impressions",
            "num_items_in_impression_global",
        ),
    ]
    list_basic_statistics = []

    for df, col, label, name in cases:
        arr_position = df[col].to_numpy()
        results_basic_statistics = _compute_basic_statistics(
            arr_data=arr_position,
            data_discrete=True,
            name=name,
        )
        list_basic_statistics.append(results_basic_statistics)

        df_pop, df_pop_perc = compute_popularity(
            df=df,
            column=col,
        )
        # Here we're not interested in ranking but understanding the frequency of values, hence, we sort by values.
        df_pop = df_pop.sort_values(
            by=col, ascending=True, inplace=False, ignore_index=True
        ).assign(cumsum=lambda df_: df_["count"].cumsum())

        plot_popularity(
            df=df_pop,
            dir_results=dir_results,
            x_data=col,
            y_data="count",
            x_label=label,
            y_label="Count",
            name=name,
            x_scale="linear",
            y_scale="log",
        )

        plot_barplot(
            df=df_pop,
            dir_results=dir_results,
            x_data=col,
            y_data="count",
            x_label=label,
            y_label="Count",
            ticks_labels=None,
            name=name,
            log=True,
        )

    df_results = pd.DataFrame.from_records(data=list_basic_statistics)
    df_results.to_csv(
        path_or_buf=os.path.join(
            dir_results, "table_statistics_position_impressions.csv"
        ),
        index=True,
        header=True,
        sep=";",
        encoding="utf-8",
        decimal=",",
        float_format="%.4f",
    )

    return {
        "table_position_impressions": df_results,
        "position_impressions_all": df_impressions_all,
        "position_impressions_global": df_impressions_global,
        "position_impressions_contextual": df_interactions_with_impressions_all,
    }


def compute_position_interactions(
    dir_results: str,
    dict_results: dict[str, pd.DataFrame],
    df_interactions_inside_impressions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if all(
        filename in dict_results
        for filename in [
            "table_position_interactions",
            "position_interactions",
        ]
    ):
        return {}

    # No need to compute on `df_interactions_all` because we're merging with interactions on impressions anyway.
    df_interactions_inside_impressions = (
        df_interactions_inside_impressions.dropna(
            inplace=False,
            ignore_index=True,
        )
        .rename(
            columns={
                "row_position": "position_interaction_on_screen",
                "recommendation_list_length": "num_items_in_impression",
            }
        )
        .assign(
            position_interaction_on_impression=lambda df_: df_.apply(
                compute_position_of_interaction_on_impression,
                axis="columns",
                raw=False,
                result_type="expand",
            )
        )
        .astype(
            dtype={
                "position_interaction_on_impression": np.int32,
                "position_interaction_on_screen": np.int32,
                "num_items_in_impression": np.int32,
            }
        )[
            [
                "user_id",
                "series_id",
                "position_interaction_on_impression",
                "position_interaction_on_screen",
                "num_items_in_impression",
            ]
        ]
    )

    cases = [
        ("position_interaction_on_impression", "Position in impressions"),
        ("position_interaction_on_screen", "Position on screen"),
        ("num_items_in_impression", "Items in impressions"),
    ]
    list_basic_statistics = []

    for col, label in cases:
        arr_position = df_interactions_inside_impressions[col].to_numpy()
        results_basic_statistics = _compute_basic_statistics(
            arr_data=arr_position,
            data_discrete=True,
            name=col,
        )
        list_basic_statistics.append(results_basic_statistics)

        df_pop, df_pop_perc = compute_popularity(
            df=df_interactions_inside_impressions,
            column=col,
        )
        # Here we're not interested in ranking but understanding the frequency of values, hence, we sort by values.
        df_pop = df_pop.sort_values(
            by=col, ascending=True, inplace=False, ignore_index=True
        ).assign(cumsum=lambda df_: df_["count"].cumsum())

        plot_popularity(
            df=df_pop,
            dir_results=dir_results,
            x_data=col,
            y_data="count",
            x_label=label,
            y_label="Count",
            name=col,
            x_scale="linear",
            y_scale="log",
        )

        plot_barplot(
            df=df_pop,
            dir_results=dir_results,
            x_data=col,
            y_data="count",
            x_label=label,
            y_label="Count",
            ticks_labels=None,
            name=col,
            log=True,
        )

    df_results = pd.DataFrame.from_records(data=list_basic_statistics)
    df_results.to_csv(
        path_or_buf=os.path.join(
            dir_results, "table_statistics_position_interactions.csv"
        ),
        index=True,
        header=True,
        sep=";",
        encoding="utf-8",
        decimal=",",
        float_format="%.4f",
    )

    return {
        "table_position_interactions": df_results,
        "position_interactions": df_interactions_inside_impressions,
    }


def compute_vision_factor(
    dir_results: str,
    dict_results: dict[str, pd.DataFrame],
    df_interactions_with_impressions_all: pd.DataFrame,
    df_interactions_outside_impressions: pd.DataFrame,
    df_interactions_inside_impressions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if all(
        filename in dict_results
        for filename in [
            "table_vision_factor",
            "vision_factor_all",
            "vision_factor_outside_impressions",
            "vision_factor_only_non_null_impressions",
        ]
    ):
        return {}

    df_interactions_with_impressions_all = df_interactions_with_impressions_all[
        df_interactions_with_impressions_all["interaction_type"]
        == InteractionType.View.value
    ]
    df_interactions_outside_impressions = df_interactions_outside_impressions[
        df_interactions_outside_impressions["interaction_type"]
        == InteractionType.View.value
    ]
    df_interactions_inside_impressions = df_interactions_inside_impressions[
        df_interactions_inside_impressions["interaction_type"]
        == InteractionType.View.value
    ]

    list_basic_statistics = []

    for df, name in [
        (
            df_interactions_with_impressions_all,
            "vision_factor_all",
        ),
        (
            df_interactions_outside_impressions,
            "vision_factor_outside_impressions",
        ),
        (
            df_interactions_inside_impressions,
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
            df_interactions_with_impressions_all["vision_factor"].to_numpy(),
            df_interactions_outside_impressions["vision_factor"].to_numpy(),
            df_interactions_inside_impressions["vision_factor"].to_numpy(),
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
            df_interactions_with_impressions_all["vision_factor"].to_numpy(),
            df_interactions_outside_impressions["vision_factor"].to_numpy(),
            df_interactions_inside_impressions["vision_factor"].to_numpy(),
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
        "vision_factor_all": df_interactions_with_impressions_all[
            "vision_factor"
        ].to_frame(),
        "vision_factor_outside_impressions": df_interactions_outside_impressions[
            "vision_factor"
        ].to_frame(),
        "vision_factor_only_non_null_impressions": df_interactions_inside_impressions[
            "vision_factor"
        ].to_frame(),
    }


def compute_ratings(
    dir_results: str,
    dict_results: dict[str, pd.DataFrame],
    df_interactions_with_impressions_all: pd.DataFrame,
    df_interactions_outside_impressions: pd.DataFrame,
    df_interactions_inside_impressions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if all(
        filename in dict_results
        for filename in [
            "table_vision_factor",
            "explicit_rating_all",
            "explicit_rating_outside_impressions",
            "explicit_rating_only_non_null_impressions",
        ]
    ):
        return {}

    df_interactions_with_impressions_all = df_interactions_with_impressions_all[
        df_interactions_with_impressions_all["interaction_type"]
        == InteractionType.Rate.value
    ]

    df_interactions_outside_impressions = df_interactions_outside_impressions[
        df_interactions_outside_impressions["interaction_type"]
        == InteractionType.Rate.value
    ]

    df_interactions_inside_impressions = df_interactions_inside_impressions[
        df_interactions_inside_impressions["interaction_type"]
        == InteractionType.Rate.value
    ]

    list_basic_statistics = []
    dfs = []
    for df, name, label in [
        (
            df_interactions_with_impressions_all,
            "explicit_ratings_all",
            "All interactions",
        ),
        (
            df_interactions_outside_impressions,
            "explicit_ratings_outside_impressions",
            "Outside\nimpressions",
        ),
        (
            df_interactions_inside_impressions,
            "explicit_ratings_only_non_null_impressions",
            "Inside\nimpressions",
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
        dfs.append((df_pop, label))

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
        "explicit_rating_all": df_interactions_with_impressions_all,
        "explicit_rating_outside_impressions": df_interactions_outside_impressions,
        "explicit_rating_only_non_null_impressions": df_interactions_inside_impressions,
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
        df["minute"] = df["datetime"].dt.minute

        df["days_in_month"] = df["datetime"].dt.days_in_month
        df["day_of_year"] = df["datetime"].dt.day_of_year

        df["month"] = df["datetime"].dt.month
        df["month_name"] = df["datetime"].dt.month_name()

        df["day_of_week"] = df["datetime"].dt.day_of_week
        df["day_name"] = df["datetime"].dt.day_name()

        new_dfs.append(df)

    return new_dfs


def transform_dataframes_for_ctr_computation(
    *,
    df_interactions_all: pd.DataFrame,
    df_interactions_outside_impressions: pd.DataFrame,
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
    # In this case, the only dataframe we must use is `df_interactions_outside_impressions`.
    # If not, we're counting impressions twice (one on the contextual impression and one on the interaction)
    df_impressions_contextual_all = pd.concat(
        objs=[
            df_interactions_outside_impressions,
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

    df_interactions: pd.DataFrame = cw_impressions_raw_data.interactions.compute()
    df_impressions_contextual = cw_impressions_raw_data.impressions.compute()
    df_impressions_global = (
        cw_impressions_raw_data.impressions_non_direct_link.compute()
    )

    metadata = compute_metadata_dataset(
        metadata=cw_impressions_raw_data.metadata,
        df_interactions=df_interactions,
        df_impressions_contextual=df_impressions_contextual,
        df_impressions_global=df_impressions_global,
    )

    df_interactions_with_impressions_all = df_interactions.merge(
        right=df_impressions_contextual,
        left_on="recommendation_id",
        left_index=False,
        right_on=None,
        right_index=True,
        how="left",
        suffixes=("", ""),
    )

    df_interactions_outside_impressions: pd.DataFrame = (
        df_interactions_with_impressions_all[
            df_interactions_with_impressions_all["recommendation_id"] == -1
        ]
    )
    df_interactions_inside_impressions: pd.DataFrame = (
        df_interactions_with_impressions_all[
            df_interactions_with_impressions_all["recommendation_id"] >= 0
        ]
    )

    (
        df_interactions_with_impressions_all,
        df_interactions_outside_impressions,
        df_interactions_inside_impressions,
    ) = transform_dataframes_add_date_time_hour_to_interactions(
        dfs=[
            df_interactions_with_impressions_all,
            df_interactions_outside_impressions,
            df_interactions_inside_impressions,
        ]
    )

    # results = compute_number_unique_items_by_item_type(
    #     dir_results=dir_results,
    #     dict_results=dict_results,
    #     df_interactions_with_impressions_all=df_interactions_with_impressions_all,
    #     df_interactions_outside_impressions=df_interactions_outside_impressions,
    #     df_interactions_inside_impressions=df_interactions_inside_impressions,
    #     metadata=metadata,
    # )
    # dict_results.update(results)

    # results = compute_number_unique_items_by_interaction_type(
    #     dir_results=dir_results,
    #     dict_results=dict_results,
    #     df_interactions_with_impressions_all=df_interactions_with_impressions_all,
    #     df_interactions_outside_impressions=df_interactions_outside_impressions,
    #     df_interactions_inside_impressions=df_interactions_inside_impressions,
    #     metadata=metadata,
    # )
    # dict_results.update(results)

    # results = compute_number_interactions_by_item_type(
    #     dir_results=dir_results,
    #     dict_results=dict_results,
    #     df_interactions_with_impressions_all=df_interactions_with_impressions_all,
    #     df_interactions_outside_impressions=df_interactions_outside_impressions,
    #     df_interactions_inside_impressions=df_interactions_inside_impressions,
    # )
    # dict_results.update(results)
    #
    # results = compute_number_of_interactions_by_interaction_type(
    #     dir_results=dir_results,
    #     dict_results=dict_results,
    #     df_interactions_with_impressions_all=df_interactions_with_impressions_all,
    #     df_interactions_outside_impressions=df_interactions_outside_impressions,
    #     df_interactions_inside_impressions=df_interactions_inside_impressions,
    # )
    # dict_results.update(results)

    results = compute_popularity_interactions(
        dir_results=dir_results,
        dict_results=dict_results,
        df_interactions_with_impressions_all=df_interactions_with_impressions_all,
        df_interactions_outside_impressions=df_interactions_outside_impressions,
        df_interactions_inside_impressions=df_interactions_inside_impressions,
    )
    dict_results.update(results)

    # results = compute_temporal_distribution_of_interactions(
    #     dir_results=dir_results,
    #     dict_results=dict_results,
    #     df_interactions_with_impressions_all=df_interactions_with_impressions_all,
    # )
    # dict_results.update(results)

    results = compute_popularity_impressions_contextual(
        dir_results=dir_results,
        dict_results=dict_results,
        df_interactions_inside_impressions=df_interactions_inside_impressions,
    )
    dict_results.update(results)

    # results = compute_temporal_distribution_of_impressions_contextual(
    #     dir_results=dir_results,
    #     dict_results=dict_results,
    #     df_interactions_inside_impressions=df_interactions_inside_impressions,
    # )
    # dict_results.update(results)

    results = compute_popularity_impressions_global(
        dir_results=dir_results,
        dict_results=dict_results,
        df_impressions_global=df_impressions_global,
    )
    dict_results.update(results)

    results = compute_correlation_interactions_impressions(
        dir_results=dir_results,
        dict_results=dict_results,
    )
    dict_results.update(results)

    # results = compute_position_interactions(
    #     dir_results=dir_results,
    #     dict_results=dict_results,
    #     df_interactions_inside_impressions=df_interactions_inside_impressions,
    # )
    # dict_results.update(results)

    # results = compute_position_impressions(
    #     dir_results=dir_results,
    #     dict_results=dict_results,
    #     df_interactions_with_impressions_all=df_interactions_with_impressions_all,
    #     df_impressions_global=df_impressions_global,
    # )
    # dict_results.update(results)

    # results = compute_vision_factor(
    #     dir_results=dir_results,
    #     dict_results=dict_results,
    #     df_interactions_with_impressions_all=df_interactions_with_impressions_all,
    #     df_interactions_outside_impressions=df_interactions_outside_impressions,
    #     df_interactions_inside_impressions=df_interactions_inside_impressions,
    # )
    # dict_results.update(results)

    # results = compute_ratings(
    #     dir_results=dir_results,
    #     dict_results=dict_results,
    #     df_interactions_with_impressions_all=df_interactions_with_impressions_all,
    #     df_interactions_outside_impressions=df_interactions_outside_impressions,
    #     df_interactions_inside_impressions=df_interactions_inside_impressions,
    # )
    # dict_results.update(results)

    # (
    #     df_interactions_with_impressions_all,
    #     df_interactions_inside_impressions,
    #     df_impressions_contextual_all,
    #     df_impressions_contextual_only_non_null_impressions,
    #     df_impressions_global,
    # ) = transform_dataframes_for_ctr_computation(
    #     df_interactions_with_impressions_all=df_interactions_with_impressions_all,
    #     df_interactions_outside_impressions=df_interactions_outside_impressions,
    #     df_interactions_inside_impressions=df_interactions_inside_impressions,
    #     df_impressions_contextual=df_impressions_contextual,
    #     df_impressions_global=df_impressions_global,
    # )
    #
    # results = compute_table_ctr(
    #     dir_results=dir_results,
    #     dict_results=dict_results,
    #     df_interactions_with_impressions_all=df_interactions_with_impressions_all,
    #     df_interactions_inside_impressions=df_interactions_inside_impressions,
    #     df_impressions_contextual_all=df_impressions_contextual_all,
    #     df_impressions_contextual_only_non_null_impressions=df_impressions_contextual_only_non_null_impressions,
    #     df_impressions_global=df_impressions_global,
    # )
    # dict_results.update(results)

    # results = compute_ctr_1d(
    #     dir_results=dir_results,
    #     dict_results=dict_results,
    #     df_interactions_with_impressions_all=df_interactions_with_impressions_all,
    #     df_interactions_inside_impressions=df_interactions_inside_impressions,
    #     df_impressions_contextual_all=df_impressions_contextual_all,
    #     df_impressions_contextual_only_non_null_impressions=df_impressions_contextual_only_non_null_impressions,
    # )
    # dict_results.update(results)
    #
    # results = compute_ctr_2d(
    #     dir_results=dir_results,
    #     dict_results=dict_results,
    #     df_interactions_with_impressions_all=df_interactions_with_impressions_all,
    #     df_interactions_inside_impressions=df_interactions_inside_impressions,
    #     df_impressions_contextual_all=df_impressions_contextual_all,
    #     df_impressions_contextual_only_non_null_impressions=df_impressions_contextual_only_non_null_impressions,
    # )
    # dict_results.update(results)
    #
    # results = compute_ctr_3d(
    #     dir_results=dir_results,
    #     dict_results=dict_results,
    #     df_interactions_with_impressions_all=df_interactions_with_impressions_all,
    #     df_interactions_inside_impressions=df_interactions_inside_impressions,
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
