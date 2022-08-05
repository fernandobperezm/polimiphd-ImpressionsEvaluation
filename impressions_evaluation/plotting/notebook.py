from typing import Union, Any

import dask.dataframe as dd
import holoviews as hv
import numpy as np
import pandas as pd


def plot_curve_statistics_popularity_by_dataset_and_feature(
    statistics: dict[str, Any],
    dataset: str,
    feature: str,
) -> tuple[hv.Curve, hv.Curve]:
    kdims = feature.capitalize().replace("_", " ").replace("id", "")
    vdims = "# Records"

    profile_length: pd.DataFrame = statistics[dataset][feature]["profile_length"].sort_values(
        by=feature,
        ascending=False,
        ignore_index=True,
        inplace=False
    ).reset_index(
        drop=False
    ).rename(
        columns={
            "index": kdims,
            feature: vdims,
        }
    )

    profile_length_normalized = statistics[dataset][feature]["profile_length_normalized"].sort_values(
        by=feature,
        ascending=False,
        ignore_index=True,
        inplace=False
    ).reset_index(
        drop=False
    ).rename(
        columns={
            "index": kdims,
            feature: vdims,
        }
    )

    curve_profile = hv.Curve(profile_length)
    curve_profile_norm = hv.Curve(profile_length_normalized)

    return curve_profile, curve_profile_norm


def plot_curve_num_records_each_dataset(
    statistics: dict[str, Any],
    datasets_to_include: list[str],
    plot_name: str,
    plot_title: str,
) -> tuple[hv.Curve, hv.Curve]:
    plot_data = [
        (
            dataset_name.replace("_no_dup", ""),
            "w/o dups" if "no_dup" in dataset_name else "w/ dups",
            dataset_statistics["num_records"]
        )
        for dataset_name, dataset_statistics in statistics.items()
    ]
    plot_df = pd.DataFrame.from_records(
        data=plot_data,
        columns=["Dataset", "Type", "# Records"],
    ).astype(
        dtype={
            "Dataset": pd.StringDtype(),
            "Type": pd.StringDtype(),
            "# Records": np.int64
        },
    )

    bars = hv.Curve(
        plot_df,
        kdims=['Dataset', 'Type'],
        name=plot_name,
        label=plot_title
    ).opts(
        width=600,
        logx=False,
        logy=False,
    )

    return bars, bars["Dataset"].isin(datasets_to_include)


def plot_dynamic_map_from_values(
    plot_func: Any,
    kdims: list[str],
    values: dict[str, Any],
) -> hv.DynamicMap:
    """

    Examples
    --------
    >>> dmap = hv.DynamicMap(
    ...   plot_curve_popularity_by_dataset_and_feature,
    ...   kdims=['dataset', 'feature'],
    ... )
    ... dmap.redim.values(
    ...   dataset=list(interaction_statistics.keys()),
    ...   feature=["user_id", "item_id", "time_step"],
    ... )

    """
    dmap = hv.DynamicMap(
        plot_func,
        kdims=kdims,
    )
    dmap.redim.values(
        values=values,
    )

    return dmap


def plot_curve_popularity_by_counts(
    df_data: Union[pd.DataFrame, dd.DataFrame],
    column_to_calculate: str,
    sort: bool,
    plot_name: str,
    plot_title: str,
    plot_x_label: str,
) -> Union[tuple[hv.Curve, hv.Curve], tuple[hv.Bars, hv.Bars]]:
    df_popularity = df_data[column_to_calculate].value_counts(
        normalize=False,
        sort=sort,
        ascending=False,
        dropna=True,
    ).to_frame(
        name="popularity",
    ).reset_index(
        drop=False,
    ).reset_index(
        drop=False,
    ).rename(
        columns={
            "level_0": "rank",
            "index": column_to_calculate,
        }
    )

    if isinstance(df_data, dd.DataFrame):
        df_popularity = df_popularity.compute()

    dim_popularity = hv.Dimension('popularity', label='Popularity')
    num_unique_points = df_popularity.shape[0]
    if num_unique_points <= 30:
        dim_column = hv.Dimension(column_to_calculate, label=plot_x_label)

        plot = hv.Bars(
            data=df_popularity,
            kdims=dim_column,
            vdims=dim_popularity,
            name=f"{plot_name}",
            label=f"{plot_title}",
        ).opts(
            xrotation=45,
        )

        plot_log_y = hv.Bars(
            data=df_popularity,
            kdims=dim_column,
            vdims=dim_popularity,
            name=f"{plot_name}",
            label=f"{plot_title} - Log Y",
        ).opts(
            xrotation=45,
        )

    else:
        dim_column = hv.Dimension("rank", label=plot_x_label)
        plot = hv.Curve(
            data=df_popularity,
            kdims=dim_column,
            vdims=dim_popularity,
            name=f"{plot_name}",
            label=f"{plot_title}",
        )

        plot_log_y = hv.Curve(
            data=df_popularity,
            kdims=dim_column,
            vdims=dim_popularity,
            name=f"{plot_name}",
            label=f"{plot_title} - Log Y",
        )

    return plot.opts(
        logx=False,
        logy=False,
    ), plot_log_y.opts(
        logx=False,
        logy=True,
    )


def plot_histogram_timestamp(
    df_data: Union[pd.DataFrame, dd.DataFrame],
    by: str,
    column_to_calculate: str,
    plot_title: str,
    plot_x_label: str,
) -> Union[hv.Bars, hv.Curve]:

    is_timestamp_datetime = df_data[column_to_calculate].dtype == "datetime64[ns]"

    if is_timestamp_datetime:
        if by == "date":
            new_timestamp_colum = df_data[column_to_calculate].dt.date
        elif by == "hour":
            new_timestamp_colum = df_data[column_to_calculate].dt.hour
        elif by == "minute":
            new_timestamp_colum = df_data[column_to_calculate].dt.minute
        elif by == "round_hour":
            # rounds the datetime to the nearest hour.
            new_timestamp_colum = df_data[column_to_calculate].dt.round(freq="H")
        elif by == "round_minute":
            # rounds the datetime to the nearest minute.
            new_timestamp_colum = df_data[column_to_calculate].dt.round(freq="T")
        elif by == "day_name":
            new_timestamp_colum = df_data[column_to_calculate].dt.day_name()
        else:
            new_timestamp_colum = df_data[column_to_calculate]
    else:
        new_timestamp_colum = df_data[column_to_calculate]

    df_popularity = new_timestamp_colum.value_counts(
        normalize=False,
        sort=False,
        ascending=False,
        dropna=True,
    ).to_frame(
        name="popularity",
    ).sort_index(
        ascending=True,
    ).reset_index(
        drop=False,
    )

    num_unique_points = df_popularity.shape[0]

    dim_popularity = hv.Dimension('popularity', label='Popularity')
    dim_new_column = hv.Dimension("index", label=plot_x_label)

    if num_unique_points < 20:
        plot = hv.Bars(
            data=df_popularity,
            kdims=dim_new_column,
            vdims=dim_popularity,
            label=f"{plot_title}",
        ).opts(
            xrotation=45,
        )
    else:
        plot = hv.Curve(
            data=df_popularity,
            kdims=dim_new_column,
            vdims=dim_popularity,
            label=f"{plot_title}",
        )

    return plot.opts(
        logx=False,
        logy=False,
    )
