import itertools
import os
import pdb
from typing import Optional, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import paxplot
import tikzplotlib
from recsys_framework_extensions.data.io import DataIO
from impressions_evaluation.experiments import commons
from impressions_evaluation.experiments import print_results
from impressions_evaluation.experiments.baselines import DIR_TRAINED_MODELS_BASELINES
from impressions_evaluation.readers.ContentWiseImpressions.statistics import (
    plot_histogram,
    plot_barplot,
    compute_popularity,
)

DIR_ANALYSIS_HYPER_PARAMETERS = os.path.join(
    commons.DIR_RESULTS_EXPORT,
    "analysis_hyperparameters",
    "",
)


def plot_parallel_coordinates(
    df: pd.DataFrame,
    dir_results: str,
    dict_mappings: dict[str, dict[str, int]],
    col_data: str,
    name: str,
) -> None:
    columns = df.columns.tolist()

    # Data in ascending order + the colormap ensures that better values are highlighted in darker colors.
    data = df.sort_values(
        by=col_data, ascending=True, ignore_index=True, inplace=False
    ).copy()  #

    paxfig = paxplot.pax_parallel(n_axes=len(columns))
    paxfig.plot(data.to_numpy())

    color_col = len(columns) - 1
    paxfig.add_colorbar(
        ax_idx=color_col,
        cmap="YlGn",  # This colormap begins with soft colors, then with darker colors.
        colorbar_kwargs={"label": columns[color_col]},
    )

    columns_to_map = list(dict_mappings.keys())
    for col in columns_to_map:
        if col not in columns:
            print(
                f"Tried to map column {col} but did not found it in all columns {columns}"
            )
            continue

        col_idx = columns.index(col)
        col_mapping = dict_mappings[col]

        ticks = []
        labels = []
        for val_str, val_int in col_mapping.items():
            ticks.append(val_int)
            labels.append(str(val_str))
        paxfig.set_ticks(ax_idx=col_idx, ticks=ticks, labels=labels)

    # paxfig.plot(
    #     data.to_numpy(), line_kwargs={"alpha": 0.3, "color": "grey", "zorder": 0}
    # )

    paxfig.set_labels(columns)

    # ticks_similarity = []
    # labels_similarity = []
    # for sim, idx in map_similarity_to_idx.items():
    #     ticks_similarity.append(idx)
    #     labels_similarity.append(sim)
    # paxfig.set_ticks(ax_idx=2, ticks=ticks_similarity, labels=labels_similarity)

    # ticks_feature = []
    # labels_feature = []
    # for feat, idx in map_feature_to_idx.items():
    #     ticks_feature.append(idx)
    #     labels_feature.append(feat)
    # paxfig.set_ticks(ax_idx=5, ticks=ticks_feature, labels=labels_feature)

    # fig = paxfig.figure
    # tikzplotlib.clean_figure(fig=fig)  # This method throws an error.
    # tikzplotlib.clean_figure(fig=paxfig)  # This method throws an error.

    folder_to_save_tikz = os.path.join(dir_results, "tikz", "")
    folder_to_save_png = os.path.join(dir_results, "png", "")
    folder_to_save_pdf = os.path.join(dir_results, "pdf", "")

    os.makedirs(folder_to_save_tikz, exist_ok=True)
    os.makedirs(folder_to_save_png, exist_ok=True)
    os.makedirs(folder_to_save_pdf, exist_ok=True)

    filename = f"plot-parallel_coordinates-{name}"
    paxfig.savefig(
        os.path.join(folder_to_save_png, f"{filename}.png"),
    )
    paxfig.savefig(
        os.path.join(folder_to_save_pdf, f"{filename}.pdf"),
        transparent=False,
    )
    # tikzplotlib generates weird figures. Will leave it commented and will try to fix it later.
    # tikzplotlib.save(
    #     os.path.join(folder_to_save_tikz, f"{filename}.tikz"),  # cannot be kwarg
    #     paxfig,  # try this one if fig does not work.
    #     encoding="utf-8",
    #     textsize=9,
    # )

    plt.close(paxfig)


def distribution_hyper_parameters(
    benchmarks: list[commons.Benchmarks],
    hyper_parameters: list[commons.EHyperParameterTuningParameters],
) -> None:
    results_hyper_parameters = []
    for benchmark, hyper_parameter in itertools.product(benchmarks, hyper_parameters):
        experiment_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[benchmark]
        experiment_hyper_parameters = (
            commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[hyper_parameter]
        )

        folder_path_results_to_load = print_results.DIR_PARQUET_RESULTS.format(
            benchmark=experiment_benchmark.benchmark.value,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
        )

        df_results_hyper_parameters = pd.read_parquet(
            path=os.path.join(
                folder_path_results_to_load, "hyper-parameters-non_processed.parquet"
            ),
            engine="pyarrow",
        )

        results_hyper_parameters.append(df_results_hyper_parameters)

    folder_path_results_to_export = DIR_ANALYSIS_HYPER_PARAMETERS
    os.makedirs(folder_path_results_to_export, exist_ok=True)

    df_results_hyper_parameters = pd.concat(
        objs=results_hyper_parameters,
        axis=0,
        ignore_index=True,  # The index should be numeric and have no special meaning.
    )

    print(df_results_hyper_parameters)

    # "benchmark", "model_type", "hyperparameter_name"
    unique_benchmarks = df_results_hyper_parameters["benchmark"].unique().tolist()
    unique_model_types = df_results_hyper_parameters["model_type"].unique().tolist()
    unique_hyperparameter_names = (
        df_results_hyper_parameters["hyperparameter_name"].unique().tolist()
    )

    dtypes = {
        "Cycling": {
            "sign": np.int32,
            "weight": np.int32,
        },
        "HFC": {
            "mode": pd.StringDtype(),
            "threshold": np.int32,
        },
        "IDF": {
            "reg_uim_frequency": np.float32,
            "reg_uim_last_seen": np.float32,
            "reg_uim_position": np.float32,
            "reg_user_frequency": np.float32,
            #
            "sign_uim_frequency": np.int32,
            "sign_uim_last_seen": np.int32,
            "sign_uim_position": np.int32,
            "sign_user_frequency": np.int32,
            #
            "func_uim_frequency": pd.StringDtype(),
            "func_uim_last_seen": pd.StringDtype(),
            "func_uim_position": pd.StringDtype(),
            "func_user_frequency": pd.StringDtype(),
        },
        "IUP": {
            "alpha": np.float32,
            "sign": np.int32,
            "weighted_user_profile_type": pd.StringDtype(),
        },
    }

    benchmark: str
    model_type: str
    hyper_parameter_name: str
    for benchmark, model_type, hyper_parameter_name in itertools.product(
        unique_benchmarks, unique_model_types, unique_hyperparameter_names
    ):
        if (
            "baseline" == model_type.lower()
            or "Baseline-IARS".lower() == model_type.lower()
        ):
            continue

        df = df_results_hyper_parameters[
            (df_results_hyper_parameters["benchmark"] == benchmark)
            & (df_results_hyper_parameters["model_type"] == model_type)
            & (
                df_results_hyper_parameters["hyperparameter_name"]
                == hyper_parameter_name
            )
        ].copy()

        if df.shape[0] == 0:
            continue

        col_dtype = dtypes[model_type][hyper_parameter_name]
        try:
            df = df.astype({"hyperparameter_value": col_dtype})
        except ValueError as e:
            print(
                f"COULD NOT CONVERT COLUMN TO SPECIFIED DTYPE {str(col_dtype)}. CONVERTING TO STRING."
            )
            df = df.astype({"hyperparameter_value": pd.StringDtype()})

        hyper_parameter_value_dtype = df["hyperparameter_value"].dtype.name

        x_data = "hyperparameter_value"
        x_label = f"{model_type}-{hyper_parameter_name}"
        y_label = "Frequency"

        name = (
            f"analysis_hyperparameters-{benchmark}-{model_type}-{hyper_parameter_name}"
        )

        if "float" in hyper_parameter_value_dtype:
            print(f"PRINTING FLOAT-HIST FOR {name}")

            plot_histogram(
                df=df,
                x_data=x_data,
                x_label=x_label,
                y_label=y_label,
                name=name,
                dir_results=folder_path_results_to_export,
            )
        elif "int" in hyper_parameter_value_dtype:
            print(f"PRINTING INT-BAR FOR {name}")

            df_pop, df_pop_perc = compute_popularity(df=df, column=x_data)

            df_pop = df_pop.sort_values(
                by=x_data,
                ascending=True,
                inplace=False,
            )

            plot_barplot(
                df=df_pop,
                x_data=x_data,
                y_data="count",
                x_label=x_label,
                y_label=y_label,
                name=name,
                dir_results=folder_path_results_to_export,
                ticks_labels=None,
                log=False,
                align="center",
            )
        elif "string" in hyper_parameter_value_dtype:
            print(f"PRINTING STR-BAR FOR {name}")

            df_pop, df_pop_perc = compute_popularity(df=df, column=x_data)

            df_pop = df_pop.sort_values(by=x_data, ascending=True, inplace=False)

            plot_barplot(
                df=df_pop,
                x_data=x_data,
                y_data="count",
                x_label=x_label,
                y_label=y_label,
                name=name,
                dir_results=folder_path_results_to_export,
                ticks_labels=None,
                log=False,
                align="center",
            )

            continue
        else:
            print(f"FERNANDO-DEBUGGER|COMPLETE THIS - ELSE.")
            continue


def _load_metadata_plug_in_impression_aware_recommender(
    *,
    benchmark: commons.Benchmarks,
    hyper_parameter: commons.EHyperParameterTuningParameters,
    rec_impression: str,
    rec_baseline: str,
) -> Optional[dict[str, Any]]:
    recommender = f"{rec_impression}_{rec_baseline}"

    return _load_metadata_recommender(
        benchmark=benchmark,
        hyper_parameter=hyper_parameter,
        recommender=recommender,
    )


def _load_metadata_recommender(
    *,
    benchmark: commons.Benchmarks,
    hyper_parameter: commons.EHyperParameterTuningParameters,
    recommender: str,
) -> Optional[dict[str, Any]]:
    experiment_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[benchmark]
    experiment_hyper_parameters = (
        commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[hyper_parameter]
    )

    folder_trained_models = DIR_TRAINED_MODELS_BASELINES.format(
        benchmark=experiment_benchmark.benchmark.value,
        evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
    )
    recommender_filename = f"{recommender}_metadata.zip"

    if recommender_filename not in os.listdir(folder_trained_models):
        return None

    data_recommender = DataIO.s_load_data(
        folder_path=folder_trained_models, file_name=recommender_filename
    )

    return data_recommender


def _prepare_recommender_data_for_parallel_plot(
    data_recommender: dict[str, Any],
    dict_mappings: dict[str, dict[str, int]],
    list_columns_to_remove: list[str],
    metrics_to_optimize: list[str],
    cutoff_to_optimize: int,
) -> Optional[pd.DataFrame]:
    df_hyperparameters: pd.DataFrame = data_recommender["hyperparameters_df"]
    df_results_validation: pd.DataFrame = data_recommender["result_on_validation_df"]

    if data_recommender["hyperparameters_df"] is None:
        return None

    if data_recommender["result_on_validation_df"] is None:
        return None

    df_results_on_metric_and_cutoff = df_results_validation.reset_index(
        drop=False, level=1
    )

    df_results_on_metric_and_cutoff = df_results_on_metric_and_cutoff[
        (df_results_on_metric_and_cutoff["cutoff"] == cutoff_to_optimize)
    ][metrics_to_optimize].astype(np.float32)

    df_hyperparameters_and_result = pd.concat(
        objs=[df_hyperparameters, df_results_on_metric_and_cutoff],
        axis="columns",
        ignore_index=False,
        verify_integrity=True,
    )

    if np.any(df_hyperparameters_and_result.isna()):
        print(f"Detected NA value.")
        return None

    for col, mapping in dict_mappings.items():
        try:
            df_hyperparameters_and_result[col] = (
                df_hyperparameters_and_result[col].map(mapping).astype(np.int32)
            )
        except pd.errors.IntCastingNaNError:
            print(df_hyperparameters_and_result)
            print(df_hyperparameters_and_result[col])
            print(mapping)

            import pdb

            pdb.set_trace()

            print(
                f"For dict mapping {dict_mappings}, tried to convert column {col} with mapping {mapping} into an integer but failed. Check plot."
            )

    if len(list_columns_to_remove) > 0:
        df_hyperparameters_and_result = df_hyperparameters_and_result.drop(
            columns=list_columns_to_remove, inplace=False
        )

    return df_hyperparameters_and_result


def _create_dict_mapping(
    recommender: str,
) -> dict[str, dict[str, int]]:
    if "p3alpha" in recommender.casefold() or "rp3beta" in recommender.casefold():
        return {"normalize_similarity": {False: 0, True: 1}}

    if recommender == "CyclingRecommender":
        return {}
    elif recommender == "HardFrequencyCappingRecommender":
        return {"mode": {"leq": 0, "geq": 1}}
    elif recommender == "ImpressionsDiscountingRecommender":
        return {
            "func_user_frequency": {
                "LINEAR": 0,
                "INVERSE": 1,
                "EXPONENTIAL": 2,
                "LOGARITHMIC": 3,
                "QUADRATIC": 4,
                "SQUARE_ROOT": 5,
            },
            "func_uim_frequency": {
                "LINEAR": 0,
                "INVERSE": 1,
                "EXPONENTIAL": 2,
                "LOGARITHMIC": 3,
                "QUADRATIC": 4,
                "SQUARE_ROOT": 5,
            },
            "func_uim_position": {
                "LINEAR": 0,
                "INVERSE": 1,
                "EXPONENTIAL": 2,
                "LOGARITHMIC": 3,
                "QUADRATIC": 4,
                "SQUARE_ROOT": 5,
            },
            "func_uim_last_seen": {
                "LINEAR": 0,
                "INVERSE": 1,
                "EXPONENTIAL": 2,
                "LOGARITHMIC": 3,
                "QUADRATIC": 4,
                "SQUARE_ROOT": 5,
            },
        }
    elif (
        recommender == "BaseWeightedUserProfileRecommender"
        or recommender == "ItemWeightedUserProfileRecommender"
        or recommender == "UserWeightedUserProfileRecommender"
    ):
        return {
            "weighted_user_profile_type": {
                "ONLY_IMPRESSIONS": 0,
                "INTERACTIONS_AND_IMPRESSIONS": 1,
            },
        }

    return {}


def _create_list_columns_to_remove_impression_aware_recommenders(
    rec_impression: str,
) -> list[str]:
    if rec_impression == "CyclingRecommender":
        return []
    elif rec_impression == "HardFrequencyCappingRecommender":
        return []
    elif rec_impression == "ImpressionsDiscountingRecommender":
        return [
            "sign_user_frequency",
            "sign_uim_frequency",
            "sign_uim_position",
            # "sign_uim_last_seen",
            #
            "reg_user_frequency",
            "reg_uim_frequency",
            "reg_uim_position",
            # "reg_uim_last_seen",
            #
            "func_user_frequency",
            "func_uim_frequency",
            "func_uim_position",
            # "func_uim_last_seen",
        ]
    elif (
        rec_impression == "BaseWeightedUserProfileRecommender"
        or rec_impression == "ItemWeightedUserProfileRecommender"
        or rec_impression == "UserWeightedUserProfileRecommender"
    ):
        return ["weighted_user_profile_type"]

    return []


def plot_parallel_hyper_parameters_plug_in_impression_aware_recommenders(
    benchmarks: list[commons.Benchmarks],
    hyper_parameters: list[commons.EHyperParameterTuningParameters],
    baseline_recommenders: list[str],
    impression_aware_recommenders: list[str],
    metrics_to_optimize: list[str],
    cutoff_to_optimize: int,
    dir_analysis_hyper_parameters: str,
) -> None:
    if len(metrics_to_optimize) == 0:
        raise ValueError(
            "Must select at least one metric to plot the parallel coordinates of hyper-parameters."
        )

    main_metric = metrics_to_optimize[-1]

    benchmark: commons.Benchmarks
    hyper_parameter: commons.EHyperParameterTuningParameters
    rec_baseline: str
    rec_impression: str

    for benchmark, hyper_parameter, rec_impression, rec_baseline in itertools.product(
        benchmarks,
        hyper_parameters,
        impression_aware_recommenders,
        baseline_recommenders,
    ):
        data_recommender = _load_metadata_plug_in_impression_aware_recommender(
            benchmark=benchmark,
            hyper_parameter=hyper_parameter,
            rec_impression=rec_impression,
            rec_baseline=rec_baseline,
        )
        if data_recommender is None:
            print(
                f"Could not find a file for the combination {benchmark}-{rec_impression}-{rec_baseline}. Skipping"
            )
            continue

        dict_mappings = _create_dict_mapping(
            recommender=rec_impression,
        )

        list_columns_to_remove = (
            _create_list_columns_to_remove_impression_aware_recommenders(
                rec_impression=rec_impression,
            )
        )

        df_hyperparameters_and_result = _prepare_recommender_data_for_parallel_plot(
            data_recommender=data_recommender,
            dict_mappings=dict_mappings,
            list_columns_to_remove=list_columns_to_remove,
            metrics_to_optimize=metrics_to_optimize,
            cutoff_to_optimize=cutoff_to_optimize,
        )
        if df_hyperparameters_and_result is None:
            print(
                f"The recommender may not be finished for the combination {benchmark}-{rec_impression}-{rec_baseline}. Skipping"
            )
            continue

        plot_parallel_coordinates(
            df=df_hyperparameters_and_result,
            dict_mappings=dict_mappings,
            dir_results=dir_analysis_hyper_parameters,
            name=f"{benchmark.value}-{rec_impression}-{rec_baseline}",
            col_data=main_metric,
        )


def plot_parallel_hyper_parameters_recommenders(
    benchmarks: list[commons.Benchmarks],
    hyper_parameters: list[commons.EHyperParameterTuningParameters],
    recommenders: list[str],
    metrics_to_optimize: list[str],
    cutoff_to_optimize: int,
    dir_analysis_hyper_parameters: str,
) -> None:
    if len(metrics_to_optimize) == 0:
        raise ValueError(
            "Must select at least one metric to plot the parallel coordinates of hyper-parameters."
        )

    os.makedirs(dir_analysis_hyper_parameters, exist_ok=True)

    main_metric = metrics_to_optimize[-1]

    benchmark: commons.Benchmarks
    hyper_parameter: commons.EHyperParameterTuningParameters
    recommender: str

    for benchmark, hyper_parameter, recommender in itertools.product(
        benchmarks, hyper_parameters, recommenders
    ):
        data_recommender = _load_metadata_recommender(
            benchmark=benchmark,
            hyper_parameter=hyper_parameter,
            recommender=recommender,
        )
        if data_recommender is None:
            print(
                f"Could not find a file for the combination {benchmark}-{recommender}. Skipping"
            )
            continue

        dict_mappings: dict[str, dict[str, int]] = _create_dict_mapping(
            recommender=recommender
        )
        list_columns_to_remove: list[str] = []

        df_hyperparameters_and_result = _prepare_recommender_data_for_parallel_plot(
            data_recommender=data_recommender,
            dict_mappings=dict_mappings,
            list_columns_to_remove=list_columns_to_remove,
            metrics_to_optimize=metrics_to_optimize,
            cutoff_to_optimize=cutoff_to_optimize,
        )
        if df_hyperparameters_and_result is None:
            print(
                f"The recommender may not be finished for the combination {benchmark}-{recommender}. Skipping"
            )
            continue

        name = f"{benchmark.value}-{recommender}"
        try:
            plot_parallel_coordinates(
                df=df_hyperparameters_and_result,
                dict_mappings=dict_mappings,
                dir_results=dir_analysis_hyper_parameters,
                name=name,
                col_data=main_metric,
            )
            print(
                f"Just exported the parallel coordinates plot for the combination {benchmark.value}-{recommender}"
            )
        except:
            print(df_hyperparameters_and_result)
            print(dict_mappings)
            print(name)
            print(main_metric)
            import pdb

            pdb.set_trace()
            print(dir_analysis_hyper_parameters)
            print(df_hyperparameters_and_result)
