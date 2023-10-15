import itertools
import logging
import os
from typing import cast, Union, Sequence, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from recsys_framework_extensions.plotting import (
    DataFrameResults,
    generate_accuracy_and_beyond_metrics_pandas,
)

from impressions_evaluation.experiments import commons
from impressions_evaluation.experiments.baselines import DIR_TRAINED_MODELS_BASELINES
from impressions_evaluation.experiments.graph_based import (
    DIR_TRAINED_MODELS_IMPRESSION_AWARE,
)
from impressions_evaluation.experiments.print_results import (
    DIR_RESULTS_MODEL_EVALUATION,
)

logger = logging.getLogger(__file__)


####################################################################################################
####################################################################################################
#                                REPRODUCIBILITY VARIABLES                            #
####################################################################################################
####################################################################################################
DIR_RESULTS_TO_EXPORT = os.path.join(
    DIR_RESULTS_MODEL_EVALUATION,
    "experiment_graph_based_impression_aware_recommenders",
    "",
)
DIR_RESULTS_TO_PROCESS = os.path.join(
    DIR_RESULTS_TO_EXPORT,
    "{benchmark}",
    "{evaluation_strategy}",
    "",
)
DIR_CSV_RESULTS = os.path.join(
    DIR_RESULTS_TO_PROCESS,
    "csv",
    "",
)
DIR_PARQUET_RESULTS = os.path.join(
    DIR_RESULTS_TO_PROCESS,
    "parquet",
    "",
)
DIR_LATEX_RESULTS = os.path.join(
    DIR_RESULTS_TO_PROCESS,
    "latex",
    "",
)

commons.FOLDERS.add(DIR_RESULTS_MODEL_EVALUATION)
commons.FOLDERS.add(DIR_RESULTS_TO_EXPORT)
commons.FOLDERS.add(DIR_RESULTS_TO_PROCESS)
commons.FOLDERS.add(DIR_CSV_RESULTS)
commons.FOLDERS.add(DIR_PARQUET_RESULTS)


RESULT_EXPORT_CUTOFFS = [5, 10, 20, 30, 40, 50, 100]

ACCURACY_METRICS_LIST = [
    "NDCG",
    "PRECISION",
    "RECALL",
    "F1",
]
BEYOND_ACCURACY_METRICS_LIST = [
    "NOVELTY",
    "DIVERSITY_MEAN_INTER_LIST",
    "COVERAGE_ITEM",
    "DIVERSITY_GINI",
    "SHANNON_ENTROPY",
]
ALL_METRICS_LIST = [
    *ACCURACY_METRICS_LIST,
    *BEYOND_ACCURACY_METRICS_LIST,
]


####################################################################################################
####################################################################################################
#             Generation of dataframes with evaluation results          #
####################################################################################################
####################################################################################################
def _print_collaborative_filtering_metrics(
    recommender_baseline: commons.RecommenderBaseline,
    experiment_benchmark: commons.ExperimentBenchmark,
    experiment_hyper_parameters: commons.HyperParameterTuningParameters,
    num_test_users: int,
    accuracy_metrics_list: list[str],
    beyond_accuracy_metrics_list: list[str],
    all_metrics_list: list[str],
    cutoffs_list: list[int],
    knn_similarity_list: list[commons.T_SIMILARITY_TYPE],
    export_experiments_folder_path: str,
) -> DataFrameResults:
    experiments_folder_path = DIR_TRAINED_MODELS_BASELINES.format(
        benchmark=experiment_benchmark.benchmark.value,
        evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
    )

    baseline_experiment_recommenders = commons.MAPPER_AVAILABLE_RECOMMENDERS[
        recommender_baseline
    ]
    base_algorithm_list = [baseline_experiment_recommenders.recommender]

    return generate_accuracy_and_beyond_metrics_pandas(
        experiments_folder_path=experiments_folder_path,
        export_experiments_folder_path=export_experiments_folder_path,
        num_test_users=num_test_users,
        base_algorithm_list=base_algorithm_list,
        knn_similarity_list=knn_similarity_list,
        other_algorithm_list=None,
        accuracy_metrics_list=accuracy_metrics_list,
        beyond_accuracy_metrics_list=beyond_accuracy_metrics_list,
        all_metrics_list=all_metrics_list,
        cutoffs_list=cutoffs_list,
        icm_names=None,
    )


def _print_pure_impression_aware_metrics(
    recommender_impressions: commons.RecommenderImpressions,
    experiment_benchmark: commons.ExperimentBenchmark,
    experiment_hyper_parameters: commons.HyperParameterTuningParameters,
    num_test_users: int,
    accuracy_metrics_list: list[str],
    beyond_accuracy_metrics_list: list[str],
    all_metrics_list: list[str],
    cutoffs_list: list[int],
    knn_similarity_list: list[commons.T_SIMILARITY_TYPE],
    export_experiments_folder_path: str,
) -> DataFrameResults:
    experiments_folder_path = DIR_TRAINED_MODELS_IMPRESSION_AWARE.format(
        benchmark=experiment_benchmark.benchmark.value,
        evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
    )

    baseline_experiment_recommenders = commons.MAPPER_AVAILABLE_RECOMMENDERS[
        recommender_impressions
    ]

    base_algorithm_list = [
        baseline_experiment_recommenders.recommender(
            urm_train=sp.csr_matrix([[]]),
            uim_train=sp.csr_matrix([[]]),
        )
    ]

    return generate_accuracy_and_beyond_metrics_pandas(
        experiments_folder_path=experiments_folder_path,
        export_experiments_folder_path=export_experiments_folder_path,
        num_test_users=num_test_users,
        base_algorithm_list=base_algorithm_list,
        knn_similarity_list=knn_similarity_list,
        other_algorithm_list=None,
        accuracy_metrics_list=accuracy_metrics_list,
        beyond_accuracy_metrics_list=beyond_accuracy_metrics_list,
        all_metrics_list=all_metrics_list,
        cutoffs_list=cutoffs_list,
        icm_names=None,
    )


def _print_frequency_impression_aware_metrics(
    recommender_impressions: commons.RecommenderImpressions,
    experiment_benchmark: commons.ExperimentBenchmark,
    experiment_hyper_parameters: commons.HyperParameterTuningParameters,
    num_test_users: int,
    accuracy_metrics_list: list[str],
    beyond_accuracy_metrics_list: list[str],
    all_metrics_list: list[str],
    cutoffs_list: list[int],
    knn_similarity_list: list[commons.T_SIMILARITY_TYPE],
    export_experiments_folder_path: str,
) -> DataFrameResults:
    experiments_folder_path = DIR_TRAINED_MODELS_IMPRESSION_AWARE.format(
        benchmark=experiment_benchmark.benchmark.value,
        evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
    )

    baseline_experiment_recommenders = commons.MAPPER_AVAILABLE_RECOMMENDERS[
        recommender_impressions
    ]

    base_algorithm_list = [
        baseline_experiment_recommenders.recommender(
            urm_train=sp.csr_matrix([[]]),
            uim_train=sp.csr_matrix([[]]),
            uim_frequency=sp.csr_matrix([[]]),
        )
    ]

    return generate_accuracy_and_beyond_metrics_pandas(
        experiments_folder_path=experiments_folder_path,
        export_experiments_folder_path=export_experiments_folder_path,
        num_test_users=num_test_users,
        base_algorithm_list=base_algorithm_list,
        knn_similarity_list=knn_similarity_list,
        other_algorithm_list=None,
        accuracy_metrics_list=accuracy_metrics_list,
        beyond_accuracy_metrics_list=beyond_accuracy_metrics_list,
        all_metrics_list=all_metrics_list,
        cutoffs_list=cutoffs_list,
        icm_names=None,
    )


def _print_plugin_impression_aware_metrics(
    recommender_baseline: commons.RecommenderBaseline,
    recommender_plugin: commons.RecommenderImpressions,
    experiment_benchmark: commons.ExperimentBenchmark,
    experiment_hyper_parameters: commons.HyperParameterTuningParameters,
    num_test_users: int,
    accuracy_metrics_list: list[str],
    beyond_accuracy_metrics_list: list[str],
    all_metrics_list: list[str],
    cutoffs_list: list[int],
    knn_similarity_list: list[commons.T_SIMILARITY_TYPE],
    export_experiments_folder_path: str,
) -> DataFrameResults:
    experiments_baseline_folder_path = DIR_TRAINED_MODELS_BASELINES.format(
        benchmark=experiment_benchmark.benchmark.value,
        evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
    )

    experiments_impression_folder_path = DIR_TRAINED_MODELS_IMPRESSION_AWARE.format(
        benchmark=experiment_benchmark.benchmark.value,
        evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
    )

    baseline_experiment_recommenders = commons.MAPPER_AVAILABLE_RECOMMENDERS[
        recommender_baseline
    ]

    impression_experiment_recommenders = commons.MAPPER_AVAILABLE_RECOMMENDERS[
        recommender_plugin
    ]

    base_algorithm_list = [
        impression_experiment_recommenders.recommender(
            urm_train=sp.csr_matrix([[]]),
            uim_train=sp.csr_matrix([[]]),
            traned_recommender=baseline_experiment_recommenders.recommender,
        )
    ]

    return generate_accuracy_and_beyond_metrics_pandas(
        experiments_folder_path=experiments_impression_folder_path,
        export_experiments_folder_path=export_experiments_folder_path,
        num_test_users=num_test_users,
        base_algorithm_list=base_algorithm_list,
        knn_similarity_list=knn_similarity_list,
        other_algorithm_list=None,
        accuracy_metrics_list=accuracy_metrics_list,
        beyond_accuracy_metrics_list=beyond_accuracy_metrics_list,
        all_metrics_list=all_metrics_list,
        cutoffs_list=cutoffs_list,
        icm_names=None,
    )


# def _print_metrics_group(
#     group_recommenders: Sequence[
#         Union[commons.RecommenderBaseline, commons.RecommenderImpressions]
#     ],
#     experiment_benchmark: commons.ExperimentBenchmark,
#     experiment_hyper_parameters: commons.HyperParameterTuningParameters,
#     num_test_users: int,
#     accuracy_metrics_list: list[str],
#     beyond_accuracy_metrics_list: list[str],
#     all_metrics_list: list[str],
#     cutoffs_list: list[int],
#     knn_similarity_list: list[commons.T_SIMILARITY_TYPE],
#     export_experiments_folder_path: str,
# ) -> DataFrameResults:
#     base_algorithm_list = []
#     for rec in group_recommenders:
#         if isinstance(rec, commons.RecommenderBaseline):
#             ...
#         elif isinstance(rec, commons.RecommenderImpressions):
#             experiments_folder_path = DIR_TRAINED_MODELS_IMPRESSION_AWARE.format(
#                 benchmark=experiment_benchmark.benchmark.value,
#                 evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
#             )
#             baseline_experiment_recommenders = _MAPPER_IMPRESSIONS_RECOMMENDERS[rec]
#         else:
#             continue
#
#     base_algorithm_list = [
#         baseline_experiment_recommenders.recommender(
#             urm_train=sp.csr_matrix([[]]),
#             uim_train=sp.csr_matrix([[]]),
#         )
#     ]
#
#     return generate_accuracy_and_beyond_metrics_pandas(
#         experiments_folder_path=experiments_folder_path,
#         export_experiments_folder_path=export_experiments_folder_path,
#         num_test_users=num_test_users,
#         base_algorithm_list=base_algorithm_list,
#         knn_similarity_list=knn_similarity_list,
#         other_algorithm_list=None,
#         accuracy_metrics_list=accuracy_metrics_list,
#         beyond_accuracy_metrics_list=beyond_accuracy_metrics_list,
#         all_metrics_list=all_metrics_list,
#         cutoffs_list=cutoffs_list,
#         icm_names=None,
#     )


def _results_to_pandas(
    dfs: list[pd.DataFrame],
    results_name: str,
    folder_path_latex: str,
    folder_path_csv: str,
    folder_path_parquet: str,
) -> None:
    MODEL_COLUMN = "Recommender"
    CUTOFF_COLUMN = "Cutoff"
    MODEL_BASE_COLUMN = "Model Name"
    MODEL_TYPE_COLUMN = "Model Type"
    ORDER_COLUMN = "Order"

    df_results: pd.DataFrame = pd.concat(
        dfs,
        axis=0,
        ignore_index=False,  # The index is the list of recommender names.
    )

    if "accuracy-metrics" == results_name:
        df_results = df_results.reset_index(
            drop=False,
        ).rename(columns={"index": MODEL_COLUMN})
    elif "times" == results_name:
        df_results = df_results.reset_index(
            drop=False
        ).rename(  # Makes the @20 column as another column.
            columns={
                "level_0": MODEL_COLUMN,
                "index": MODEL_COLUMN,
            }
        )
    elif "hyper-parameters" == results_name:
        # Resulting dataframe
        # Index: (algorithm_row_label, hyperparameter_name)
        # Columns: [hyperparameter_value]
        df_results = df_results.reset_index(drop=False).rename(
            columns={
                "algorithm_row_label": MODEL_COLUMN,
                "hyperparameter_name": "Hyper-Parameter",
                "hyperparameter_value": "Value",
            }
        )
    else:
        return

    df_results[MODEL_COLUMN] = (
        df_results[MODEL_COLUMN]
        .apply(
            lambda val: val.replace("ImpressionsDirected", "") + "-DG"
            if "ImpressionsDirected" in val
            else val,
        )
        .apply(
            lambda val: val.replace("ImpressionsProfile", "") + "-UP"
            if "ImpressionsProfile" in val
            else val,
        )
    )

    with pd.option_context("max_colwidth", 1000):
        df_results.to_csv(
            path_or_buf=os.path.join(folder_path_csv, f"{results_name}.csv"),
            index=True,
            header=True,
            encoding="utf-8",
            na_rep="-",
        )
        df_results.to_latex(
            buf=os.path.join(folder_path_latex, f"{results_name}.tex"),
            index=True,
            header=True,
            escape=False,
            float_format="{:.4f}".format,
            encoding="utf-8",
            na_rep="-",
            longtable=True,
        )


def _process_results_dataframe(
    *,
    dfs: list[pd.DataFrame],
    benchmark: commons.Benchmarks,
    results_name: str,
    folder_path_csv: str,
    folder_path_parquet: str,
) -> None:
    """
    Saves on disk a dataframe as follows:
    # benchmark | recommender | model_base | model_type | experiment_type | <columns of the dataframe>.
    """
    filename_non_processed = f"{results_name}-non_processed"

    column_recommender = "recommender"
    column_benchmark = "benchmark"
    column_model_base = "model_base"
    column_model_type = "model_type"
    column_model_base_order = "model_base_order"
    column_model_type_order = "model_type_order"
    column_experiment_type = "experiment_type"

    def normalize_dataframe_accuracy_metrics(df: pd.DataFrame) -> pd.DataFrame:
        return df.reset_index(
            drop=False,
            names=[column_recommender],
        )

    def normalize_dataframe_times(df: pd.DataFrame) -> pd.DataFrame:
        return df.reset_index(
            drop=False,
            names=[column_recommender],
        )

    def normalize_dataframe_hyper_parameters(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.reset_index(
                drop=False,
            ).rename(columns={"algorithm_row_label": column_recommender})
            # Force strings in the column because some hyper-parameter values are strings, others are floats or even integers. To avoid clashes when storing the data in parquet it is just easier to convert to string.
            .astype({"hyperparameter_value": pd.StringDtype()})
        )

    def convert_recommender_name_to_model_base(recommender_name: str) -> str:
        return (
            recommender_name
            # Impression-aware graph-based recommenders -- they are not base recommenders so we set them as empty.
            .replace("ImpressionsProfileWithFrequency", "")
            .replace("ImpressionsDirectedWithFrequency", "")
            .replace("ImpressionsProfile", "")
            .replace("ImpressionsDirected", "")
            .replace("WithFrequency", "")
            .replace("Extended", "")
            # Framework recommenders.
            .replace("Alpha", "alpha")  # To be consistent with the framework
            .replace("Beta", "beta")  # To be consistent with the framework
            .replace("Recommender", "")
            .replace("KNNCF", "KNN CF")
            .replace("CF", " ")
            # Remaining characters
            .replace("  ", "")
            .replace("_", "")
            .strip()
        )

    def convert_recommender_name_to_model_base_type(recommender_name: str) -> str:
        # We cannot use the recommender name, eg CyclingRecommender.RECOMMENDER_NAME because this dataframe has the recommender names already pre-processed for printing.
        if "ImpressionsProfileWithFrequency" in recommender_name:
            return "IP-F"

        if "ImpressionsProfile" in recommender_name:
            return "IP-E"

        if "ImpressionsDirectedWithFrequency" in recommender_name:
            return "DG-F"

        if "ImpressionsDirected" in recommender_name:
            return "DG-E"

        # Covers framework recommenders.
        return "Baseline"

    def convert_recommender_name_to_experiment_type(recommender_name: str) -> str:
        return ""

    def convert_model_base_to_model_base_order(model_base: str) -> float:
        model_base_order = 100.0

        if "Random" in model_base:
            model_base_order = 0.0
        if "TopPop" in model_base:
            model_base_order = 1.0

        if "P3alpha" in model_base or "ExtendedP3alpha" in model_base:
            model_base_order = 2.0
        if "RP3beta" in model_base or "ExtendedRP3beta" in model_base:
            model_base_order = 3.0
        if "LightGCN" in model_base or "ExtendedLightGCN" in model_base:
            model_base_order = 4.0

        return model_base_order

    def convert_model_type_to_model_type_order(model_type: str) -> int:
        model_order = 100
        if model_type == "Baseline":
            model_order = 0

        if model_type == "IP-E":
            model_order = 1
        if model_type == "IP-F":
            model_order = 2

        if model_type == "DG-E":
            model_order = 3
        if model_type == "DG-F":
            model_order = 4

        return model_order

    # This creates a dataframe with the following structure:
    # # benchmark | recommender | model_base | model_type | experiment_type | <columns of the dataframe>.
    df_results_accuracy_metrics: pd.DataFrame = pd.concat(
        objs=dfs,
        axis=0,
        ignore_index=False,  # The index is the list of recommender names.
    )

    if results_name == "accuracy-metrics":
        df_results_accuracy_metrics = normalize_dataframe_accuracy_metrics(
            df=df_results_accuracy_metrics
        )
    elif results_name == "times":
        df_results_accuracy_metrics = normalize_dataframe_times(
            df=df_results_accuracy_metrics
        )
    elif results_name == "hyper-parameters":
        df_results_accuracy_metrics = normalize_dataframe_hyper_parameters(
            df=df_results_accuracy_metrics
        )
    else:
        raise NotImplementedError(
            f'Currently we only load results of "accuracy-metrics", "times", or "hyper-parameters". Received: {results_name}'
        )

    df_results_accuracy_metrics[column_benchmark] = benchmark.value

    df_results_accuracy_metrics[column_model_base] = (
        df_results_accuracy_metrics[column_recommender]
        .apply(convert_recommender_name_to_model_base, convert_dtype=True)
        .astype(pd.StringDtype())
    )

    df_results_accuracy_metrics[column_model_type] = (
        df_results_accuracy_metrics[column_recommender]
        .apply(convert_recommender_name_to_model_base_type, convert_dtype=True)
        .astype(pd.StringDtype())
    )

    df_results_accuracy_metrics[column_experiment_type] = (
        df_results_accuracy_metrics[column_recommender]
        .apply(convert_recommender_name_to_experiment_type, convert_dtype=True)
        .astype(pd.StringDtype())
    )

    df_results_accuracy_metrics[column_model_base_order] = (
        df_results_accuracy_metrics[column_model_base]
        .apply(convert_model_base_to_model_base_order, convert_dtype=True)
        .astype(np.float32)
    )

    df_results_accuracy_metrics[column_model_type_order] = (
        df_results_accuracy_metrics[column_model_type]
        .apply(convert_model_type_to_model_type_order, convert_dtype=True)
        .astype(np.int32)
    )

    with pd.option_context("max_colwidth", 1000):
        df_results_accuracy_metrics.to_csv(
            path_or_buf=os.path.join(folder_path_csv, f"{filename_non_processed}.csv"),
            index=True,
            header=True,
            encoding="utf-8",
            na_rep="-",
            sep=";",
            decimal=".",
        )
        df_results_accuracy_metrics.to_parquet(
            path=os.path.join(folder_path_parquet, f"{filename_non_processed}.parquet"),
            engine="pyarrow",
            compression=None,
            index=True,
        )


def _export_results_accuracy_metrics(
    *,
    df_results: pd.DataFrame,
    results_name: str,
    cutoffs: Sequence[int],
    metrics: Sequence[str],
    order: Literal["cutoff", "metric"],
    benchmarks: list[commons.Benchmarks],
    folder_path_csv: str,
) -> None:
    """
    Saves on disk a dataframe as follows:
    * dataset | recommender | variant | type | (cutoff, metric_1) | ... | (cutoff, metric_n)
    """
    column_benchmark_order = "benchmark_order"

    column_benchmark = "benchmark"
    column_model_base = "model_base"
    column_model_type = "model_type"
    column_model_base_order = "model_base_order"
    column_model_type_order = "model_type_order"

    column_experiment_type = "experiment_type"

    column_export_benchmark = "Dataset"
    column_export_model_base = "Recommender"
    column_export_model_type = "Variant"
    column_export_experiment_type = "Experiment"

    benchmarks_order = {
        benchmark.value: idx for idx, benchmark in enumerate(benchmarks)
    }

    df_results[column_benchmark_order] = (
        df_results[column_benchmark].map(benchmarks_order).astype(np.int32)
    )

    df_results = df_results.sort_values(
        by=[
            column_benchmark_order,
            column_model_base_order,
            column_model_type_order,
            column_experiment_type,
        ],
        ascending=True,
        inplace=False,
        ignore_index=True,
    )

    # This creates a dataframe
    # benchmark | model_base | model_type | experiment_type | (cutoff, metric_1) | ... | (cutoff, metric_n)
    column_tuple_benchmark = ("benchmark", "")
    column_tuple_model_base = ("model_base", "")
    column_tuple_model_type = ("model_type", "")
    column_tuple_experiment_type = ("experiment_type", "")

    columns_cutoff = [str(cutoff) for cutoff in cutoffs]
    columns_metrics = [str(metric) for metric in metrics]

    if order == "cutoff":
        columns_pairs_cutoff_metric = [
            (cutoff, metric) for cutoff in columns_cutoff for metric in columns_metrics
        ]
    elif order == "metric":
        columns_pairs_cutoff_metric = [
            (cutoff, metric) for metric in columns_metrics for cutoff in columns_cutoff
        ]

    else:
        raise ValueError("")

    df_results = df_results[
        [
            column_tuple_benchmark,
            column_tuple_model_base,
            column_tuple_model_type,
            column_tuple_experiment_type,
            *columns_pairs_cutoff_metric,
        ]
    ].rename(
        columns={
            column_benchmark: column_export_benchmark,
            column_model_base: column_export_model_base,
            column_model_type: column_export_model_type,
            column_experiment_type: column_export_experiment_type,
        }
    )

    filename_export = f"accuracy-metrics-export-{results_name}"
    with pd.option_context("max_colwidth", 1000):
        df_results.to_csv(
            path_or_buf=os.path.join(folder_path_csv, f"{filename_export}.csv"),
            index=True,
            header=True,
            encoding="utf-8",
            na_rep="-",
            sep=";",
            decimal=",",
            float_format="%.4f",
        )


def _export_results_time(
    *,
    df_results: pd.DataFrame,
    results_name: str,
    benchmarks: list[commons.Benchmarks],
    folder_path_csv: str,
) -> None:
    """
    Saves on disk a dataframe as follows:
    * Dataset | Recommender | Variant | Experiment | Train time | Recommendation time | Recommendation throughput
    """
    column_benchmark_order = "benchmark_order"
    column_model_base_order = "model_base_order"
    column_model_type_order = "model_type_order"

    column_benchmark = "benchmark"
    column_model_base = "model_base"
    column_model_type = "model_type"
    column_experiment_type = "experiment_type"
    column_train_time = "Train Time"
    column_recommendation_time = "Recommendation Time"
    column_recommendation_throughput = "Recommendation Throughput"

    column_export_benchmark = "Dataset"
    column_export_model_base = "Recommender"
    column_export_model_type = "Variant"
    column_export_experiment_type = "Experiment"
    column_export_train_time = "Train time"
    column_export_recommendation_time = "Recommendation time"
    column_export_recommendation_throughput = "Recommendation throughput"

    benchmarks_order = {
        benchmark.value: idx for idx, benchmark in enumerate(benchmarks)
    }

    df_results[column_benchmark_order] = (
        df_results[column_benchmark].map(benchmarks_order).astype(np.int32)
    )

    df_results = df_results.sort_values(
        by=[
            column_benchmark_order,
            column_model_base_order,
            column_model_type_order,
            column_experiment_type,
        ],
        ascending=True,
        inplace=False,
        ignore_index=True,
    )

    # This creates a dataframe
    # Dataset | Recommender | Variant | Experiment | Train time | Recommendation time | Recommendation throughput
    df_results = df_results[
        [
            column_benchmark,
            column_model_base,
            column_model_type,
            column_experiment_type,
            column_train_time,
            column_recommendation_time,
            column_recommendation_throughput,
        ]
    ].rename(
        columns={
            column_benchmark: column_export_benchmark,
            column_model_base: column_export_model_base,
            column_model_type: column_export_model_type,
            column_experiment_type: column_export_experiment_type,
            column_train_time: column_export_train_time,
            column_recommendation_time: column_export_recommendation_time,
            column_recommendation_throughput: column_export_recommendation_throughput,
        }
    )

    filename_export = f"times-export-{results_name}"
    with pd.option_context("max_colwidth", 1000):
        df_results.to_csv(
            path_or_buf=os.path.join(folder_path_csv, f"{filename_export}.csv"),
            index=True,
            header=True,
            encoding="utf-8",
            na_rep="-",
            sep=";",
            decimal=",",
            float_format="%.4f",
        )


def _export_results_hyper_parameters(
    *,
    df_results: pd.DataFrame,
    results_name: str,
    benchmarks: list[commons.Benchmarks],
    folder_path_csv: str,
) -> None:
    """
    Saves on disk a dataframe as follows:
    * Dataset | Variant | Recommender | Experiment | Hyper-parameter | Value
    """
    column_benchmark_order = "benchmark_order"
    column_model_base_order = "model_base_order"
    column_model_type_order = "model_type_order"

    column_benchmark = "benchmark"
    column_model_base = "model_base"
    column_model_type = "model_type"
    column_experiment_type = "experiment_type"
    column_hyper_parameter_name = "hyperparameter_name"
    column_hyper_parameter_value = "hyperparameter_value"

    column_export_benchmark = "Dataset"
    column_export_model_base = "Recommender"
    column_export_model_type = "Variant"
    column_export_experiment_type = "Experiment"
    column_export_hyper_parameter_name = "Hyper-parameter"
    column_export_hyper_parameter_value = "Value"

    benchmarks_order = {
        benchmark.value: idx for idx, benchmark in enumerate(benchmarks)
    }

    df_results[column_benchmark_order] = (
        df_results[column_benchmark].map(benchmarks_order).astype(np.int32)
    )

    df_results = df_results.sort_values(
        by=[
            column_benchmark_order,
            column_model_type_order,
            column_model_base_order,
            column_experiment_type,
        ],
        ascending=True,
        inplace=False,
        ignore_index=True,
    )

    # This creates a dataframe
    # Dataset | Variant | Recommender | Experiment | Hyper-parameter | Value
    df_results = df_results[
        [
            column_benchmark,
            column_model_type,
            column_model_base,
            column_experiment_type,
            column_hyper_parameter_name,
            column_hyper_parameter_value,
        ]
    ].rename(
        columns={
            column_benchmark: column_export_benchmark,
            column_model_base: column_export_model_base,
            column_model_type: column_export_model_type,
            column_experiment_type: column_export_experiment_type,
            column_hyper_parameter_name: column_export_hyper_parameter_name,
            column_hyper_parameter_value: column_export_hyper_parameter_value,
        }
    )

    filename_export = f"hyper_parameters-export-{results_name}"
    with pd.option_context("max_colwidth", 1000):
        df_results.to_csv(
            path_or_buf=os.path.join(folder_path_csv, f"{filename_export}.csv"),
            index=True,
            header=True,
            encoding="utf-8",
            na_rep="-",
            sep=";",
            decimal=",",
            float_format="%.4f",
        )


def print_results(
    results_interface: list[
        tuple[
            commons.Benchmarks,
            commons.EHyperParameterTuningParameters,
            list[
                Sequence[
                    Union[commons.RecommenderBaseline, commons.RecommenderImpressions]
                ]
            ],
        ]
    ]
) -> None:
    """
    Public method that exports into CSV and LaTeX tables the evaluation metrics, hyper-parameters, and times.
    """
    printed_experiments: set[
        tuple[commons.Benchmarks, commons.EHyperParameterTuningParameters]
    ] = set()

    for benchmark, hyper_parameters, groups_recs in results_interface:
        experiment_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[benchmark]
        experiment_hyper_parameters = (
            commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[hyper_parameters]
        )

        data_reader = commons.get_reader_from_benchmark(
            benchmark_config=experiment_benchmark.config,
            benchmark=experiment_benchmark.benchmark,
        )

        dataset = data_reader.dataset
        interaction_data_splits = dataset.get_urm_splits(
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy,
        )

        urm_test = interaction_data_splits.sp_urm_test
        num_test_users = cast(int, np.sum(np.ediff1d(urm_test.indptr) >= 1))

        folder_path_export_latex = DIR_ACCURACY_METRICS_BASELINES_LATEX.format(
            benchmark=experiment_benchmark.benchmark.value,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
        )
        folder_path_export_csv = DIR_CSV_RESULTS.format(
            benchmark=experiment_benchmark.benchmark.value,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
        )
        folder_path_export_parquet = DIR_PARQUET_RESULTS.format(
            benchmark=experiment_benchmark.benchmark.value,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
        )

        os.makedirs(folder_path_export_latex, exist_ok=True)
        os.makedirs(folder_path_export_csv, exist_ok=True)
        os.makedirs(folder_path_export_parquet, exist_ok=True)

        knn_similarity_list = experiment_hyper_parameters.knn_similarity_types

        results_all = []
        for group in groups_recs:
            for rec in group:
                if isinstance(rec, commons.RecommenderBaseline):
                    results = _print_collaborative_filtering_metrics(
                        recommender_baseline=rec,
                        experiment_benchmark=experiment_benchmark,
                        experiment_hyper_parameters=experiment_hyper_parameters,
                        num_test_users=num_test_users,
                        accuracy_metrics_list=ACCURACY_METRICS_LIST,
                        beyond_accuracy_metrics_list=BEYOND_ACCURACY_METRICS_LIST,
                        all_metrics_list=ALL_METRICS_LIST,
                        cutoffs_list=RESULT_EXPORT_CUTOFFS,
                        knn_similarity_list=knn_similarity_list,
                        export_experiments_folder_path=folder_path_export_latex,
                    )

                elif rec in [
                    commons.RecommenderImpressions.CYCLING,
                    commons.RecommenderImpressions.IMPRESSIONS_DISCOUNTING,
                    commons.RecommenderImpressions.USER_WEIGHTED_USER_PROFILE,
                    commons.RecommenderImpressions.ITEM_WEIGHTED_USER_PROFILE,
                ]:
                    rec_baseline = cast(commons.RecommenderBaseline, group[0])

                    results = _print_plugin_impression_aware_metrics(
                        recommender_baseline=rec_baseline,
                        recommender_plugin=rec,
                        experiment_benchmark=experiment_benchmark,
                        experiment_hyper_parameters=experiment_hyper_parameters,
                        num_test_users=num_test_users,
                        accuracy_metrics_list=ACCURACY_METRICS_LIST,
                        beyond_accuracy_metrics_list=BEYOND_ACCURACY_METRICS_LIST,
                        all_metrics_list=ALL_METRICS_LIST,
                        cutoffs_list=RESULT_EXPORT_CUTOFFS,
                        knn_similarity_list=knn_similarity_list,
                        export_experiments_folder_path=folder_path_export_latex,
                    )

                elif isinstance(rec, commons.RecommenderImpressions):
                    results = _print_pure_impression_aware_metrics(
                        recommender_impressions=rec,
                        experiment_benchmark=experiment_benchmark,
                        experiment_hyper_parameters=experiment_hyper_parameters,
                        num_test_users=num_test_users,
                        accuracy_metrics_list=ACCURACY_METRICS_LIST,
                        beyond_accuracy_metrics_list=BEYOND_ACCURACY_METRICS_LIST,
                        all_metrics_list=ALL_METRICS_LIST,
                        cutoffs_list=RESULT_EXPORT_CUTOFFS,
                        knn_similarity_list=knn_similarity_list,
                        export_experiments_folder_path=folder_path_export_latex,
                    )

                else:
                    continue

                results_all.append(results)

        _results_to_pandas(
            dfs=[res.df_results for res in results_all],
            results_name="accuracy-metrics",
            folder_path_latex=folder_path_export_latex,
            folder_path_csv=folder_path_export_csv,
            folder_path_parquet=folder_path_export_parquet,
        )

        _results_to_pandas(
            dfs=[res.df_times for res in results_all],
            results_name="times",
            folder_path_latex=folder_path_export_latex,
            folder_path_csv=folder_path_export_csv,
            folder_path_parquet=folder_path_export_parquet,
        )

        _results_to_pandas(
            dfs=[res.df_hyper_params for res in results_all],
            results_name="hyper-parameters",
            folder_path_latex=folder_path_export_latex,
            folder_path_csv=folder_path_export_csv,
            folder_path_parquet=folder_path_export_parquet,
        )

        printed_experiments.add((benchmark, hyper_parameters))

    logger.info(
        f"Successfully finished exporting accuracy and beyond-accuracy results to LaTeX"
    )


def process_results(
    results_interface: tuple[
        list[commons.Benchmarks],
        list[commons.EHyperParameterTuningParameters],
        list[
            Union[
                commons.RecommenderBaseline,
                commons.RecommenderImpressions,
                tuple[commons.RecommenderBaseline, commons.RecommenderImpressions],
            ]
        ],
    ],
) -> None:
    """
    Public method that exports into CSV and LaTeX tables the evaluation metrics, hyper-parameters, and times.
    """
    printed_experiments: set[
        tuple[commons.Benchmarks, commons.EHyperParameterTuningParameters]
    ] = set()

    benchmarks, hyper_parameters, recommenders = results_interface

    for benchmark, hyper_params in itertools.product(benchmarks, hyper_parameters):
        experiment_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[benchmark]
        experiment_hyper_parameters = (
            commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[hyper_params]
        )

        data_reader = commons.get_reader_from_benchmark(
            benchmark_config=experiment_benchmark.config,
            benchmark=experiment_benchmark.benchmark,
        )

        dataset = data_reader.dataset
        interaction_data_splits = dataset.get_urm_splits(
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy,
        )

        urm_test = interaction_data_splits.sp_urm_test
        num_test_users = cast(int, np.sum(np.ediff1d(urm_test.indptr) >= 1))

        folder_path_export_latex = DIR_LATEX_RESULTS.format(
            benchmark=experiment_benchmark.benchmark.value,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
        )
        folder_path_export_csv = DIR_CSV_RESULTS.format(
            benchmark=experiment_benchmark.benchmark.value,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
        )
        folder_path_export_parquet = DIR_PARQUET_RESULTS.format(
            benchmark=experiment_benchmark.benchmark.value,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
        )

        os.makedirs(folder_path_export_latex, exist_ok=True)
        os.makedirs(folder_path_export_csv, exist_ok=True)
        os.makedirs(folder_path_export_parquet, exist_ok=True)

        knn_similarity_list = experiment_hyper_parameters.knn_similarity_types

        results_all = []
        for rec in recommenders:
            results = None

            if isinstance(rec, commons.RecommenderBaseline):
                results = _print_collaborative_filtering_metrics(
                    recommender_baseline=rec,
                    experiment_benchmark=experiment_benchmark,
                    experiment_hyper_parameters=experiment_hyper_parameters,
                    num_test_users=num_test_users,
                    accuracy_metrics_list=ACCURACY_METRICS_LIST,
                    beyond_accuracy_metrics_list=BEYOND_ACCURACY_METRICS_LIST,
                    all_metrics_list=ALL_METRICS_LIST,
                    cutoffs_list=RESULT_EXPORT_CUTOFFS,
                    knn_similarity_list=knn_similarity_list,
                    export_experiments_folder_path=folder_path_export_latex,
                )

            elif isinstance(rec, commons.RecommenderImpressions):
                if rec in [
                    commons.RecommenderImpressions.P3_ALPHA_ONLY_IMPRESSIONS,
                    commons.RecommenderImpressions.P3_ALPHA_DIRECTED_INTERACTIONS_IMPRESSIONS,
                    commons.RecommenderImpressions.RP3_BETA_ONLY_IMPRESSIONS,
                    commons.RecommenderImpressions.RP3_BETA_DIRECTED_INTERACTIONS_IMPRESSIONS,
                ]:
                    results = _print_pure_impression_aware_metrics(
                        recommender_impressions=rec,
                        experiment_benchmark=experiment_benchmark,
                        experiment_hyper_parameters=experiment_hyper_parameters,
                        num_test_users=num_test_users,
                        accuracy_metrics_list=ACCURACY_METRICS_LIST,
                        beyond_accuracy_metrics_list=BEYOND_ACCURACY_METRICS_LIST,
                        all_metrics_list=ALL_METRICS_LIST,
                        cutoffs_list=RESULT_EXPORT_CUTOFFS,
                        knn_similarity_list=knn_similarity_list,
                        export_experiments_folder_path=folder_path_export_latex,
                    )

                if rec in [
                    commons.RecommenderImpressions.P3_ALPHA_ONLY_IMPRESSIONS_FREQUENCY,
                    commons.RecommenderImpressions.P3_ALPHA_DIRECTED_INTERACTIONS_IMPRESSIONS_FREQUENCY,
                    commons.RecommenderImpressions.RP3_BETA_ONLY_IMPRESSIONS_FREQUENCY,
                    commons.RecommenderImpressions.RP3_BETA_DIRECTED_INTERACTIONS_IMPRESSIONS_FREQUENCY,
                ]:
                    results = _print_frequency_impression_aware_metrics(
                        recommender_impressions=rec,
                        experiment_benchmark=experiment_benchmark,
                        experiment_hyper_parameters=experiment_hyper_parameters,
                        num_test_users=num_test_users,
                        accuracy_metrics_list=ACCURACY_METRICS_LIST,
                        beyond_accuracy_metrics_list=BEYOND_ACCURACY_METRICS_LIST,
                        all_metrics_list=ALL_METRICS_LIST,
                        cutoffs_list=RESULT_EXPORT_CUTOFFS,
                        knn_similarity_list=knn_similarity_list,
                        export_experiments_folder_path=folder_path_export_latex,
                    )

            elif isinstance(rec, tuple):
                rec_baseline: commons.RecommenderBaseline
                rec_impressions: commons.RecommenderImpressions
                rec_baseline, rec_impressions = rec

                if rec_impressions in [
                    commons.RecommenderImpressions.HARD_FREQUENCY_CAPPING,
                    commons.RecommenderImpressions.CYCLING,
                    commons.RecommenderImpressions.IMPRESSIONS_DISCOUNTING,
                    commons.RecommenderImpressions.USER_WEIGHTED_USER_PROFILE,
                    commons.RecommenderImpressions.ITEM_WEIGHTED_USER_PROFILE,
                ]:
                    results = _print_plugin_impression_aware_metrics(
                        recommender_baseline=rec_baseline,
                        recommender_plugin=rec_impressions,
                        experiment_benchmark=experiment_benchmark,
                        experiment_hyper_parameters=experiment_hyper_parameters,
                        num_test_users=num_test_users,
                        accuracy_metrics_list=ACCURACY_METRICS_LIST,
                        beyond_accuracy_metrics_list=BEYOND_ACCURACY_METRICS_LIST,
                        all_metrics_list=ALL_METRICS_LIST,
                        cutoffs_list=RESULT_EXPORT_CUTOFFS,
                        knn_similarity_list=knn_similarity_list,
                        export_experiments_folder_path=folder_path_export_latex,
                    )

            else:
                continue

            if results is not None:
                results_all.append(results)

        _process_results_dataframe(
            dfs=[res.df_results for res in results_all],
            results_name="accuracy-metrics",
            benchmark=benchmark,
            folder_path_csv=folder_path_export_csv,
            folder_path_parquet=folder_path_export_parquet,
        )

        _process_results_dataframe(
            dfs=[res.df_times for res in results_all],
            results_name="times",
            benchmark=benchmark,
            folder_path_csv=folder_path_export_csv,
            folder_path_parquet=folder_path_export_parquet,
        )

        _process_results_dataframe(
            dfs=[res.df_hyper_params for res in results_all],
            results_name="hyper-parameters",
            benchmark=benchmark,
            folder_path_csv=folder_path_export_csv,
            folder_path_parquet=folder_path_export_parquet,
        )

        printed_experiments.add((benchmark, hyper_params))

        logger.info(
            f"Successfully finished exporting accuracy and beyond-accuracy results to LaTeX"
        )


def export_evaluation_results(
    benchmarks: list[commons.Benchmarks],
    hyper_parameters: list[commons.EHyperParameterTuningParameters],
) -> None:
    results_accuracy_metrics: list[pd.DataFrame] = []
    results_times: list[pd.DataFrame] = []
    results_hyper_parameters: list[pd.DataFrame] = []

    for benchmark, hyper_parameter in itertools.product(benchmarks, hyper_parameters):
        experiment_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[benchmark]
        experiment_hyper_parameters = (
            commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[hyper_parameter]
        )

        folder_path_results_to_load = DIR_PARQUET_RESULTS.format(
            benchmark=experiment_benchmark.benchmark.value,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
        )

        df_results_accuracy = pd.read_parquet(
            path=os.path.join(
                folder_path_results_to_load, "accuracy-metrics-non_processed.parquet"
            ),
            engine="pyarrow",
        )
        df_results_times = pd.read_parquet(
            path=os.path.join(
                folder_path_results_to_load, "times-non_processed.parquet"
            ),
            engine="pyarrow",
        )
        df_results_hyper_parameters = pd.read_parquet(
            path=os.path.join(
                folder_path_results_to_load, "hyper-parameters-non_processed.parquet"
            ),
            engine="pyarrow",
        )

        results_accuracy_metrics.append(df_results_accuracy)
        results_times.append(df_results_times)
        results_hyper_parameters.append(df_results_hyper_parameters)

    folder_path_results_to_export = DIR_RESULTS_TO_EXPORT
    os.makedirs(folder_path_results_to_export, exist_ok=True)

    df_results_accuracy = pd.concat(
        objs=results_accuracy_metrics,
        axis=0,
        ignore_index=True,  # The index should be numeric and have no special meaning.
    )
    df_results_times = pd.concat(
        objs=results_times,
        axis=0,
        ignore_index=True,  # The index should be numeric and have no special meaning.
    )
    df_results_hyper_parameters = pd.concat(
        objs=results_hyper_parameters,
        axis=0,
        ignore_index=True,  # The index should be numeric and have no special meaning.
    )

    _export_results_accuracy_metrics(
        benchmarks=benchmarks,
        df_results=df_results_accuracy,
        folder_path_csv=folder_path_results_to_export,
        cutoffs=[20],
        metrics=[
            "NDCG",
            "PRECISION",
            "RECALL",
            "F1",
            "COVERAGE_ITEM",
            "DIVERSITY_GINI",
            "NOVELTY",
        ],
        order="cutoff",
        results_name="one_cutoff-all_metrics",
    )

    _export_results_accuracy_metrics(
        benchmarks=benchmarks,
        df_results=df_results_accuracy,
        folder_path_csv=folder_path_results_to_export,
        cutoffs=[5, 10, 20, 50, 100],
        metrics=["NDCG", "COVERAGE_ITEM"],
        order="metric",
        results_name="all_cutoffs-two_metrics",
    )

    _export_results_time(
        benchmarks=benchmarks,
        df_results=df_results_times,
        folder_path_csv=folder_path_results_to_export,
        results_name="all",
    )

    _export_results_hyper_parameters(
        benchmarks=benchmarks,
        df_results=df_results_hyper_parameters,
        folder_path_csv=folder_path_results_to_export,
        results_name="all",
    )
