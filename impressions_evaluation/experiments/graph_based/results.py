import logging
import os
from typing import cast, Union, Sequence

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
    DIR_ACCURACY_METRICS_BASELINES_LATEX,
    DIR_CSV_RESULTS,
    DIR_PARQUET_RESULTS,
    ACCURACY_METRICS_LIST,
    BEYOND_ACCURACY_METRICS_LIST,
    ALL_METRICS_LIST,
    RESULT_EXPORT_CUTOFFS,
)

logger = logging.getLogger(__file__)


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

        printed_experiments.add((benchmark, hyper_parameters))

        logger.info(
            f"Successfully finished exporting accuracy and beyond-accuracy results to LaTeX"
        )
