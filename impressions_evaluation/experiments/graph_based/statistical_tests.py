import itertools
import logging
import os
from typing import Union

import pandas as pd

from impressions_evaluation.experiments import commons
from impressions_evaluation.experiments.baselines import DIR_TRAINED_MODELS_BASELINES
from impressions_evaluation.experiments.graph_based import (
    DIR_TRAINED_MODELS_IMPRESSION_AWARE,
)


logger = logging.getLogger(__name__)

####################################################################################################
####################################################################################################
#                                FOLDERS VARIABLES                            #
####################################################################################################
####################################################################################################
DIR_STATISTICAL_TESTS = os.path.join(
    commons.DIR_RESULTS_EXPORT,
    "{script_name}",
    "statistical_tests",
    "{benchmark}",
    "{evaluation_strategy}",
    "",
)

DIR_STATISTICAL_TESTS_EXPORT = os.path.join(
    commons.DIR_RESULTS_EXPORT, "{script_name}", "statistical_tests", "",
)

commons.FOLDERS.add(DIR_STATISTICAL_TESTS)
commons.FOLDERS.add(DIR_STATISTICAL_TESTS_EXPORT)


####################################################################################################
####################################################################################################
#                    Statistical Tests                              #
####################################################################################################
####################################################################################################
def _compute_statistical_test_on_users(
    experiment_case_statistical_test: commons.ExperimentCaseStatisticalTest,
) -> Union[tuple[pd.DataFrame, pd.DataFrame], tuple[None, None]]:
    experiment_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
        experiment_case_statistical_test.benchmark
    ]
    experiment_hyper_parameters = (
        commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
            experiment_case_statistical_test.hyper_parameter_tuning_parameters
        ]
    )

    experiment_recommender_baseline = commons.MAPPER_AVAILABLE_RECOMMENDERS[
        experiment_case_statistical_test.recommender_baseline
    ]
    experiment_recommenders_impressions = [
        commons.MAPPER_AVAILABLE_RECOMMENDERS[recommender_impressions]
        for recommender_impressions in experiment_case_statistical_test.recommenders_impressions
    ]

    benchmark_reader = commons.get_reader_from_benchmark(
        benchmark_config=experiment_benchmark.config,
        benchmark=experiment_benchmark.benchmark,
    )

    dataset = benchmark_reader.dataset

    interactions_data_splits = dataset.get_urm_splits(
        evaluation_strategy=experiment_hyper_parameters.evaluation_strategy
    )
    impressions_data_splits = dataset.get_uim_splits(
        evaluation_strategy=experiment_hyper_parameters.evaluation_strategy,
    )

    # Only load features of the train+validation split.
    impressions_feature_frequency_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_benchmark.benchmark,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_FREQUENCY,
            impressions_feature_column=commons.ImpressionsFeatureColumnsFrequency.FREQUENCY,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    impressions_feature_position_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_benchmark.benchmark,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_POSITION,
            impressions_feature_column=commons.ImpressionsFeatureColumnsPosition.POSITION,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    impressions_feature_timestamp_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_benchmark.benchmark,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_TIMESTAMP,
            impressions_feature_column=commons.ImpressionsFeatureColumnsTimestamp.TIMESTAMP,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    if commons.Benchmarks.FINNNoSlates == experiment_benchmark.benchmark:
        impressions_feature_last_seen_train_validation = dataset.sparse_matrix_impression_feature(
            feature=commons.get_feature_key_by_benchmark(
                benchmark=experiment_benchmark.benchmark,
                evaluation_strategy=experiment_hyper_parameters.evaluation_strategy,
                impressions_feature=commons.ImpressionsFeatures.USER_ITEM_LAST_SEEN,
                impressions_feature_column=commons.ImpressionsFeatureColumnsLastSeen.EUCLIDEAN,
                impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
            )
        )

    else:
        impressions_feature_last_seen_train_validation = dataset.sparse_matrix_impression_feature(
            feature=commons.get_feature_key_by_benchmark(
                benchmark=experiment_benchmark.benchmark,
                evaluation_strategy=experiment_hyper_parameters.evaluation_strategy,
                impressions_feature=commons.ImpressionsFeatures.USER_ITEM_LAST_SEEN,
                impressions_feature_column=commons.ImpressionsFeatureColumnsLastSeen.TOTAL_DAYS,
                impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
            )
        )

    folder_path_recommender_baseline = DIR_TRAINED_MODELS_BASELINES.format(
        benchmark=experiment_benchmark.benchmark.value,
        evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
    )
    folder_path_recommender_impression_aware = (
        DIR_TRAINED_MODELS_IMPRESSION_AWARE.format(
            benchmark=experiment_benchmark.benchmark.value,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
        )
    )
    folder_path_export_statistical_tests = DIR_STATISTICAL_TESTS.format(
        script_name=experiment_case_statistical_test.script_name,
        benchmark=experiment_benchmark.benchmark.value,
        evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
    )

    file_name_postfix = commons.MAPPER_FILE_NAME_POSTFIX[
        commons.TrainedRecommenderType.TRAIN_VALIDATION
    ]

    urm_train = interactions_data_splits.sp_urm_train_validation
    uim_train = impressions_data_splits.sp_uim_train_validation

    recommender_trained_baseline = commons.load_recommender_trained_baseline(
        recommender_baseline_class=experiment_recommender_baseline.recommender,
        folder_path=folder_path_recommender_baseline,
        file_name_postfix=file_name_postfix,
        urm_train=urm_train.copy(),
        similarity=None,
    )

    if recommender_trained_baseline is None:
        return None, None

    recommender_baseline = recommender_trained_baseline
    recommender_baseline_name = (
        f"{recommender_trained_baseline.RECOMMENDER_NAME}_{file_name_postfix}"
    )
    recommender_baseline_folder = folder_path_recommender_baseline

    recommenders_impressions = [
        commons.load_recommender_trained_impressions(
            recommender_class_impressions=experiment_recommender.recommender,
            folder_path=folder_path_recommender_impression_aware,
            file_name_postfix=file_name_postfix,
            urm_train=urm_train.copy(),
            uim_train=uim_train.copy(),
            uim_frequency=impressions_feature_frequency_train_validation.copy(),
            uim_position=impressions_feature_position_train_validation.copy(),
            uim_timestamp=impressions_feature_timestamp_train_validation.copy(),
            uim_last_seen=impressions_feature_last_seen_train_validation.copy(),
            recommender_baseline=recommender_baseline,
        )
        for experiment_recommender in experiment_recommenders_impressions
    ]

    recommenders_impressions = [
        rec_imp for rec_imp in recommenders_impressions if rec_imp is not None
    ]

    recommenders_impressions_names = [
        f"{rec_imp.RECOMMENDER_NAME}_{file_name_postfix}"
        for rec_imp in recommenders_impressions
        if rec_imp is not None
    ]

    recommenders_impressions_folders = [
        folder_path_recommender_impression_aware
        for rec_imp in recommenders_impressions
        if rec_imp is not None
    ]

    if len(recommenders_impressions) == 0:
        # We require a recommender that is already optimized.
        logger.warning(
            "Early-skipping on %(recommender_name)s.",
            {"recommender_name": _compute_statistical_test_on_users.__name__},
        )
        return None, None

    import random
    import numpy as np

    random.seed(experiment_hyper_parameters.reproducibility_seed)
    np.random.seed(experiment_hyper_parameters.reproducibility_seed)

    evaluators = commons.get_evaluators(
        data_splits=interactions_data_splits,
        experiment_hyper_parameter_tuning_parameters=experiment_hyper_parameters,
    )

    logger.debug(
        "Running statistical tests with the following parameters: %s",
        {
            "recommender_baseline": recommender_baseline,
            "recommender_baseline_name": recommender_baseline_name,
            "recommender_baseline_folder": recommender_baseline_folder,
            "recommender_others": recommenders_impressions,
            "recommender_others_names": recommenders_impressions_names,
            "recommender_others_folders": recommenders_impressions_folders,
            "folder_export_results": folder_path_export_statistical_tests,
        },
    )

    (
        df_statistical_tests_groupwise,
        df_statistical_tests_pairwise,
    ) = evaluators.test.compute_recommenders_statistical_tests(
        dataset=experiment_benchmark.benchmark.value,
        recommender_baseline=recommender_baseline,
        recommender_baseline_name=recommender_baseline_name,
        recommender_baseline_folder=recommender_baseline_folder,
        recommender_others=recommenders_impressions,
        recommender_others_names=recommenders_impressions_names,
        recommender_others_folders=recommenders_impressions_folders,
        folder_export_results=folder_path_export_statistical_tests,
    )

    return df_statistical_tests_groupwise, df_statistical_tests_pairwise


def _export_pairwise_statistical_tests(
    *,
    df_results_pairwise: pd.DataFrame,
    cutoffs: list[int],
    metrics: list[str],
    folder_path_results_to_export: str,
    results_name: str,
) -> None:
    # We are only interested in the statistical test with a bonferroni correction to adjust for multiple pair-wise comparisons.
    statistical_tests = [
        "wilcoxon",
        "wilcoxon_zsplit",
        "bonferroni-wilcoxon",
        "bonferroni-wilcoxon_zsplit",
    ]
    # We want to know only if impression-aware recommenders are better than baselines.
    alternative_hipotheses = ["greater"]
    alpha_significance_levels = [0.05]
    inside_columns = ["p_value"]

    for (
        statistical_test,
        alternative_hipothesis,
        alpha,
        inside_col,
    ) in itertools.product(
        statistical_tests,
        alternative_hipotheses,
        alpha_significance_levels,
        inside_columns,
    ):
        columns_to_test = [
            ("dataset", "", "", "", "", ""),
            ("recommender_base", "", "", "", "", ""),
            ("recommender_other", "", "", "", "", ""),
        ] + [
            (
                str(cutoff),
                str(metric),
                str(statistical_test),
                str(alternative_hipothesis),
                str(alpha),
                str(inside_col),
            )
            for cutoff, metric in itertools.product(cutoffs, metrics)
        ]

        df_to_export = df_results_pairwise[columns_to_test].copy()
        df_to_export[("recommender_base", "", "", "", "", "")] = (
            df_to_export[("recommender_base", "", "", "", "", "")]
            .str.replace("_best_model_last", "")
            .str.replace("Recommender", "")
        )

        df_to_export[("recommender_other", "", "", "", "", "")] = (
            df_to_export[("recommender_other", "", "", "", "", "")]
            .str.replace("_best_model_last", "")
            .str.replace("Recommender", "")
            .str.replace("P3Alpha", "")
            .str.replace("RP3Beta", "")
            .str.replace("ImpressionsProfileWithFrequency", "IP-F")
            .str.replace("ImpressionsProfile", "IP-E")
            .str.replace("ImpressionsDirectedWithFrequency", "DG-F")
            .str.replace("ImpressionsDirected", "DG-E")
        )

        # Removes the multi-level columns: <statistical_test>, <alternative_hipothesis>, <alpha>, "p_value"
        df_to_export = df_to_export.droplevel([2, 3, 4, 5], axis="columns")

        df_to_export.to_csv(
            os.path.join(
                folder_path_results_to_export,
                f"table-statistical_test_pvalues-{results_name}-{statistical_test}-{alternative_hipothesis}.csv",
            ),
            index=False,
            sep=";",
        )


def export_statistical_tests(
    experiment_cases_statistical_tests_interface: commons.ExperimentCasesStatisticalTestInterface,
) -> None:
    list_df_results_groupwise = []
    list_df_results_pairwise = []

    column_dataset = ("dataset", "", "", "", "", "")

    for benchmark, hyper_parameters in itertools.product(
        experiment_cases_statistical_tests_interface.to_use_benchmarks,
        experiment_cases_statistical_tests_interface.to_use_hyper_parameter_tuning_parameters,
    ):
        for (
            experiment_case_statistical_test
        ) in experiment_cases_statistical_tests_interface.experiment_cases:
            if (
                benchmark != experiment_case_statistical_test.benchmark
                or hyper_parameters
                != experiment_case_statistical_test.hyper_parameter_tuning_parameters
            ):
                continue

            (
                df_statistical_tests_groupwise,
                df_statistical_tests_pairwise,
            ) = _compute_statistical_test_on_users(
                experiment_case_statistical_test=experiment_case_statistical_test,
            )

            if df_statistical_tests_groupwise is not None:
                if column_dataset not in df_statistical_tests_groupwise.columns:
                    logger.warning("Adding dataset column to groupwise df.")
                    df_statistical_tests_groupwise[column_dataset] = benchmark.value

                list_df_results_groupwise.append(df_statistical_tests_groupwise)

            if df_statistical_tests_pairwise is not None:
                if column_dataset not in df_statistical_tests_pairwise.columns:
                    logger.warning("Adding dataset column to pairwise df.")
                    df_statistical_tests_pairwise[column_dataset] = benchmark.value

                list_df_results_pairwise.append(df_statistical_tests_pairwise)

    df_results_groupwise = pd.concat(
        objs=list_df_results_groupwise,
        axis="index",
        ignore_index=True,
    )

    df_results_pairwise = pd.concat(
        objs=list_df_results_pairwise,
        axis="index",
        ignore_index=True,
    )

    folder_path_results_to_export = DIR_STATISTICAL_TESTS_EXPORT.format(
        script_name=experiment_cases_statistical_tests_interface.to_use_script_name,
    )
    os.makedirs(folder_path_results_to_export, exist_ok=True)

    _export_pairwise_statistical_tests(
        df_results_pairwise=df_results_pairwise,
        cutoffs=[20],
        metrics=["NDCG", "PRECISION", "RECALL", "F1"],
        folder_path_results_to_export=folder_path_results_to_export,
        results_name="one_cutoff-all_metrics",
    )

    _export_pairwise_statistical_tests(
        df_results_pairwise=df_results_pairwise,
        cutoffs=[5, 10, 20, 50, 100],
        metrics=["NDCG"],
        folder_path_results_to_export=folder_path_results_to_export,
        results_name="all_cutoffs-one_metric",
    )


def compute_statistical_tests(
    experiment_cases_statistical_tests_interface: commons.ExperimentCasesStatisticalTestInterface,
) -> None:
    """
    Public method runs the statistical tests on the recommendations.
    """
    # First compute baselines.
    for (
        experiment_case_statistical_test
    ) in experiment_cases_statistical_tests_interface.experiment_cases:
        _compute_statistical_test_on_users(
            experiment_case_statistical_test=experiment_case_statistical_test,
        )
