import logging
import os

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

commons.FOLDERS.add(DIR_STATISTICAL_TESTS)


####################################################################################################
####################################################################################################
#                    Statistical Tests                              #
####################################################################################################
####################################################################################################
def _compute_statistical_test_on_users(
    experiment_case_statistical_test: commons.ExperimentCaseStatisticalTest,
) -> None:
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
        return

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
        return

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

    evaluators.test.compute_recommenders_statistical_tests(
        recommender_baseline=recommender_baseline,
        recommender_baseline_name=recommender_baseline_name,
        recommender_baseline_folder=recommender_baseline_folder,
        recommender_others=recommenders_impressions,
        recommender_others_names=recommenders_impressions_names,
        recommender_others_folders=recommenders_impressions_folders,
        folder_export_results=folder_path_export_statistical_tests,
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
