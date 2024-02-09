import logging
import os
import uuid

import impressions_evaluation.experiments.commons as commons
from impressions_evaluation.experiments.baselines import DIR_TRAINED_MODELS_BASELINES
from impressions_evaluation.experiments.impression_aware.re_ranking import (
    DIR_TRAINED_MODELS_RE_RANKING,
)
from impressions_evaluation.experiments.impression_aware.user_profiles import (
    DIR_TRAINED_MODELS_USER_PROFILES,
)

from Recommenders.BaseSimilarityMatrixRecommender import (
    BaseItemSimilarityMatrixRecommender,
    BaseUserSimilarityMatrixRecommender,
)
from recsys_framework_extensions.dask import DaskInterface

from impressions_evaluation.experiments.print_results import (
    DIR_RESULTS_MODEL_EVALUATION,
)

logger = logging.getLogger(__name__)

####################################################################################################
####################################################################################################
#                                FOLDERS VARIABLES                            #
####################################################################################################
####################################################################################################
DIR_STATISTICAL_TESTS = os.path.join(
    commons.DIR_RESULTS_EXPORT,
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
    experiment_baseline_similarity: str,
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
        experiment_case_statistical_test.recommender
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

    folder_path_recommender_impressions_re_ranking = (
        DIR_TRAINED_MODELS_RE_RANKING.format(
            benchmark=experiment_benchmark.benchmark.value,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
        )
    )
    folder_path_recommender_impressions_user_profiles = (
        DIR_TRAINED_MODELS_USER_PROFILES.format(
            benchmark=experiment_benchmark.benchmark.value,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
        )
    )
    folder_path_export_statistical_tests = DIR_STATISTICAL_TESTS.format(
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
        similarity=experiment_baseline_similarity,
    )

    if recommender_trained_baseline is None:
        return

    recommender_baseline = recommender_trained_baseline
    recommender_baseline_name = (
        f"{recommender_trained_baseline.RECOMMENDER_NAME}_{file_name_postfix}"
    )
    recommender_baseline_folder = folder_path_recommender_baseline

    recommenders_impressions = []
    recommenders_impressions_names = []
    recommenders_impressions_folders = []

    recommender_trained_impressions_re_ranking_cycling = commons.load_recommender_trained_impressions(
        recommender_class_impressions=experiment_recommender_impressions_reranking_cycling.recommender,
        folder_path=folder_path_recommender_impressions_re_ranking,
        file_name_postfix=file_name_postfix,
        urm_train=urm_train.copy(),
        uim_train=uim_train.copy(),
        uim_frequency=impressions_feature_frequency_train_validation.copy(),
        uim_position=impressions_feature_position_train_validation.copy(),
        uim_timestamp=impressions_feature_timestamp_train_validation.copy(),
        uim_last_seen=impressions_feature_last_seen_train_validation.copy(),
        recommender_baseline=recommender_baseline,
    )

    if recommender_trained_impressions_re_ranking_cycling is not None:
        recommenders_impressions.append(
            recommender_trained_impressions_re_ranking_cycling
        )
        recommenders_impressions_names.append(
            f"{recommender_trained_impressions_re_ranking_cycling.RECOMMENDER_NAME}_{file_name_postfix}",
        )
        recommenders_impressions_folders.append(
            folder_path_recommender_impressions_re_ranking,
        )

    recommender_trained_impressions_re_ranking_impressions_discounting = commons.load_recommender_trained_impressions(
        recommender_class_impressions=experiment_recommender_impressions_reranking_impressions_discounting.recommender,
        folder_path=folder_path_recommender_impressions_re_ranking,
        file_name_postfix=file_name_postfix,
        urm_train=urm_train.copy(),
        uim_train=uim_train.copy(),
        uim_frequency=impressions_feature_frequency_train_validation.copy(),
        uim_position=impressions_feature_position_train_validation.copy(),
        uim_timestamp=impressions_feature_timestamp_train_validation.copy(),
        uim_last_seen=impressions_feature_last_seen_train_validation.copy(),
        recommender_baseline=recommender_baseline,
    )

    if recommender_trained_impressions_re_ranking_impressions_discounting is not None:
        recommenders_impressions.append(
            recommender_trained_impressions_re_ranking_impressions_discounting
        )
        recommenders_impressions_names.append(
            f"{recommender_trained_impressions_re_ranking_impressions_discounting.RECOMMENDER_NAME}_{file_name_postfix}",
        )
        recommenders_impressions_folders.append(
            folder_path_recommender_impressions_re_ranking,
        )

    recommender_has_user_similarity = isinstance(
        recommender_trained_baseline, BaseUserSimilarityMatrixRecommender
    ) or isinstance(recommender_trained_folded, BaseUserSimilarityMatrixRecommender)
    recommender_has_item_similarity = isinstance(
        recommender_trained_baseline, BaseItemSimilarityMatrixRecommender
    ) or isinstance(recommender_trained_folded, BaseItemSimilarityMatrixRecommender)

    if recommender_has_user_similarity:
        recommender_class_impressions = experiment_recommender_impressions_profiles_user
    elif recommender_has_item_similarity:
        recommender_class_impressions = experiment_recommender_impressions_profiles_item
    else:
        return

    recommender_trained_impressions_user_profiles = (
        commons.load_recommender_trained_impressions(
            recommender_class_impressions=recommender_class_impressions.recommender,
            folder_path=folder_path_recommender_impressions_user_profiles,
            file_name_postfix=file_name_postfix,
            urm_train=urm_train.copy(),
            uim_train=uim_train.copy(),
            uim_frequency=impressions_feature_frequency_train_validation.copy(),
            uim_position=impressions_feature_position_train_validation.copy(),
            uim_timestamp=impressions_feature_timestamp_train_validation.copy(),
            uim_last_seen=impressions_feature_last_seen_train_validation.copy(),
            recommender_baseline=recommender_baseline,
        )
    )

    if recommender_trained_impressions_user_profiles is not None:
        recommenders_impressions.append(recommender_trained_impressions_user_profiles)
        recommenders_impressions_names.append(
            f"{recommender_trained_impressions_user_profiles.RECOMMENDER_NAME}_{file_name_postfix}",
        )
        recommenders_impressions_folders.append(
            folder_path_recommender_impressions_user_profiles,
        )

    if len(recommenders_impressions) == 0:
        # We require a recommender that is already optimized.
        logger.warning(
            f"Early-skipping on {_compute_statistical_test_on_users.__name__}."
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
            experiment_baseline_similarity=similarity,
        )
