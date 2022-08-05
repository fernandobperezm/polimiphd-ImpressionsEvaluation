import os
import uuid
from typing import Optional

from recsys_framework_extensions.dask import DaskInterface
from recsys_framework_extensions.logging import get_logger

import impressions_evaluation.experiments.commons as commons
from impressions_evaluation.experiments.baselines import DIR_TRAINED_MODELS_BASELINES as FOLDER_BASE_BASELINES
from impressions_evaluation.experiments.baselines import DIR_TRAINED_MODELS_BASELINES as \
    FOLDER_HYPER_PARAMETER_TUNING_BASELINES
from impressions_evaluation.experiments.time_aware import DIR_TRAINED_MODELS_TIME_AWARE as FOLDER_BASE_HEURISTICS
from impressions_evaluation.experiments.time_aware import DIR_TRAINED_MODELS_TIME_AWARE as \
    FOLDER_HYPER_PARAMETER_TUNING_IMPRESSIONS_HEURISTICS
from impressions_evaluation.experiments.re_ranking import DIR_TRAINED_MODELS_RE_RANKING as FOLDER_BASE_RE_RANKING
from impressions_evaluation.experiments.re_ranking import DIR_TRAINED_MODELS_RE_RANKING as \
    FOLDER_HYPER_PARAMETER_TUNING_IMPRESSIONS_RE_RANKING
from impressions_evaluation.experiments.user_profiles import DIR_TRAINED_MODELS_USER_PROFILES as FOLDER_BASE_USER_PROFILES
from impressions_evaluation.experiments.user_profiles import DIR_TRAINED_MODELS_USER_PROFILES as \
    FOLDER_HYPER_PARAMETER_TUNING_IMPRESSIONS_USER_PROFILES

from impressions_evaluation.impression_recommenders.user_profile.folding import FoldedMatrixFactorizationRecommender


logger = get_logger(__name__)


####################################################################################################
####################################################################################################
#                                FOLDERS VARIABLES                            #
####################################################################################################
####################################################################################################
DIR_CI_RECOMMENDER_BASELINE = os.path.join(
    FOLDER_BASE_BASELINES,
    "statistics",
    "",
)
DIR_CI_RECOMMENDER_HEURISTICS = os.path.join(
    FOLDER_BASE_HEURISTICS,
    "statistics",
    "",
)
DIR_CI_RECOMMENDER_RE_RANKING = os.path.join(
    FOLDER_BASE_RE_RANKING,
    "statistics",
    "",
)
DIR_CI_RECOMMENDER_USER_PROFILES = os.path.join(
    FOLDER_BASE_USER_PROFILES,
    "statistics",
    "",
)


commons.FOLDERS.add(DIR_CI_RECOMMENDER_BASELINE)
commons.FOLDERS.add(DIR_CI_RECOMMENDER_HEURISTICS)
commons.FOLDERS.add(DIR_CI_RECOMMENDER_RE_RANKING)
commons.FOLDERS.add(DIR_CI_RECOMMENDER_USER_PROFILES)


####################################################################################################
####################################################################################################
#                    Confidence Interval Computation                              #
####################################################################################################
####################################################################################################
def _compute_confidence_intervals_recommender_trained_baseline(
    experiment_case_baseline: commons.ExperimentCase,
    experiment_baseline_similarity: Optional[str],
) -> None:
    """
    Runs in a dask worker the hyper-parameter tuning of a re-ranking impression recommender.

    This method should not be called from outside.
    """

    experiment_baseline_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
        experiment_case_baseline.benchmark
    ]
    experiment_baseline_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
        experiment_case_baseline.recommender
    ]
    experiment_baseline_hyper_parameters = commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
        experiment_case_baseline.hyper_parameter_tuning_parameters
    ]

    assert experiment_baseline_recommender.search_hyper_parameters is not None

    benchmark_reader = commons.get_reader_from_benchmark(
        benchmark_config=experiment_baseline_benchmark.config,
        benchmark=experiment_baseline_benchmark.benchmark,
    )

    dataset = benchmark_reader.dataset

    interactions_data_splits = dataset.get_urm_splits(
        evaluation_strategy=experiment_baseline_hyper_parameters.evaluation_strategy
    )

    folder_path_recommender_baseline = FOLDER_HYPER_PARAMETER_TUNING_BASELINES.format(
        benchmark=experiment_baseline_benchmark.benchmark.value,
        evaluation_strategy=experiment_baseline_hyper_parameters.evaluation_strategy.value,
    )
    folder_path_export_confidence_intervals = DIR_CI_RECOMMENDER_BASELINE.format(
        benchmark=experiment_baseline_benchmark.benchmark.value,
        evaluation_strategy=experiment_baseline_hyper_parameters.evaluation_strategy.value,
    )

    file_name_postfix = commons.MAPPER_FILE_NAME_POSTFIX[commons.TrainedRecommenderType.TRAIN_VALIDATION]

    urm_train = interactions_data_splits.sp_urm_train_validation

    recommender_trained_baseline = commons.load_recommender_trained_baseline(
        recommender_baseline_class=experiment_baseline_recommender.recommender,
        folder_path=folder_path_recommender_baseline,
        file_name_postfix=file_name_postfix,
        urm_train=urm_train.copy(),
        similarity=experiment_baseline_similarity,
    )

    if recommender_trained_baseline is None:
        # We require a recommender that is already optimized.
        logger.warning(
            f"Early-skipping on {_compute_confidence_intervals_recommender_trained_baseline.__name__}. Could not load "
            f"trained recommenders for {experiment_baseline_recommender.recommender} with the benchmark "
            f"{experiment_baseline_benchmark.benchmark}."
        )
        return

    import random
    import numpy as np

    random.seed(experiment_baseline_hyper_parameters.reproducibility_seed)
    np.random.seed(experiment_baseline_hyper_parameters.reproducibility_seed)

    evaluators = commons.get_evaluators(
        data_splits=interactions_data_splits,
        experiment_hyper_parameter_tuning_parameters=experiment_baseline_hyper_parameters,
    )

    evaluators.test.compute_recommender_confidence_intervals(
        recommender=recommender_trained_baseline,
        recommender_name=f"{recommender_trained_baseline.RECOMMENDER_NAME}_{file_name_postfix}",
        folder_export_results=folder_path_export_confidence_intervals,
    )


def _compute_confidence_intervals_recommender_trained_folded(
    experiment_case_baseline: commons.ExperimentCase,
    experiment_baseline_similarity: Optional[str],
) -> None:
    """
    Runs in a dask worker the hyper-parameter tuning of a re-ranking impression recommender.

    This method should not be called from outside.
    """

    experiment_baseline_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
        experiment_case_baseline.benchmark
    ]
    experiment_baseline_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
        experiment_case_baseline.recommender
    ]
    experiment_baseline_hyper_parameters = commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
        experiment_case_baseline.hyper_parameter_tuning_parameters
    ]

    assert experiment_baseline_recommender.search_hyper_parameters is not None

    benchmark_reader = commons.get_reader_from_benchmark(
        benchmark_config=experiment_baseline_benchmark.config,
        benchmark=experiment_baseline_benchmark.benchmark,
    )

    dataset = benchmark_reader.dataset

    interactions_data_splits = dataset.get_urm_splits(
        evaluation_strategy=experiment_baseline_hyper_parameters.evaluation_strategy
    )

    folder_path_recommender_baseline = FOLDER_HYPER_PARAMETER_TUNING_BASELINES.format(
        benchmark=experiment_baseline_benchmark.benchmark.value,
        evaluation_strategy=experiment_baseline_hyper_parameters.evaluation_strategy.value,
    )
    folder_path_recommender_folded = FOLDER_HYPER_PARAMETER_TUNING_BASELINES.format(
        benchmark=experiment_baseline_benchmark.benchmark.value,
        evaluation_strategy=experiment_baseline_hyper_parameters.evaluation_strategy.value,
    )
    folder_path_export_confidence_intervals = DIR_CI_RECOMMENDER_BASELINE.format(
        benchmark=experiment_baseline_benchmark.benchmark.value,
        evaluation_strategy=experiment_baseline_hyper_parameters.evaluation_strategy.value,
    )

    file_name_postfix = commons.MAPPER_FILE_NAME_POSTFIX[commons.TrainedRecommenderType.TRAIN_VALIDATION]

    urm_train = interactions_data_splits.sp_urm_train_validation

    recommender_trained_baseline = commons.load_recommender_trained_baseline(
        recommender_baseline_class=experiment_baseline_recommender.recommender,
        folder_path=folder_path_recommender_baseline,
        file_name_postfix=file_name_postfix,
        urm_train=urm_train.copy(),
        similarity=experiment_baseline_similarity,
    )

    if recommender_trained_baseline is None:
        return

    recommender_trained_folded = commons.load_recommender_trained_folded(
            recommender_baseline_instance=recommender_trained_baseline,
            folder_path=folder_path_recommender_folded,
            file_name_postfix=file_name_postfix,
            urm_train=urm_train.copy()
        )

    if recommender_trained_folded is None:
        # We require a recommender that is already optimized.
        logger.warning(
            f"Early-skipping on {_compute_confidence_intervals_recommender_trained_folded.__name__}. Could not load "
            f"trained recommenders for {experiment_baseline_recommender.recommender} with the benchmark "
            f"{experiment_baseline_benchmark.benchmark}."
        )
        return

    import random
    import numpy as np

    random.seed(experiment_baseline_hyper_parameters.reproducibility_seed)
    np.random.seed(experiment_baseline_hyper_parameters.reproducibility_seed)

    evaluators = commons.get_evaluators(
        data_splits=interactions_data_splits,
        experiment_hyper_parameter_tuning_parameters=experiment_baseline_hyper_parameters,
    )

    evaluators.test.compute_recommender_confidence_intervals(
        recommender=recommender_trained_folded,
        recommender_name=f"{recommender_trained_folded.RECOMMENDER_NAME}_{file_name_postfix}",
        folder_export_results=folder_path_export_confidence_intervals,
    )


def _compute_confidence_intervals_recommender_trained_impressions_heuristics(
    experiment_case_impressions: commons.ExperimentCase,
) -> None:
    """
    Runs in a dask worker the hyper-parameter tuning of a re-ranking impression recommender.

    This method should not be called from outside.
    """

    experiment_impressions_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
        experiment_case_impressions.benchmark
    ]
    experiment_impressions_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
        experiment_case_impressions.recommender
    ]
    experiment_impressions_hyper_parameters = commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
        experiment_case_impressions.hyper_parameter_tuning_parameters
    ]

    assert experiment_impressions_recommender.search_hyper_parameters is not None

    benchmark_reader = commons.get_reader_from_benchmark(
        benchmark_config=experiment_impressions_benchmark.config,
        benchmark=experiment_impressions_benchmark.benchmark,
    )

    dataset = benchmark_reader.dataset

    interactions_data_splits = dataset.get_urm_splits(
        evaluation_strategy=experiment_impressions_hyper_parameters.evaluation_strategy
    )
    impressions_data_splits = dataset.get_uim_splits(
        evaluation_strategy=experiment_impressions_hyper_parameters.evaluation_strategy,
    )

    # Only load features of the train+validation split.
    impressions_feature_frequency_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_impressions_benchmark.benchmark,
            evaluation_strategy=experiment_impressions_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_FREQUENCY,
            impressions_feature_column=commons.ImpressionsFeatureColumnsFrequency.FREQUENCY,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    impressions_feature_position_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_impressions_benchmark.benchmark,
            evaluation_strategy=experiment_impressions_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_POSITION,
            impressions_feature_column=commons.ImpressionsFeatureColumnsPosition.POSITION,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    impressions_feature_timestamp_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_impressions_benchmark.benchmark,
            evaluation_strategy=experiment_impressions_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_TIMESTAMP,
            impressions_feature_column=commons.ImpressionsFeatureColumnsTimestamp.TIMESTAMP,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    if commons.Benchmarks.FINNNoSlates == experiment_impressions_benchmark.benchmark:
        impressions_feature_last_seen_train_validation = dataset.sparse_matrix_impression_feature(
            feature=commons.get_feature_key_by_benchmark(
                benchmark=experiment_impressions_benchmark.benchmark,
                evaluation_strategy=experiment_impressions_hyper_parameters.evaluation_strategy,
                impressions_feature=commons.ImpressionsFeatures.USER_ITEM_LAST_SEEN,
                impressions_feature_column=commons.ImpressionsFeatureColumnsLastSeen.EUCLIDEAN,
                impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
            )
        )

    else:
        impressions_feature_last_seen_train_validation = dataset.sparse_matrix_impression_feature(
            feature=commons.get_feature_key_by_benchmark(
                benchmark=experiment_impressions_benchmark.benchmark,
                evaluation_strategy=experiment_impressions_hyper_parameters.evaluation_strategy,
                impressions_feature=commons.ImpressionsFeatures.USER_ITEM_LAST_SEEN,
                impressions_feature_column=commons.ImpressionsFeatureColumnsLastSeen.TOTAL_DAYS,
                impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
            )
        )

    folder_path_recommender_impressions = FOLDER_HYPER_PARAMETER_TUNING_IMPRESSIONS_HEURISTICS.format(
        benchmark=experiment_impressions_benchmark.benchmark.value,
        evaluation_strategy=experiment_impressions_hyper_parameters.evaluation_strategy.value,
    )
    folder_path_export_confidence_intervals = DIR_CI_RECOMMENDER_HEURISTICS.format(
        benchmark=experiment_impressions_benchmark.benchmark.value,
        evaluation_strategy=experiment_impressions_hyper_parameters.evaluation_strategy.value,
    )

    file_name_postfix = commons.MAPPER_FILE_NAME_POSTFIX[commons.TrainedRecommenderType.TRAIN_VALIDATION]

    urm_train = interactions_data_splits.sp_urm_train_validation
    uim_train = impressions_data_splits.sp_uim_train_validation

    recommender_trained_impressions = commons.load_recommender_trained_impressions(
        recommender_class_impressions=experiment_impressions_recommender.recommender,
        folder_path=folder_path_recommender_impressions,
        file_name_postfix=file_name_postfix,
        urm_train=urm_train.copy(),
        uim_train=uim_train.copy(),
        uim_frequency=impressions_feature_frequency_train_validation.copy(),
        uim_position=impressions_feature_position_train_validation.copy(),
        uim_timestamp=impressions_feature_timestamp_train_validation.copy(),
        uim_last_seen=impressions_feature_last_seen_train_validation.copy(),
        recommender_baseline=None,
    )

    if recommender_trained_impressions is None:
        # We require a recommender that is already optimized.
        logger.warning(
            f"Early-skipping on {_compute_confidence_intervals_recommender_trained_impressions_heuristics.__name__}. "
            f"Could not load trained recommenders for {experiment_impressions_recommender.recommender} with the "
            f"benchmark {experiment_impressions_benchmark.benchmark}."
        )
        return

    import random
    import numpy as np

    random.seed(experiment_impressions_hyper_parameters.reproducibility_seed)
    np.random.seed(experiment_impressions_hyper_parameters.reproducibility_seed)

    evaluators = commons.get_evaluators(
        data_splits=interactions_data_splits,
        experiment_hyper_parameter_tuning_parameters=experiment_impressions_hyper_parameters,
    )

    evaluators.test.compute_recommender_confidence_intervals(
        recommender=recommender_trained_impressions,
        recommender_name=f"{recommender_trained_impressions.RECOMMENDER_NAME}_{file_name_postfix}",
        folder_export_results=folder_path_export_confidence_intervals,
    )


def _compute_confidence_intervals_recommender_trained_impressions(
    experiment_case_impressions: commons.ExperimentCase,
    experiment_case_baseline: commons.ExperimentCase,
    experiment_baseline_similarity: Optional[str],
    folder_hyper_parameter_tuning_impressions: str,
    folder_path_export_confidence_intervals: str,
    try_folded_recommender: bool,
) -> None:
    """
    Runs in a dask worker the hyper-parameter tuning of a re-ranking impression recommender.

    This method should not be called from outside.
    """

    experiment_impressions_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
        experiment_case_impressions.benchmark
    ]
    experiment_impressions_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
        experiment_case_impressions.recommender
    ]
    experiment_impressions_hyper_parameters = commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
        experiment_case_impressions.hyper_parameter_tuning_parameters
    ]

    experiment_baseline_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
        experiment_case_baseline.benchmark
    ]
    experiment_baseline_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
        experiment_case_baseline.recommender
    ]
    experiment_baseline_hyper_parameters = commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
        experiment_case_baseline.hyper_parameter_tuning_parameters
    ]

    assert experiment_impressions_recommender.search_hyper_parameters is not None

    benchmark_reader = commons.get_reader_from_benchmark(
        benchmark_config=experiment_impressions_benchmark.config,
        benchmark=experiment_impressions_benchmark.benchmark,
    )

    dataset = benchmark_reader.dataset

    interactions_data_splits = dataset.get_urm_splits(
        evaluation_strategy=experiment_impressions_hyper_parameters.evaluation_strategy
    )
    impressions_data_splits = dataset.get_uim_splits(
        evaluation_strategy=experiment_impressions_hyper_parameters.evaluation_strategy,
    )

    # Only load features of the train+validation split.
    impressions_feature_frequency_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_impressions_benchmark.benchmark,
            evaluation_strategy=experiment_impressions_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_FREQUENCY,
            impressions_feature_column=commons.ImpressionsFeatureColumnsFrequency.FREQUENCY,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    impressions_feature_position_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_impressions_benchmark.benchmark,
            evaluation_strategy=experiment_impressions_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_POSITION,
            impressions_feature_column=commons.ImpressionsFeatureColumnsPosition.POSITION,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    impressions_feature_timestamp_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_impressions_benchmark.benchmark,
            evaluation_strategy=experiment_impressions_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_TIMESTAMP,
            impressions_feature_column=commons.ImpressionsFeatureColumnsTimestamp.TIMESTAMP,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    if commons.Benchmarks.FINNNoSlates == experiment_impressions_benchmark.benchmark:
        impressions_feature_last_seen_train_validation = dataset.sparse_matrix_impression_feature(
            feature=commons.get_feature_key_by_benchmark(
                benchmark=experiment_impressions_benchmark.benchmark,
                evaluation_strategy=experiment_impressions_hyper_parameters.evaluation_strategy,
                impressions_feature=commons.ImpressionsFeatures.USER_ITEM_LAST_SEEN,
                impressions_feature_column=commons.ImpressionsFeatureColumnsLastSeen.EUCLIDEAN,
                impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
            )
        )

    else:
        impressions_feature_last_seen_train_validation = dataset.sparse_matrix_impression_feature(
            feature=commons.get_feature_key_by_benchmark(
                benchmark=experiment_impressions_benchmark.benchmark,
                evaluation_strategy=experiment_impressions_hyper_parameters.evaluation_strategy,
                impressions_feature=commons.ImpressionsFeatures.USER_ITEM_LAST_SEEN,
                impressions_feature_column=commons.ImpressionsFeatureColumnsLastSeen.TOTAL_DAYS,
                impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
            )
        )

    from impressions_evaluation.experiments.baselines import DIR_TRAINED_MODELS_BASELINES as folder_hyper_parameter_tuning_baselines

    folder_path_recommender_baseline = folder_hyper_parameter_tuning_baselines.format(
        benchmark=experiment_baseline_benchmark.benchmark.value,
        evaluation_strategy=experiment_baseline_hyper_parameters.evaluation_strategy.value,
    )
    folder_path_recommender_folded = folder_hyper_parameter_tuning_baselines.format(
        benchmark=experiment_baseline_benchmark.benchmark.value,
        evaluation_strategy=experiment_baseline_hyper_parameters.evaluation_strategy.value,
    )
    folder_path_recommender_impressions = folder_hyper_parameter_tuning_impressions.format(
        benchmark=experiment_baseline_benchmark.benchmark.value,
        evaluation_strategy=experiment_baseline_hyper_parameters.evaluation_strategy.value,
    )

    file_name_postfix = commons.MAPPER_FILE_NAME_POSTFIX[commons.TrainedRecommenderType.TRAIN_VALIDATION]

    urm_train = interactions_data_splits.sp_urm_train_validation
    uim_train = impressions_data_splits.sp_uim_train_validation

    recommender_trained_baseline = commons.load_recommender_trained_baseline(
        recommender_baseline_class=experiment_baseline_recommender.recommender,
        folder_path=folder_path_recommender_baseline,
        file_name_postfix=file_name_postfix,
        urm_train=urm_train.copy(),
        similarity=experiment_baseline_similarity,
    )

    if recommender_trained_baseline is None:
        return

    recommender_trained_folded: Optional[FoldedMatrixFactorizationRecommender] = None
    if try_folded_recommender:
        recommender_trained_folded = commons.load_recommender_trained_folded(
            recommender_baseline_instance=recommender_trained_baseline,
            folder_path=folder_path_recommender_folded,
            file_name_postfix=file_name_postfix,
            urm_train=urm_train.copy()
        )

    if try_folded_recommender and recommender_trained_folded is None:
        return

    recommender_trained_impressions = commons.load_recommender_trained_impressions(
        recommender_class_impressions=experiment_impressions_recommender.recommender,
        folder_path=folder_path_recommender_impressions,
        file_name_postfix=file_name_postfix,
        urm_train=urm_train.copy(),
        uim_train=uim_train.copy(),
        uim_frequency=impressions_feature_frequency_train_validation.copy(),
        uim_position=impressions_feature_position_train_validation.copy(),
        uim_timestamp=impressions_feature_timestamp_train_validation.copy(),
        uim_last_seen=impressions_feature_last_seen_train_validation.copy(),
        recommender_baseline=(
            recommender_trained_folded
            if try_folded_recommender
            else recommender_trained_baseline
        )
    )

    if recommender_trained_impressions is None:
        # We require a recommender that is already optimized.
        logger.warning(
            f"Early-skipping on {_compute_confidence_intervals_recommender_trained_impressions.__name__}. Could not load "
            f"trained recommenders for {experiment_baseline_recommender.recommender} with the benchmark "
            f"{experiment_baseline_benchmark.benchmark}. Folded Recommender? {try_folded_recommender}"
        )
        return

    import random
    import numpy as np

    random.seed(experiment_impressions_hyper_parameters.reproducibility_seed)
    np.random.seed(experiment_impressions_hyper_parameters.reproducibility_seed)

    evaluators = commons.get_evaluators(
        data_splits=interactions_data_splits,
        experiment_hyper_parameter_tuning_parameters=experiment_impressions_hyper_parameters,
    )

    evaluators.test.compute_recommender_confidence_intervals(
        recommender=recommender_trained_impressions,
        recommender_name=f"{recommender_trained_impressions.RECOMMENDER_NAME}_{file_name_postfix}",
        folder_export_results=folder_path_export_confidence_intervals,
    )


def compute_confidence_intervals(
    dask_interface: DaskInterface,
    experiment_cases_interface_baselines: commons.ExperimentCasesInterface,
    experiment_cases_interface_impressions_heuristics: commons.ExperimentCasesInterface,
    experiment_cases_interface_impressions_re_ranking: commons.ExperimentCasesInterface,
    experiment_cases_interface_impressions_user_profiles: commons.ExperimentCasesInterface,
) -> None:
    """
    Public method that instructs dask to run in dask workers the hyper-parameter tuning of the impressions discounting
    recommenders.

    Processes are always preferred than threads as the hyper-parameter tuning loop is probably not thread-safe.
    """
    # First compute baselines.
    for case_baseline in experiment_cases_interface_baselines.experiment_cases:
        baseline_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
            case_baseline.benchmark
        ]
        baseline_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
            case_baseline.recommender
        ]
        baseline_hyper_parameters = commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
            case_baseline.hyper_parameter_tuning_parameters
        ]

        similarities = commons.get_similarities_by_recommender_class(
            recommender_class=baseline_recommender.recommender,
            knn_similarities=baseline_hyper_parameters.knn_similarity_types,
        )

        for similarity in similarities:
            dask_interface.submit_job(
                job_key=(
                    f"_compute_confidence_intervals_recommender_trained_baseline"
                    f"|{baseline_benchmark.benchmark.value}"
                    f"|{baseline_recommender.recommender.RECOMMENDER_NAME}"
                    f"|{similarity}"
                    f"|{uuid.uuid4()}"
                ),
                job_priority=(
                    baseline_benchmark.priority
                    * baseline_recommender.priority
                ),
                job_info={
                    "recommender": baseline_recommender.recommender.RECOMMENDER_NAME,
                    "benchmark": baseline_benchmark.benchmark.value,
                    "similarity": similarity,
                },
                method=_compute_confidence_intervals_recommender_trained_baseline,
                method_kwargs={
                    "experiment_case_baseline": case_baseline,
                    "experiment_baseline_similarity": similarity,
                }
            )

    # Then folded baselines
    for case_baseline in experiment_cases_interface_baselines.experiment_cases:
        baseline_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
            case_baseline.benchmark
        ]
        baseline_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
            case_baseline.recommender
        ]
        baseline_hyper_parameters = commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
            case_baseline.hyper_parameter_tuning_parameters
        ]

        similarities = commons.get_similarities_by_recommender_class(
            recommender_class=baseline_recommender.recommender,
            knn_similarities=baseline_hyper_parameters.knn_similarity_types,
        )

        for similarity in similarities:
            dask_interface.submit_job(
                job_key=(
                    f"_compute_confidence_intervals_recommender_trained_folded"
                    f"|{baseline_benchmark.benchmark.value}"
                    f"|{baseline_recommender.recommender.RECOMMENDER_NAME}"
                    f"|{similarity}"
                    f"|{uuid.uuid4()}"
                ),
                job_priority=(
                    baseline_benchmark.priority
                    * baseline_recommender.priority
                ),
                job_info={
                    "recommender": baseline_recommender.recommender.RECOMMENDER_NAME,
                    "benchmark": baseline_benchmark.benchmark.value,
                    "similarity": similarity,
                },
                method=_compute_confidence_intervals_recommender_trained_folded,
                method_kwargs={
                    "experiment_case_baseline": case_baseline,
                    "experiment_baseline_similarity": similarity,
                }
            )

    # Then impressions heuristics
    for case_impressions_heuristics in experiment_cases_interface_impressions_heuristics.experiment_cases:
        impressions_heuristics_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
            case_impressions_heuristics.benchmark
        ]
        impressions_heuristics_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
            case_impressions_heuristics.recommender
        ]

        dask_interface.submit_job(
            job_key=(
                f"_compute_confidence_intervals_recommender_trained_folded"
                f"|{impressions_heuristics_benchmark.benchmark.value}"
                f"|{impressions_heuristics_recommender.recommender.RECOMMENDER_NAME}"
                f"|{uuid.uuid4()}"
            ),
            job_priority=(
                impressions_heuristics_benchmark.priority
                * impressions_heuristics_recommender.priority
            ),
            job_info={
                "recommender": impressions_heuristics_recommender.recommender.RECOMMENDER_NAME,
                "benchmark": impressions_heuristics_benchmark.benchmark.value,
            },
            method=_compute_confidence_intervals_recommender_trained_impressions_heuristics,
            method_kwargs={
                "experiment_case_impressions": case_impressions_heuristics,
            }
        )

    # Then impressions re-ranking
    for case_impressions_re_ranking in experiment_cases_interface_impressions_re_ranking.experiment_cases:
        for case_baseline in experiment_cases_interface_baselines.experiment_cases:
            experiment_can_be_tested = (
                case_impressions_re_ranking.benchmark == case_baseline.benchmark
                and case_baseline.hyper_parameter_tuning_parameters == case_baseline.hyper_parameter_tuning_parameters
            )

            if not experiment_can_be_tested:
                continue

            re_ranking_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
                case_impressions_re_ranking.benchmark
            ]
            re_ranking_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
                case_impressions_re_ranking.recommender
            ]

            baseline_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
                case_baseline.recommender
            ]
            baseline_hyper_parameters = commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
                case_baseline.hyper_parameter_tuning_parameters
            ]

            similarities = commons.get_similarities_by_recommender_class(
                recommender_class=baseline_recommender.recommender,
                knn_similarities=baseline_hyper_parameters.knn_similarity_types,
            )

            for similarity in similarities:
                for try_folded_recommender in [True, False]:
                    dask_interface.submit_job(
                        job_key=(
                            f"_compute_impressions_heuristics_hyper_parameter_tuning"
                            f"|{case_impressions_re_ranking.benchmark.value}"
                            f"|{re_ranking_recommender.recommender.RECOMMENDER_NAME}"
                            f"|{baseline_recommender.recommender.RECOMMENDER_NAME}"
                            f"|{similarity}"
                            f"|{try_folded_recommender}"
                            f"|{uuid.uuid4()}"
                        ),
                        job_priority=(
                            re_ranking_benchmark.priority
                            * re_ranking_recommender.priority
                            * baseline_recommender.priority
                        ),
                        job_info={
                            "recommender": re_ranking_recommender.recommender.RECOMMENDER_NAME,
                            "baseline": baseline_recommender.recommender.RECOMMENDER_NAME,
                            "benchmark": re_ranking_benchmark.benchmark.value,
                            "similarity": similarity,
                            "try_folded_recommender": try_folded_recommender,
                        },
                        method=_compute_confidence_intervals_recommender_trained_impressions,
                        method_kwargs={
                            "experiment_case_impressions": case_impressions_re_ranking,
                            "experiment_case_baseline": case_baseline,
                            "experiment_baseline_similarity": similarity,
                            "folder_hyper_parameter_tuning_impressions": FOLDER_HYPER_PARAMETER_TUNING_IMPRESSIONS_RE_RANKING,
                            "folder_path_export_confidence_intervals": DIR_CI_RECOMMENDER_RE_RANKING,
                            "try_folded_recommender": try_folded_recommender,
                        }
                    )

    # Then impressions user-profiles
    for case_impressions_user_profiles in experiment_cases_interface_impressions_user_profiles.experiment_cases:
        for case_baseline in experiment_cases_interface_baselines.experiment_cases:
            experiment_can_be_tested = (
                case_impressions_user_profiles.benchmark == case_baseline.benchmark
                and case_baseline.hyper_parameter_tuning_parameters == case_baseline.hyper_parameter_tuning_parameters
            )

            if not experiment_can_be_tested:
                continue

            user_profiles_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
                case_impressions_user_profiles.benchmark
            ]
            user_profiles_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
                case_impressions_user_profiles.recommender
            ]

            baseline_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
                case_baseline.recommender
            ]
            baseline_hyper_parameters = commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
                case_baseline.hyper_parameter_tuning_parameters
            ]

            similarities = commons.get_similarities_by_recommender_class(
                recommender_class=baseline_recommender.recommender,
                knn_similarities=baseline_hyper_parameters.knn_similarity_types,
            )

            for similarity in similarities:
                for try_folded_recommender in [True, False]:
                    dask_interface.submit_job(
                        job_key=(
                            f"_compute_impressions_heuristics_hyper_parameter_tuning"
                            f"|{case_impressions_user_profiles.benchmark.value}"
                            f"|{user_profiles_recommender.recommender.RECOMMENDER_NAME}"
                            f"|{baseline_recommender.recommender.RECOMMENDER_NAME}"
                            f"|{similarity}"
                            f"|{try_folded_recommender}"
                            f"|{uuid.uuid4()}"
                        ),
                        job_priority=(
                            user_profiles_benchmark.priority
                            * user_profiles_recommender.priority
                            * baseline_recommender.priority
                        ),
                        job_info={
                            "recommender": user_profiles_recommender.recommender.RECOMMENDER_NAME,
                            "baseline": baseline_recommender.recommender.RECOMMENDER_NAME,
                            "benchmark": user_profiles_benchmark.benchmark.value,
                            "similarity": similarity,
                            "try_folded_recommender": try_folded_recommender,
                        },
                        method=_compute_confidence_intervals_recommender_trained_impressions,
                        method_kwargs={
                            "experiment_case_impressions": case_impressions_user_profiles,
                            "experiment_case_baseline": case_baseline,
                            "experiment_baseline_similarity": similarity,
                            "folder_hyper_parameter_tuning_impressions": FOLDER_HYPER_PARAMETER_TUNING_IMPRESSIONS_USER_PROFILES,
                            "folder_path_export_confidence_intervals": DIR_CI_RECOMMENDER_USER_PROFILES,
                            "try_folded_recommender": try_folded_recommender,
                        }
                    )
