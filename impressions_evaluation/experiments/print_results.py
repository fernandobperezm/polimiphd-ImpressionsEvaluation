import itertools
import logging
import os
from typing import Type, Optional, cast, Union, Sequence, Literal

import Recommenders.Recommender_import_list as recommenders
import numpy as np
import pandas as pd
import scipy.sparse as sp
from Recommenders.BaseMatrixFactorizationRecommender import (
    BaseMatrixFactorizationRecommender,
)
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.BaseSimilarityMatrixRecommender import (
    BaseSimilarityMatrixRecommender,
    BaseUserSimilarityMatrixRecommender,
    BaseItemSimilarityMatrixRecommender,
)
from recsys_framework_extensions.data.io import DataIO
from recsys_framework_extensions.data.mixins import InteractionsDataSplits
from recsys_framework_extensions.plotting import (
    generate_accuracy_and_beyond_metrics_pandas,
    DataFrameResults,
)

import impressions_evaluation.experiments.baselines as baselines
import impressions_evaluation.experiments.commons as commons
import impressions_evaluation.experiments.impression_aware.heuristics as heuristics
import impressions_evaluation.experiments.impression_aware.re_ranking as re_ranking
import impressions_evaluation.experiments.impression_aware.user_profiles as user_profiles
from impressions_evaluation.experiments.impression_aware import (
    DIR_TRAINED_MODELS_IMPRESSION_AWARE,
)
from impressions_evaluation.impression_recommenders.re_ranking.cycling import (
    CyclingRecommender,
)
from impressions_evaluation.impression_recommenders.re_ranking.impressions_discounting import (
    ImpressionsDiscountingRecommender,
)
from impressions_evaluation.impression_recommenders.user_profile.folding import (
    FoldedMatrixFactorizationRecommender,
)
from impressions_evaluation.impression_recommenders.user_profile.weighted import (
    BaseWeightedUserProfileRecommender,
    ItemWeightedUserProfileRecommender,
    UserWeightedUserProfileRecommender,
)

logger = logging.getLogger(__name__)


####################################################################################################
####################################################################################################
#                                REPRODUCIBILITY VARIABLES                            #
####################################################################################################
####################################################################################################
DIR_RESULTS_TO_EXPORT = os.path.join(
    commons.DIR_RESULTS_EXPORT,
    "{script_name}",
    "model_evaluation",
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
#                       Utility to mock the loading an already-tuned recommender                   #
####################################################################################################
####################################################################################################
def mock_trained_recommender(
    experiment_recommender: commons.ExperimentRecommender,
    experiment_benchmark: commons.ExperimentBenchmark,
    experiment_hyper_parameter_tuning_parameters: commons.HyperParameterTuningParameters,
    experiment_model_dir: str,
    data_splits: InteractionsDataSplits,
    similarity: Optional[str],
    model_type: baselines.TrainedRecommenderType,
    try_folded_recommender: bool,
) -> Optional[BaseRecommender]:
    """Loads to memory an mock of an already-trained recommender.

    This function loads the requested recommender (`experiment_recommender`) on disk. It can load a folded-in
    or the original version of the recommender.

    """
    if baselines.TrainedRecommenderType.TRAIN == model_type:
        urm_train = data_splits.sp_urm_train
        file_name_postfix = "best_model"
    elif baselines.TrainedRecommenderType.TRAIN_VALIDATION == model_type:
        urm_train = data_splits.sp_urm_train_validation
        file_name_postfix = "best_model_last"
    else:
        raise ValueError(
            f"{mock_trained_recommender.__name__} failed because it received an invalid instance of the "
            f"enum {baselines.TrainedRecommenderType} (received value {model_type}). Valid values are "
            f"{list(baselines.TrainedRecommenderType)}"
        )

    recommender_name = f"{experiment_recommender.recommender.RECOMMENDER_NAME}"
    if experiment_recommender.recommender in [
        recommenders.ItemKNNCFRecommender,
        recommenders.UserKNNCFRecommender,
    ]:
        assert similarity is not None
        assert (
            similarity
            in experiment_hyper_parameter_tuning_parameters.knn_similarity_types
        )

        recommender_name = (
            f"{experiment_recommender.recommender.RECOMMENDER_NAME}_{similarity}"
        )

    folder_path = experiment_model_dir.format(
        benchmark=experiment_benchmark.benchmark.value,
        evaluation_strategy=experiment_hyper_parameter_tuning_parameters.evaluation_strategy.value,
    )

    recommender_metadata_exists = DataIO.s_file_exists(
        folder_path=folder_path,
        file_name=f"{recommender_name}_metadata",
    )

    recommender_data_exists = DataIO.s_file_exists(
        folder_path=folder_path,
        file_name=f"{recommender_name}_{file_name_postfix}",
    )

    if not recommender_metadata_exists and not recommender_data_exists:
        return None

    trained_recommender_instance = experiment_recommender.recommender(
        URM_train=urm_train.copy(),
    )
    trained_recommender_instance.RECOMMENDER_NAME = recommender_name

    can_recommender_be_folded = isinstance(
        trained_recommender_instance,
        BaseMatrixFactorizationRecommender,
    )

    if try_folded_recommender:
        if can_recommender_be_folded:
            trained_recommender_instance = cast(
                BaseMatrixFactorizationRecommender,
                trained_recommender_instance,
            )

            setattr(
                trained_recommender_instance,
                FoldedMatrixFactorizationRecommender.ATTR_NAME_ITEM_FACTORS,
                np.array([], np.float32),
            )

            trained_folded_recommender_instance = FoldedMatrixFactorizationRecommender(
                urm_train=urm_train.copy(),
                trained_recommender=trained_recommender_instance,
            )

            return trained_folded_recommender_instance
        else:
            return None
    else:
        return trained_recommender_instance


####################################################################################################
####################################################################################################
#             Results exporting          #
####################################################################################################
####################################################################################################
def _print_baselines_metrics(
    *,
    recommenders_baselines: list[commons.RecommenderBaseline],
    experiment_benchmark: commons.ExperimentBenchmark,
    experiment_hyper_parameters: commons.HyperParameterTuningParameters,
    interaction_data_splits: InteractionsDataSplits,
    num_test_users: int,
    accuracy_metrics_list: list[str],
    beyond_accuracy_metrics_list: list[str],
    all_metrics_list: list[str],
    cutoffs_list: list[int],
    knn_similarity_list: list[commons.T_SIMILARITY_TYPE],
    export_experiments_folder_path: str,
) -> DataFrameResults:
    experiments_folder_path = baselines.DIR_TRAINED_MODELS_BASELINES.format(
        benchmark=experiment_benchmark.benchmark.value,
        evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
    )

    baseline_experiment_recommenders = [
        commons.MAPPER_AVAILABLE_RECOMMENDERS[rec] for rec in recommenders_baselines
    ]

    base_algorithm_list = []
    for baseline_experiment_recommender in baseline_experiment_recommenders:
        base_algorithm_list.append(baseline_experiment_recommender.recommender)

        similarities: list[commons.T_SIMILARITY_TYPE] = [None]  # type: ignore
        if baseline_experiment_recommender.recommender in [
            recommenders.ItemKNNCFRecommender,
            recommenders.UserKNNCFRecommender,
        ]:
            similarities = knn_similarity_list

        # TODO: REMOVE FOLDED RECOMMENDERS
        # for similarity in similarities:
        #     # This inner loop is to load Folded Recommenders. If we cannot mock a Folded recommender,
        #     # then this method returns None.
        #     loaded_recommender = mock_trained_recommender(
        #         experiment_recommender=baseline_experiment_recommender,
        #         experiment_benchmark=experiment_benchmark,
        #         experiment_hyper_parameter_tuning_parameters=experiment_hyper_parameters,
        #         experiment_model_dir=experiments_folder_path,
        #         data_splits=interaction_data_splits,
        #         similarity=similarity,
        #         model_type=baselines.TrainedRecommenderType.TRAIN_VALIDATION,
        #         try_folded_recommender=True,
        #     )
        #
        #     if loaded_recommender is None:
        #         logger.warning(
        #             f"The recommender {baseline_experiment_recommender.recommender} for the dataset {experiment_benchmark} "
        #             f"returned empty. Skipping."
        #         )
        #         continue
        #
        #     base_algorithm_list.append(loaded_recommender)

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


def _print_impressions_heuristics_metrics(
    *,
    recommenders_impressions_heuristics: list[commons.RecommenderImpressions],
    experiment_hyper_parameters: commons.HyperParameterTuningParameters,
    experiment_benchmark: commons.ExperimentBenchmark,
    num_test_users: int,
    accuracy_metrics_list: list[str],
    beyond_accuracy_metrics_list: list[str],
    all_metrics_list: list[str],
    cutoffs_list: list[int],
    export_experiments_folder_path: str,
) -> DataFrameResults:
    experiments_folder_path = DIR_TRAINED_MODELS_IMPRESSION_AWARE.format(
        benchmark=experiment_benchmark.benchmark.value,
        evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
    )

    impressions_heuristics_recommenders = [
        commons.MAPPER_AVAILABLE_RECOMMENDERS[rec].recommender
        for rec in recommenders_impressions_heuristics
    ]

    return generate_accuracy_and_beyond_metrics_pandas(
        experiments_folder_path=experiments_folder_path,
        export_experiments_folder_path=export_experiments_folder_path,
        num_test_users=num_test_users,
        base_algorithm_list=impressions_heuristics_recommenders,
        knn_similarity_list=[],
        other_algorithm_list=None,
        accuracy_metrics_list=accuracy_metrics_list,
        beyond_accuracy_metrics_list=beyond_accuracy_metrics_list,
        all_metrics_list=all_metrics_list,
        cutoffs_list=cutoffs_list,
        icm_names=None,
    )


def _print_impressions_re_ranking_metrics(
    *,
    recommenders_baselines: list[commons.RecommenderBaseline],
    recommenders_impressions_re_ranking: list[commons.RecommenderImpressions],
    experiment_benchmark: commons.ExperimentBenchmark,
    experiment_hyper_parameters: commons.HyperParameterTuningParameters,
    interaction_data_splits: InteractionsDataSplits,
    num_test_users: int,
    accuracy_metrics_list: list[str],
    beyond_accuracy_metrics_list: list[str],
    all_metrics_list: list[str],
    cutoffs_list: list[int],
    knn_similarity_list: list[commons.T_SIMILARITY_TYPE],
    export_experiments_folder_path: str,
) -> DataFrameResults:
    baseline_experiments_folder_path = baselines.DIR_TRAINED_MODELS_BASELINES.format(
        benchmark=experiment_benchmark.benchmark.value,
        evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
    )

    re_ranking_experiments_folder_path = DIR_TRAINED_MODELS_IMPRESSION_AWARE.format(
        benchmark=experiment_benchmark.benchmark.value,
        evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
    )

    baseline_experiment_recommenders = [
        commons.MAPPER_AVAILABLE_RECOMMENDERS[rec] for rec in recommenders_baselines
    ]

    re_ranking_experiment_recommenders = [
        commons.MAPPER_AVAILABLE_RECOMMENDERS[rec]
        for rec in recommenders_impressions_re_ranking
    ]

    base_algorithm_list = []
    for re_ranking_experiment_recommender in re_ranking_experiment_recommenders:
        for baseline_experiment_recommender in baseline_experiment_recommenders:
            similarities: list[commons.T_SIMILARITY_TYPE] = [None]  # type: ignore
            if baseline_experiment_recommender.recommender in [
                recommenders.ItemKNNCFRecommender,
                recommenders.UserKNNCFRecommender,
            ]:
                similarities = knn_similarity_list

            for similarity in similarities:
                # TODO: REMOVE FOLDED RECOMMENDERS, IT WAS [True, False]
                for try_folded_recommender in [False]:
                    loaded_baseline_recommender = mock_trained_recommender(
                        experiment_recommender=baseline_experiment_recommender,
                        experiment_benchmark=experiment_benchmark,
                        experiment_hyper_parameter_tuning_parameters=experiment_hyper_parameters,
                        experiment_model_dir=baseline_experiments_folder_path,
                        data_splits=interaction_data_splits,
                        similarity=similarity,
                        model_type=baselines.TrainedRecommenderType.TRAIN_VALIDATION,
                        try_folded_recommender=try_folded_recommender,
                    )

                    if loaded_baseline_recommender is None:
                        continue

                    re_ranking_class = cast(
                        Type[
                            Union[CyclingRecommender, ImpressionsDiscountingRecommender]
                        ],
                        re_ranking_experiment_recommender.recommender,
                    )

                    re_ranking_recommender = re_ranking_class(
                        urm_train=interaction_data_splits.sp_urm_train_validation,
                        uim_position=sp.csr_matrix([[]]),
                        uim_frequency=sp.csr_matrix([[]]),
                        uim_last_seen=sp.csr_matrix([[]]),
                        trained_recommender=loaded_baseline_recommender,
                    )

                    re_ranking_recommender.RECOMMENDER_NAME = (
                        f"{re_ranking_class.RECOMMENDER_NAME}"
                        f"_{loaded_baseline_recommender.RECOMMENDER_NAME}"
                    )

                    base_algorithm_list.append(re_ranking_recommender)

    return generate_accuracy_and_beyond_metrics_pandas(
        experiments_folder_path=re_ranking_experiments_folder_path,
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


# TODO: REMOVE?
def _print_ablation_impressions_re_ranking_metrics(
    ablation_re_ranking_experiment_cases_interface: commons.ExperimentCasesInterface,
    baseline_experiment_cases_interface: commons.ExperimentCasesInterface,
    baseline_experiment_benchmark: commons.ExperimentBenchmark,
    baseline_experiment_hyper_parameters: commons.HyperParameterTuningParameters,
    interaction_data_splits: InteractionsDataSplits,
    num_test_users: int,
    accuracy_metrics_list: list[str],
    beyond_accuracy_metrics_list: list[str],
    all_metrics_list: list[str],
    cutoffs_list: list[int],
    knn_similarity_list: list[commons.T_SIMILARITY_TYPE],
    export_experiments_folder_path: str,
) -> DataFrameResults:
    baseline_experiments_folder_path = baselines.DIR_TRAINED_MODELS_BASELINES.format(
        benchmark=baseline_experiment_benchmark.benchmark.value,
        evaluation_strategy=baseline_experiment_hyper_parameters.evaluation_strategy.value,
    )

    re_ranking_experiments_folder_path = re_ranking.DIR_TRAINED_MODELS_RE_RANKING.format(
        benchmark=baseline_experiment_benchmark.benchmark.value,
        evaluation_strategy=baseline_experiment_hyper_parameters.evaluation_strategy.value,
    )

    baseline_experiment_recommenders = [
        commons.MAPPER_AVAILABLE_RECOMMENDERS[rec]
        for rec in baseline_experiment_cases_interface.to_use_recommenders
    ]

    re_ranking_experiment_recommenders = [
        commons.MAPPER_AVAILABLE_RECOMMENDERS[rec]
        for rec in ablation_re_ranking_experiment_cases_interface.to_use_recommenders
    ]

    base_algorithm_list = []
    for re_ranking_experiment_recommender in re_ranking_experiment_recommenders:
        for baseline_experiment_recommender in baseline_experiment_recommenders:
            similarities: list[commons.T_SIMILARITY_TYPE] = [None]  # type: ignore
            if baseline_experiment_recommender.recommender in [
                recommenders.ItemKNNCFRecommender,
                recommenders.UserKNNCFRecommender,
            ]:
                similarities = knn_similarity_list

            for similarity in similarities:
                # TODO: REMOVE FOLDED RECOMMENDERS, IT WAS [True, False]
                for try_folded_recommender in [False]:
                    loaded_baseline_recommender = mock_trained_recommender(
                        experiment_recommender=baseline_experiment_recommender,
                        experiment_benchmark=baseline_experiment_benchmark,
                        experiment_hyper_parameter_tuning_parameters=baseline_experiment_hyper_parameters,
                        experiment_model_dir=baseline_experiments_folder_path,
                        data_splits=interaction_data_splits,
                        similarity=similarity,
                        model_type=baselines.TrainedRecommenderType.TRAIN_VALIDATION,
                        try_folded_recommender=try_folded_recommender,
                    )

                    if loaded_baseline_recommender is None:
                        continue

                    re_ranking_class = cast(
                        Type[
                            Union[CyclingRecommender, ImpressionsDiscountingRecommender]
                        ],
                        re_ranking_experiment_recommender.recommender,
                    )

                    re_ranking_recommender = re_ranking_class(
                        urm_train=interaction_data_splits.sp_urm_train_validation,
                        uim_position=sp.csr_matrix([[]]),
                        uim_frequency=sp.csr_matrix([[]]),
                        uim_last_seen=sp.csr_matrix([[]]),
                        trained_recommender=loaded_baseline_recommender,
                    )

                    re_ranking_recommender.RECOMMENDER_NAME = (
                        f"ABLATION_UIM_FREQUENCY"
                        f"_{re_ranking_class.RECOMMENDER_NAME}"
                        f"_{loaded_baseline_recommender.RECOMMENDER_NAME}"
                    )

                    base_algorithm_list.append(re_ranking_recommender)

    return generate_accuracy_and_beyond_metrics_pandas(
        experiments_folder_path=re_ranking_experiments_folder_path,
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


def _print_impressions_user_profiles_metrics(
    *,
    recommenders_baselines: list[commons.RecommenderBaseline],
    recommenders_impressions_user_profiles: list[commons.RecommenderImpressions],
    experiment_benchmark: commons.ExperimentBenchmark,
    experiment_hyper_parameters: commons.HyperParameterTuningParameters,
    interaction_data_splits: InteractionsDataSplits,
    num_test_users: int,
    accuracy_metrics_list: list[str],
    beyond_accuracy_metrics_list: list[str],
    all_metrics_list: list[str],
    cutoffs_list: list[int],
    knn_similarity_list: list[commons.T_SIMILARITY_TYPE],
    export_experiments_folder_path: str,
) -> DataFrameResults:
    baseline_experiments_folder_path = baselines.DIR_TRAINED_MODELS_BASELINES.format(
        benchmark=experiment_benchmark.benchmark.value,
        evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
    )

    user_profiles_experiments_folder_path = DIR_TRAINED_MODELS_IMPRESSION_AWARE.format(
        benchmark=experiment_benchmark.benchmark.value,
        evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
    )

    baseline_experiment_recommenders = [
        commons.MAPPER_AVAILABLE_RECOMMENDERS[rec] for rec in recommenders_baselines
    ]

    user_profiles_experiment_recommenders = [
        commons.MAPPER_AVAILABLE_RECOMMENDERS[rec]
        for rec in recommenders_impressions_user_profiles
    ]

    base_algorithm_list = []
    for user_profiles_experiment_recommender in user_profiles_experiment_recommenders:
        requires_similarity = issubclass(
            user_profiles_experiment_recommender.recommender,
            BaseWeightedUserProfileRecommender,
        )
        if not requires_similarity:
            logger.warning(
                f"The recommender {user_profiles_experiment_recommender.recommender} does not require a "
                f"similarity matrix (should never happen). Skipping."
            )
            continue

        for baseline_experiment_recommender in baseline_experiment_recommenders:
            similarities: list[commons.T_SIMILARITY_TYPE] = [None]  # type: ignore
            if baseline_experiment_recommender.recommender in [
                recommenders.ItemKNNCFRecommender,
                recommenders.UserKNNCFRecommender,
            ]:
                similarities = knn_similarity_list

            for similarity in similarities:
                # TODO: REMOVE FOLDED RECOMMENDERS, IT WAS [True, False]
                for try_folded_recommender in [False]:
                    loaded_baseline_recommender = mock_trained_recommender(
                        experiment_recommender=baseline_experiment_recommender,
                        experiment_benchmark=experiment_benchmark,
                        experiment_hyper_parameter_tuning_parameters=experiment_hyper_parameters,
                        experiment_model_dir=baseline_experiments_folder_path,
                        data_splits=interaction_data_splits,
                        similarity=similarity,
                        model_type=baselines.TrainedRecommenderType.TRAIN_VALIDATION,
                        try_folded_recommender=try_folded_recommender,
                    )

                    if loaded_baseline_recommender is None:
                        logger.warning(
                            f"The recommender {baseline_experiment_recommender.recommender} for the dataset "
                            f"{experiment_benchmark} returned empty. Skipping."
                        )
                        continue

                    requires_user_similarity = issubclass(
                        user_profiles_experiment_recommender.recommender,
                        UserWeightedUserProfileRecommender,
                    )

                    requires_item_similarity = issubclass(
                        user_profiles_experiment_recommender.recommender,
                        ItemWeightedUserProfileRecommender,
                    )

                    recommender_has_user_similarity = isinstance(
                        loaded_baseline_recommender,
                        BaseUserSimilarityMatrixRecommender,
                    )
                    recommender_has_item_similarity = isinstance(
                        loaded_baseline_recommender,
                        BaseItemSimilarityMatrixRecommender,
                    )

                    if requires_user_similarity and not recommender_has_user_similarity:
                        logger.warning(
                            f"Recommender {user_profiles_experiment_recommender.recommender} requires a user-user similarity "
                            f"but instance of {loaded_baseline_recommender.__class__} does not inherit from "
                            f"{BaseUserSimilarityMatrixRecommender}. Skip"
                        )
                        continue

                    if requires_item_similarity and not recommender_has_item_similarity:
                        logger.warning(
                            f"Recommender {user_profiles_experiment_recommender.recommender} requires an item-item similarity "
                            f"but instance of {loaded_baseline_recommender.__class__} does not inherit from "
                            f"{BaseItemSimilarityMatrixRecommender}. Skip"
                        )
                        continue

                    loaded_baseline_recommender = cast(
                        BaseSimilarityMatrixRecommender,
                        loaded_baseline_recommender,
                    )

                    user_profiles_class = cast(
                        Type[BaseWeightedUserProfileRecommender],
                        user_profiles_experiment_recommender.recommender,
                    )

                    setattr(
                        loaded_baseline_recommender,
                        BaseWeightedUserProfileRecommender.ATTR_NAME_W_SPARSE,
                        sp.csr_matrix([], dtype=np.float32),
                    )

                    user_profiles_recommender = user_profiles_class(
                        urm_train=interaction_data_splits.sp_urm_train_validation,
                        uim_train=sp.csr_matrix([[]]),
                        trained_recommender=loaded_baseline_recommender,
                    )

                    base_algorithm_list.append(user_profiles_recommender)

    return generate_accuracy_and_beyond_metrics_pandas(
        experiments_folder_path=user_profiles_experiments_folder_path,
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


def _print_impressions_signal_analysis_re_ranking_metrics(
    *,
    recommenders_baselines: list[commons.RecommenderBaseline],
    recommenders_impressions_signal_analysis_re_ranking: list[
        tuple[commons.RecommenderImpressions, str]
    ],
    experiment_benchmark: commons.ExperimentBenchmark,
    experiment_hyper_parameters: commons.HyperParameterTuningParameters,
    interaction_data_splits: InteractionsDataSplits,
    num_test_users: int,
    accuracy_metrics_list: list[str],
    beyond_accuracy_metrics_list: list[str],
    all_metrics_list: list[str],
    cutoffs_list: list[int],
    knn_similarity_list: list[commons.T_SIMILARITY_TYPE],
    export_experiments_folder_path: str,
) -> DataFrameResults:
    baseline_experiments_folder_path = baselines.DIR_TRAINED_MODELS_BASELINES.format(
        benchmark=experiment_benchmark.benchmark.value,
        evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
    )

    re_ranking_experiments_folder_path = DIR_TRAINED_MODELS_IMPRESSION_AWARE.format(
        benchmark=experiment_benchmark.benchmark.value,
        evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
    )

    baseline_experiment_recommenders = [
        commons.MAPPER_AVAILABLE_RECOMMENDERS[rec] for rec in recommenders_baselines
    ]

    signal_analysis_re_ranking_experiment_recommenders = [
        (commons.MAPPER_AVAILABLE_RECOMMENDERS[rec], case)
        for rec, case in recommenders_impressions_signal_analysis_re_ranking
    ]

    base_algorithm_list = []
    for (
        signal_analysis_re_ranking_experiment_recommender,
        signal_analysis_case,
    ) in signal_analysis_re_ranking_experiment_recommenders:
        for baseline_experiment_recommender in baseline_experiment_recommenders:
            similarities: list[commons.T_SIMILARITY_TYPE] = [None]  # type: ignore
            if baseline_experiment_recommender.recommender in [
                recommenders.ItemKNNCFRecommender,
                recommenders.UserKNNCFRecommender,
            ]:
                similarities = knn_similarity_list

            for similarity in similarities:
                # TODO: REMOVE FOLDED RECOMMENDERS, IT WAS [True, False]
                for try_folded_recommender in [False]:
                    loaded_baseline_recommender = mock_trained_recommender(
                        experiment_recommender=baseline_experiment_recommender,
                        experiment_benchmark=experiment_benchmark,
                        experiment_hyper_parameter_tuning_parameters=experiment_hyper_parameters,
                        experiment_model_dir=baseline_experiments_folder_path,
                        data_splits=interaction_data_splits,
                        similarity=similarity,
                        model_type=baselines.TrainedRecommenderType.TRAIN_VALIDATION,
                        try_folded_recommender=try_folded_recommender,
                    )

                    if loaded_baseline_recommender is None:
                        continue

                    signal_analysis_re_ranking_class = cast(
                        Type[
                            Union[CyclingRecommender, ImpressionsDiscountingRecommender]
                        ],
                        signal_analysis_re_ranking_experiment_recommender.recommender,
                    )

                    signal_analysis_re_ranking_recommender = (
                        signal_analysis_re_ranking_class(
                            urm_train=interaction_data_splits.sp_urm_train_validation,
                            uim_position=sp.csr_matrix([[]]),
                            uim_frequency=sp.csr_matrix([[]]),
                            uim_last_seen=sp.csr_matrix([[]]),
                            trained_recommender=loaded_baseline_recommender,
                        )
                    )

                    signal_analysis_re_ranking_recommender.RECOMMENDER_NAME = (
                        f"{signal_analysis_re_ranking_class.RECOMMENDER_NAME}"
                        f"_{loaded_baseline_recommender.RECOMMENDER_NAME}"
                        f"_{signal_analysis_case}"
                    )

                    base_algorithm_list.append(signal_analysis_re_ranking_recommender)

    return generate_accuracy_and_beyond_metrics_pandas(
        experiments_folder_path=re_ranking_experiments_folder_path,
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
        # We cannot use the recommender name, eg CyclingRecommender.RECOMMENDER_NAME because this dataframe has the recommender names already pre-processed for printing.
        if "LastImpressions" in recommender_name:
            return "Last impression"

        # First FrequencyRecency and then Recency, if not, the latter catches the conditional.
        if "FrequencyRecency" in recommender_name:
            return "Frequency & recency"
        if "Recency" in recommender_name:
            return "Recency"

        # Folded must be last because we want to ensure that we went through all impression-aware recommenders first.
        if "FoldedMF" in recommender_name:
            recommender_name = recommender_name.replace("FoldedMF", "") + " Folded"

        # Covers framework recommenders and removes extra stuff.
        return (
            recommender_name
            # Impression-aware plug-in recommenders -- they are not base recommenders so we set them as empty.
            .replace("HardFrequencyCapping", "")
            .replace("Cycling", "")
            .replace("ImpressionsDiscounting", "")
            .replace("ItemWeightedUserProfile", "")
            .replace("UserWeightedUserProfile", "")
            # Types of experiments
            # TODO: Think in a better way to do types of experiments.
            .replace("ABLATION ONLY IMPRESSIONS FEATURES", "")
            .replace("ABLATION ONLY UIM FREQUENCY", "")
            .replace("SIGNAL ANALYSIS NEGATIVE ABLATION ONLY UIM FREQUENCY", "")
            .replace("SIGNAL ANALYSIS POSITIVE ABLATION ONLY UIM FREQUENCY", "")
            .replace("SIGNAL ANALYSIS SIGN ALL POSITIVE", "")
            .replace("SIGNAL ANALYSIS SIGN ALL NEGATIVE", "")
            .replace("SIGNAL ANALYSIS SIGN POSITIVE", "")
            .replace("SIGNAL ANALYSIS SIGN NEGATIVE", "")
            .replace("SIGNAL ANALYSIS LESS OR EQUAL THRESHOLD", "")
            .replace("SIGNAL ANALYSIS GREAT OR EQUAL THRESHOLD", "")
            # Framework recommenders.
            .replace("Recommender", "")
            .replace("KNNCF", "KNN CF")
            .replace("CF", " ")
            .replace("ElasticNet", "EN")
            # Remaining characters
            .replace("  ", "")
            .replace("_", "")
            .strip()
        )

    def convert_recommender_name_to_model_base_type(recommender_name: str) -> str:
        # We cannot use the recommender name, eg CyclingRecommender.RECOMMENDER_NAME because this dataframe has the recommender names already pre-processed for printing.
        if (
            "LastImpressions" in recommender_name
            or "Recency" in recommender_name
            or "FrequencyRecency" in recommender_name
        ):
            return "Baseline-IARS"

        if "HardFrequencyCapping" in recommender_name:
            return "HFC"
        if "Cycling" in recommender_name:
            return "Cycling"
        if "ImpressionsDiscounting" in recommender_name:
            return "IDF"
        if "ItemWeightedUserProfile" in recommender_name:
            return "IUP"
        if "UserWeightedUserProfile" in recommender_name:
            return "IUP"

        # Folded must be last because we want to ensure that we went through all impression-aware recommenders first.
        if "FoldedMF" in recommender_name:
            return "Baseline"

        # Covers framework recommenders.
        return "Baseline"

    def convert_recommender_name_to_experiment_type(recommender_name: str) -> str:
        if "ABLATION ONLY IMPRESSIONS FEATURES" in recommender_name:
            return "ABLATION OIF"

        if "ABLATION ONLY UIM FREQUENCY" in recommender_name:
            return "ABLATION UIM"

        if "SIGNAL ANALYSIS NEGATIVE ABLATION ONLY UIM FREQUENCY" in recommender_name:
            return "SIGNAL ANALYSIS NEGATIVE ABLATION ONLY UIM FREQUENCY"

        if "SIGNAL ANALYSIS NEGATIVE ABLATION ONLY UIM FREQUENCY" in recommender_name:
            return "SIGNAL ANALYSIS NEGATIVE ABLATION ONLY UIM FREQUENCY"

        if "SIGNAL ANALYSIS POSITIVE ABLATION ONLY UIM FREQUENCY" in recommender_name:
            return "SIGNAL ANALYSIS POSITIVE ABLATION ONLY UIM FREQUENCY"

        if "SIGNAL ANALYSIS SIGN ALL POSITIVE" in recommender_name:
            return "SIGNAL ANALYSIS SIGN ALL POSITIVE"

        if "SIGNAL ANALYSIS SIGN ALL NEGATIVE" in recommender_name:
            return "SIGNAL ANALYSIS SIGN ALL NEGATIVE"

        if "SIGNAL ANALYSIS SIGN POSITIVE" in recommender_name:
            return "SIGNAL ANALYSIS SIGN POSITIVE"

        if "SIGNAL ANALYSIS SIGN NEGATIVE" in recommender_name:
            return "SIGNAL ANALYSIS SIGN NEGATIVE"

        if "SIGNAL ANALYSIS LESS OR EQUAL THRESHOLD" in recommender_name:
            return "SIGNAL ANALYSIS LESS OR EQUAL THRESHOLD"

        if "SIGNAL ANALYSIS GREAT OR EQUAL THRESHOLD" in recommender_name:
            return "SIGNAL ANALYSIS GREAT OR EQUAL THRESHOLD"

        return ""

    def convert_model_base_to_model_base_order(model_base: str) -> float:
        model_base_order = 100.0

        if "Random" in model_base:
            model_base_order = 0.0
        if "TopPop" in model_base:
            model_base_order = 1.0

        if "ItemKNN" in model_base:
            model_base_order = 2.0
        if "UserKNN" in model_base:
            model_base_order = 3.0

        if "P3alpha" in model_base:
            model_base_order = 4.0
        if "RP3beta" in model_base:
            model_base_order = 5.0

        if "PureSVD" in model_base:
            model_base_order = 6.0
        if "NMF" in model_base:
            model_base_order = 7.0

        if (
            "MF SVDpp" in model_base
            or "SVD++" in model_base
            or "Funk SVD" in model_base
        ):
            model_base_order = 8.0
        if "MF BPR" in model_base:
            model_base_order = 9.0

        if "SLIM ElasticNet" in model_base:
            model_base_order = 10.0
        if "SLIM BPR" in model_base:
            model_base_order = 11.0

        if "EASE R" in model_base:
            model_base_order = 12.0

        if "LightFM" in model_base:
            model_base_order = 13.0

        if "MultVAE" in model_base:
            model_base_order = 14.0

        if "Last impression" in model_base:
            return 97.0
        if "Frequency & recency" in model_base:
            return 99.0
        if "Recency" in model_base:
            return 98.0

        if "asymmetric" in model_base:
            model_base_order += 0.1
        if "cosine" in model_base:
            model_base_order += 0.2
        if "dice" in model_base:
            model_base_order += 0.3
        if "jaccard" in model_base:
            model_base_order += 0.4
        if "tversky" in model_base:
            model_base_order += 0.5

        if "Folded" in model_base:
            model_base_order += 1000.0

        return model_base_order

    def convert_model_type_to_model_type_order(model_type: str) -> int:
        model_order = 100
        if model_type == "Baseline":
            model_order = 0
        if model_type == "Cycling":
            model_order = 1
        if model_type == "HFC":
            model_order = 2
        if model_type == "IDF":
            model_order = 3
        if model_type == "IUP":
            model_order = 4
        if model_type == "Baseline-IARS":
            model_order = 10

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
            index=False,
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
            index=False,
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
            index=False,
            header=True,
            encoding="utf-8",
            na_rep="-",
            sep=";",
            decimal=",",
            float_format="%.4f",
        )


def process_evaluation_results(
    *,
    benchmarks: list[commons.Benchmarks],
    hyper_parameters: list[commons.EHyperParameterTuningParameters],
    recommenders_baselines: list[commons.RecommenderBaseline],
    recommenders_impressions_heuristics: list[commons.RecommenderImpressions],
    recommenders_impressions_re_ranking: list[commons.RecommenderImpressions],
    recommenders_impressions_user_profiles: list[commons.RecommenderImpressions],
    recommenders_impressions_signal_analysis_re_ranking: list[
        tuple[commons.RecommenderImpressions, str]
    ],
    script_name: str,
    # TODO: INCLUDE signal analysis
    # recommenders_impressions_signal_analysis: list[commons.RecommenderImpressions],
    # baseline_experiment_cases_interface: commons.ExperimentCasesInterface,
    # impressions_heuristics_experiment_cases_interface: commons.ExperimentCasesInterface,
    # re_ranking_experiment_cases_interface: commons.ExperimentCasesInterface,
    # TODO: REMOVE ABLATION.
    # ablation_re_ranking_experiment_cases_interface: commons.ExperimentCasesInterface,
    # user_profiles_experiment_cases_interface: commons.ExperimentCasesInterface,
) -> None:
    """
    Public method that exports into CSV and LaTeX tables the evaluation metrics, hyper-parameters, and times.
    """
    printed_experiments: set[
        tuple[commons.Benchmarks, commons.EHyperParameterTuningParameters]
    ] = set()

    for benchmark, hyper_parameter in itertools.product(benchmarks, hyper_parameters):
        if (benchmark, hyper_parameter) in printed_experiments:
            continue

        printed_experiments.add((benchmark, hyper_parameter))

        experiment_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[benchmark]
        experiment_hyper_parameters = (
            commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[hyper_parameter]
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
            script_name=script_name,
            benchmark=experiment_benchmark.benchmark.value,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
        )
        folder_path_export_csv = DIR_CSV_RESULTS.format(
            script_name=script_name,
            benchmark=experiment_benchmark.benchmark.value,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
        )
        folder_path_export_parquet = DIR_PARQUET_RESULTS.format(
            script_name=script_name,
            benchmark=experiment_benchmark.benchmark.value,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
        )

        os.makedirs(folder_path_export_latex, exist_ok=True)
        os.makedirs(folder_path_export_csv, exist_ok=True)
        os.makedirs(folder_path_export_parquet, exist_ok=True)

        knn_similarity_list = experiment_hyper_parameters.knn_similarity_types

        results_baselines = _print_baselines_metrics(
            recommenders_baselines=recommenders_baselines,
            experiment_benchmark=experiment_benchmark,
            experiment_hyper_parameters=experiment_hyper_parameters,
            interaction_data_splits=interaction_data_splits,
            num_test_users=num_test_users,
            accuracy_metrics_list=ACCURACY_METRICS_LIST,
            beyond_accuracy_metrics_list=BEYOND_ACCURACY_METRICS_LIST,
            all_metrics_list=ALL_METRICS_LIST,
            cutoffs_list=RESULT_EXPORT_CUTOFFS,
            knn_similarity_list=knn_similarity_list,
            export_experiments_folder_path=folder_path_export_latex,
        )

        results_heuristics = _print_impressions_heuristics_metrics(
            recommenders_impressions_heuristics=recommenders_impressions_heuristics,
            experiment_hyper_parameters=experiment_hyper_parameters,
            experiment_benchmark=experiment_benchmark,
            num_test_users=num_test_users,
            accuracy_metrics_list=ACCURACY_METRICS_LIST,
            beyond_accuracy_metrics_list=BEYOND_ACCURACY_METRICS_LIST,
            all_metrics_list=ALL_METRICS_LIST,
            cutoffs_list=RESULT_EXPORT_CUTOFFS,
            export_experiments_folder_path=folder_path_export_latex,
        )

        results_re_ranking = _print_impressions_re_ranking_metrics(
            recommenders_baselines=recommenders_baselines,
            recommenders_impressions_re_ranking=recommenders_impressions_re_ranking,
            experiment_benchmark=experiment_benchmark,
            experiment_hyper_parameters=experiment_hyper_parameters,
            interaction_data_splits=interaction_data_splits,
            num_test_users=num_test_users,
            accuracy_metrics_list=ACCURACY_METRICS_LIST,
            beyond_accuracy_metrics_list=BEYOND_ACCURACY_METRICS_LIST,
            all_metrics_list=ALL_METRICS_LIST,
            cutoffs_list=RESULT_EXPORT_CUTOFFS,
            knn_similarity_list=knn_similarity_list,
            export_experiments_folder_path=folder_path_export_latex,
        )

        results_user_profiles = _print_impressions_user_profiles_metrics(
            recommenders_baselines=recommenders_baselines,
            recommenders_impressions_user_profiles=recommenders_impressions_user_profiles,
            experiment_hyper_parameters=experiment_hyper_parameters,
            experiment_benchmark=experiment_benchmark,
            interaction_data_splits=interaction_data_splits,
            num_test_users=num_test_users,
            accuracy_metrics_list=ACCURACY_METRICS_LIST,
            beyond_accuracy_metrics_list=BEYOND_ACCURACY_METRICS_LIST,
            all_metrics_list=ALL_METRICS_LIST,
            cutoffs_list=RESULT_EXPORT_CUTOFFS,
            knn_similarity_list=knn_similarity_list,
            export_experiments_folder_path=folder_path_export_latex,
        )

        results_signal_analysis_re_ranking = _print_impressions_signal_analysis_re_ranking_metrics(
            recommenders_baselines=recommenders_baselines,
            recommenders_impressions_signal_analysis_re_ranking=recommenders_impressions_signal_analysis_re_ranking,
            experiment_benchmark=experiment_benchmark,
            experiment_hyper_parameters=experiment_hyper_parameters,
            interaction_data_splits=interaction_data_splits,
            num_test_users=num_test_users,
            accuracy_metrics_list=ACCURACY_METRICS_LIST,
            beyond_accuracy_metrics_list=BEYOND_ACCURACY_METRICS_LIST,
            all_metrics_list=ALL_METRICS_LIST,
            cutoffs_list=RESULT_EXPORT_CUTOFFS,
            knn_similarity_list=knn_similarity_list,
            export_experiments_folder_path=folder_path_export_latex,
        )

        # TODO: REMOVE ABLATION.
        # results_ablation_re_ranking = _print_ablation_impressions_re_ranking_metrics(
        #     ablation_re_ranking_experiment_cases_interface=ablation_re_ranking_experiment_cases_interface,
        #     baseline_experiment_cases_interface=baseline_experiment_cases_interface,
        #     baseline_experiment_benchmark=experiment_benchmark,
        #     baseline_experiment_hyper_parameters=experiment_hyper_parameters,
        #     interaction_data_splits=interaction_data_splits,
        #     num_test_users=num_test_users,
        #     accuracy_metrics_list=ACCURACY_METRICS_LIST,
        #     beyond_accuracy_metrics_list=BEYOND_ACCURACY_METRICS_LIST,
        #     all_metrics_list=ALL_METRICS_LIST,
        #     cutoffs_list=RESULT_EXPORT_CUTOFFS,
        #     knn_similarity_list=knn_similarity_list,
        #     export_experiments_folder_path=folder_path_export_latex,
        # )

        _process_results_dataframe(
            dfs=[
                results_baselines.df_results,
                results_heuristics.df_results,
                results_re_ranking.df_results,
                results_user_profiles.df_results,
                results_signal_analysis_re_ranking.df_results,
                # TODO: REMOVE ABLATION.
                # results_ablation_re_ranking.df_results,
            ],
            results_name="accuracy-metrics",
            benchmark=benchmark,
            folder_path_csv=folder_path_export_csv,
            folder_path_parquet=folder_path_export_parquet,
        )

        _process_results_dataframe(
            dfs=[
                results_baselines.df_times,
                results_heuristics.df_times,
                results_re_ranking.df_times,
                results_user_profiles.df_times,
                results_signal_analysis_re_ranking.df_times,
                # TODO: REMOVE ABLATION.
                # results_ablation_re_ranking.df_times,
            ],
            results_name="times",
            benchmark=benchmark,
            folder_path_csv=folder_path_export_csv,
            folder_path_parquet=folder_path_export_parquet,
        )

        _process_results_dataframe(
            dfs=[
                results_baselines.df_hyper_params,
                results_heuristics.df_hyper_params,
                results_re_ranking.df_hyper_params,
                results_user_profiles.df_hyper_params,
                results_signal_analysis_re_ranking.df_hyper_params,
                # TODO: REMOVE ABLATION
                # results_ablation_re_ranking.df_hyper_params,
            ],
            results_name="hyper-parameters",
            benchmark=benchmark,
            folder_path_csv=folder_path_export_csv,
            folder_path_parquet=folder_path_export_parquet,
        )

        logger.info(
            "Successfully finished processing accuracy, beyond-accuracy, hyper-parameters, and time results to CSV and parquet. Files are located at two locations. \n\t* CSV path: %(csv_path)s. \n\t* Parquet path: %(parquet_path)s",
            {
                "csv_path": folder_path_export_csv,
                "parquet_path": folder_path_export_parquet,
            },
        )


def export_evaluation_results(
    *,
    benchmarks: list[commons.Benchmarks],
    hyper_parameters: list[commons.EHyperParameterTuningParameters],
    script_name: str,
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
            script_name=script_name,
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

    df_results_accuracy = pd.concat(
        objs=results_accuracy_metrics,
        axis=0,
        ignore_index=True,  # The index should be numeric and has no special meaning.
    )
    df_results_times = pd.concat(
        objs=results_times,
        axis=0,
        ignore_index=True,  # The index should be numeric and has no special meaning.
    )
    df_results_hyper_parameters = pd.concat(
        objs=results_hyper_parameters,
        axis=0,
        ignore_index=True,  # The index should be numeric and has no special meaning.
    )

    folder_path_results_to_export = DIR_RESULTS_TO_EXPORT.format(
        script_name=script_name
    )
    os.makedirs(folder_path_results_to_export, exist_ok=True)

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
