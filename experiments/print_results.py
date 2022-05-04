import itertools
import os
from typing import Type, Optional, cast, Union

import Recommenders.Recommender_import_list as recommenders
import numpy as np
import pandas as pd
import scipy.sparse as sp
from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.BaseSimilarityMatrixRecommender import BaseSimilarityMatrixRecommender, \
    BaseUserSimilarityMatrixRecommender, BaseItemSimilarityMatrixRecommender
from recsys_framework_extensions.data.io import DataIO
from recsys_framework_extensions.data.mixins import InteractionsDataSplits
from recsys_framework_extensions.logging import get_logger
from recsys_framework_extensions.plotting import generate_accuracy_and_beyond_metrics_pandas, DataFrameResults

import experiments.baselines as baselines
import experiments.commons as commons
import experiments.heuristics as heuristics
import experiments.re_ranking as re_ranking
import experiments.user_profiles as user_profiles
from impression_recommenders.re_ranking.cycling import CyclingRecommender
from impression_recommenders.re_ranking.impressions_discounting import ImpressionsDiscountingRecommender
from impression_recommenders.user_profile.folding import FoldedMatrixFactorizationRecommender
from impression_recommenders.user_profile.weighted import BaseWeightedUserProfileRecommender, \
    ItemWeightedUserProfileRecommender, UserWeightedUserProfileRecommender

logger = get_logger(__name__)


####################################################################################################
####################################################################################################
#                                REPRODUCIBILITY VARIABLES                            #
####################################################################################################
####################################################################################################
BASE_FOLDER = os.path.join(
    commons.RESULTS_EXPERIMENTS_DIR,
    "results_export",
    "{benchmark}",
    "{evaluation_strategy}",
    "",
)
DIR_CSV_RESULTS = os.path.join(
    BASE_FOLDER,
    "csv",
    "",
)
DIR_PARQUET_RESULTS = os.path.join(
    BASE_FOLDER,
    "parquet",
    "",
)
DIR_LATEX_RESULTS = os.path.join(
    BASE_FOLDER,
    "latex",
    "",
)
DIR_ARTICLE_ACCURACY_METRICS_BASELINES_LATEX = os.path.join(
    DIR_LATEX_RESULTS,
    "article-accuracy_and_beyond_accuracy",
    "",
)
DIR_ACCURACY_METRICS_BASELINES_LATEX = os.path.join(
    DIR_LATEX_RESULTS,
    "accuracy_and_beyond_accuracy",
    "",
)

commons.FOLDERS.add(BASE_FOLDER)
commons.FOLDERS.add(DIR_CSV_RESULTS)
commons.FOLDERS.add(DIR_PARQUET_RESULTS)
commons.FOLDERS.add(DIR_ACCURACY_METRICS_BASELINES_LATEX)
commons.FOLDERS.add(DIR_ARTICLE_ACCURACY_METRICS_BASELINES_LATEX)

RESULT_EXPORT_CUTOFFS = [20]

ACCURACY_METRICS_LIST = [
    "PRECISION",
    "RECALL",
    "MAP",
    "MRR",
    "NDCG",
    "F1",
]
BEYOND_ACCURACY_METRICS_LIST = [
    "NOVELTY",
    "DIVERSITY_MEAN_INTER_LIST",
    "COVERAGE_ITEM",
    "DIVERSITY_GINI",
    "SHANNON_ENTROPY"
]
ALL_METRICS_LIST = [
    *ACCURACY_METRICS_LIST,
    *BEYOND_ACCURACY_METRICS_LIST,
]


ARTICLE_BASELINES: list[Type[BaseRecommender]] = [
    recommenders.Random,
    recommenders.TopPop,
    recommenders.UserKNNCFRecommender,
    recommenders.ItemKNNCFRecommender,
    recommenders.RP3betaRecommender,
    recommenders.PureSVDRecommender,
    recommenders.NMFRecommender,
    recommenders.IALSRecommender,
    recommenders.SLIMElasticNetRecommender,
    recommenders.SLIM_BPR_Cython,
    recommenders.MatrixFactorization_BPR_Cython,
    recommenders.LightFMCFRecommender,
    recommenders.MultVAERecommender,
    # recommenders.EASE_R_Recommender,
]
ARTICLE_KNN_SIMILARITY_LIST: list[commons.T_SIMILARITY_TYPE] = [
    "asymmetric",
]
ARTICLE_CUTOFF = [20]
ARTICLE_ACCURACY_METRICS_LIST = [
    "PRECISION",
    "RECALL",
    "MRR",
    "NDCG",
]
ARTICLE_BEYOND_ACCURACY_METRICS_LIST = [
    "NOVELTY",
    "COVERAGE_ITEM",
    "DIVERSITY_MEAN_INTER_LIST",
    "DIVERSITY_GINI",
]
ARTICLE_ALL_METRICS_LIST = [
    *ARTICLE_ACCURACY_METRICS_LIST,
    *ARTICLE_BEYOND_ACCURACY_METRICS_LIST,
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
            f"{list(baselines.TrainedRecommenderType)}")

    recommender_name = f"{experiment_recommender.recommender.RECOMMENDER_NAME}"
    if experiment_recommender.recommender in [recommenders.ItemKNNCFRecommender, recommenders.UserKNNCFRecommender]:
        assert similarity is not None
        assert similarity in experiment_hyper_parameter_tuning_parameters.knn_similarity_types

        recommender_name = f"{experiment_recommender.recommender.RECOMMENDER_NAME}_{similarity}"

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
                np.array([], np.float32)
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
    baseline_experiment_cases_interface: commons.ExperimentCasesInterface,
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
    experiments_folder_path = baselines.HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR.format(
        benchmark=experiment_benchmark.benchmark.value,
        evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
    )

    baseline_experiment_recommenders = [
        commons.MAPPER_AVAILABLE_RECOMMENDERS[rec]
        for rec in baseline_experiment_cases_interface.to_use_recommenders
    ]

    base_algorithm_list = []
    for baseline_experiment_recommender in baseline_experiment_recommenders:
        base_algorithm_list.append(baseline_experiment_recommender.recommender)

        similarities: list[commons.T_SIMILARITY_TYPE] = [None]  # type: ignore
        if baseline_experiment_recommender.recommender in [recommenders.ItemKNNCFRecommender, recommenders.UserKNNCFRecommender]:
            similarities = knn_similarity_list

        for similarity in similarities:
            # This inner loop is to load Folded Recommenders. If we cannot mock a Folded recommender,
            # then this method returns None.
            loaded_recommender = mock_trained_recommender(
                experiment_recommender=baseline_experiment_recommender,
                experiment_benchmark=experiment_benchmark,
                experiment_hyper_parameter_tuning_parameters=experiment_hyper_parameters,
                experiment_model_dir=experiments_folder_path,
                data_splits=interaction_data_splits,
                similarity=similarity,
                model_type=baselines.TrainedRecommenderType.TRAIN_VALIDATION,
                try_folded_recommender=True,
            )

            if loaded_recommender is None:
                logger.warning(
                    f"The recommender {baseline_experiment_recommender.recommender} for the dataset {experiment_benchmark} "
                    f"returned empty. Skipping."
                )
                continue

            base_algorithm_list.append(loaded_recommender)

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
        icm_names=None
    )


def _print_impressions_heuristics_metrics(
    heuristics_experiment_cases_interface: commons.ExperimentCasesInterface,
    experiment_hyper_parameters: commons.HyperParameterTuningParameters,
    experiment_benchmark: commons.ExperimentBenchmark,
    num_test_users: int,
    accuracy_metrics_list: list[str],
    beyond_accuracy_metrics_list: list[str],
    all_metrics_list: list[str],
    cutoffs_list: list[int],
    export_experiments_folder_path: str,
) -> DataFrameResults:
    experiments_folder_path = heuristics.HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR.format(
        benchmark=experiment_benchmark.benchmark.value,
        evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
    )

    impressions_heuristics_recommenders = [
        commons.MAPPER_AVAILABLE_RECOMMENDERS[rec].recommender
        for rec in heuristics_experiment_cases_interface.to_use_recommenders
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
        icm_names=None
    )


def _print_impressions_re_ranking_metrics(
    re_ranking_experiment_cases_interface: commons.ExperimentCasesInterface,
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
    baseline_experiments_folder_path = baselines.HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR.format(
        benchmark=baseline_experiment_benchmark.benchmark.value,
        evaluation_strategy=baseline_experiment_hyper_parameters.evaluation_strategy.value,
    )

    re_ranking_experiments_folder_path = re_ranking.HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR.format(
        benchmark=baseline_experiment_benchmark.benchmark.value,
        evaluation_strategy=baseline_experiment_hyper_parameters.evaluation_strategy.value,
    )

    baseline_experiment_recommenders = [
        commons.MAPPER_AVAILABLE_RECOMMENDERS[rec]
        for rec in baseline_experiment_cases_interface.to_use_recommenders
    ]

    re_ranking_experiment_recommenders = [
        commons.MAPPER_AVAILABLE_RECOMMENDERS[rec]
        for rec in re_ranking_experiment_cases_interface.to_use_recommenders
    ]

    base_algorithm_list = []
    for re_ranking_experiment_recommender in re_ranking_experiment_recommenders:
        for baseline_experiment_recommender in baseline_experiment_recommenders:
            similarities: list[commons.T_SIMILARITY_TYPE] = [None]  # type: ignore
            if baseline_experiment_recommender.recommender in [
                recommenders.ItemKNNCFRecommender,
                recommenders.UserKNNCFRecommender
            ]:
                similarities = knn_similarity_list

            for similarity in similarities:
                for try_folded_recommender in [True, False]:
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
                        Type[Union[CyclingRecommender, ImpressionsDiscountingRecommender]],
                        re_ranking_experiment_recommender.recommender
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
        icm_names=None
    )


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
    baseline_experiments_folder_path = baselines.HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR.format(
        benchmark=baseline_experiment_benchmark.benchmark.value,
        evaluation_strategy=baseline_experiment_hyper_parameters.evaluation_strategy.value,
    )

    re_ranking_experiments_folder_path = re_ranking.HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR.format(
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
                recommenders.UserKNNCFRecommender
            ]:
                similarities = knn_similarity_list

            for similarity in similarities:
                for try_folded_recommender in [True, False]:
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
                        Type[Union[CyclingRecommender, ImpressionsDiscountingRecommender]],
                        re_ranking_experiment_recommender.recommender
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
        icm_names=None
    )


def _print_impressions_user_profiles_metrics(
    baseline_experiment_cases_interface: commons.ExperimentCasesInterface,
    user_profiles_experiment_cases_interface: commons.ExperimentCasesInterface,
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
    baseline_experiments_folder_path = baselines.HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR.format(
        benchmark=baseline_experiment_benchmark.benchmark.value,
        evaluation_strategy=baseline_experiment_hyper_parameters.evaluation_strategy.value,
    )

    user_profiles_experiments_folder_path = user_profiles.HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR.format(
        benchmark=baseline_experiment_benchmark.benchmark.value,
        evaluation_strategy=baseline_experiment_hyper_parameters.evaluation_strategy.value,
    )

    baseline_experiment_recommenders = [
        commons.MAPPER_AVAILABLE_RECOMMENDERS[rec]
        for rec in baseline_experiment_cases_interface.to_use_recommenders
    ]

    user_profiles_experiment_recommenders = [
        commons.MAPPER_AVAILABLE_RECOMMENDERS[rec]
        for rec in user_profiles_experiment_cases_interface.to_use_recommenders
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
                recommenders.UserKNNCFRecommender
            ]:
                similarities = knn_similarity_list

            for similarity in similarities:
                for try_folded_recommender in [True, False]:
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
                        logger.warning(
                            f"The recommender {baseline_experiment_recommender.recommender} for the dataset "
                            f"{baseline_experiment_benchmark} returned empty. Skipping."
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
                        user_profiles_experiment_recommender.recommender
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
        icm_names=None
    )


def _model_orders(value):
    if value == "Baseline":
        return 0
    if value == "Folded":
        return 1
    if value == "Cycling":
        return 2
    if value == "Cycling Folded":
        return 3
    if value == "Impressions Discounting":
        return 4
    if value == "Impressions Discounting Folded":
        return 5
    if value == "Item Weighted Profile":
        return 6
    if value == "Item Weighted Profile Folded":
        return 7
    if value == "User Weighted Profile":
        return 8
    if value == "User Weighted Profile Folded":
        return 9
    else:
        return 20


def _results_to_pandas(
    df_baselines: pd.DataFrame,
    df_heuristics: pd.DataFrame,
    df_re_ranking: pd.DataFrame,
    df_ablation_re_ranking: pd.DataFrame,
    df_user_profiles: pd.DataFrame,
    results_name: str,
    folder_path_latex: str,
    folder_path_csv: str,
    folder_path_parquet: str,
) -> None:
    MODEL_COLUMN = "Model"
    CUTOFF_COLUMN = "Cutoff"
    MODEL_BASE_COLUMN = "Model Name"
    MODEL_TYPE_COLUMN = "Model Type"
    ORDER_COLUMN = "Order"

    df_results: pd.DataFrame = pd.concat(
        [
            df_baselines,
            df_heuristics,
            df_re_ranking,
            df_ablation_re_ranking,
            df_user_profiles,
        ],
        axis=0,
        ignore_index=False,  # The index is the list of recommender names.
    )

    if "accuracy-metrics" == results_name:
        df_results = (
            df_results
                .stack(0, dropna=False)  # Sets the @20 column as index.
                .reset_index(drop=False)  # Makes the @20 column as another column.
                .rename(
                    columns={
                        "level_0": MODEL_COLUMN,
                        "level_1": CUTOFF_COLUMN,
                        "algorithm_row_label": MODEL_COLUMN,
                    })
        )
    elif "times" == results_name:
        df_results = (
            df_results
                .reset_index(drop=False)  # Makes the @20 column as another column.
                .rename(
                    columns={
                        "level_0": MODEL_COLUMN,
                        "index": MODEL_COLUMN,
                    })
        )
    elif "hyper-parameters" == results_name:
        # Resulting dataframe
        # Index: (algorithm_row_label, hyperparameter_name)
        # Columns: [hyperparameter_value]
        df_results = (
            df_results
                .reset_index(drop=False)
                .rename(
                    columns={
                        "algorithm_row_label": MODEL_COLUMN,
                        "hyperparameter_name": "Hyper-Parameter",
                        "hyperparameter_value": "Value",
                    })
        )
    else:
        return

    df_results[MODEL_COLUMN] = (
        df_results[MODEL_COLUMN]
            .str.replace("ImpressionsDiscounting", "Impressions Discounting")
            .str.replace("FrequencyRecency", "Frequency & Recency")
            .str.replace("LastImpressions", "Last Impressions")
            .str.replace("ItemWeightedUserProfile", "Impressions as User Profiles")
            .str.replace("UserWeightedUserProfile", "Impressions as User Profiles")
            .str.replace("FoldedMF", "Folded")
            .str.replace("ABLATION UIM FREQUENCY", "Ablation")
            .str.strip()
    )

    df_results[MODEL_BASE_COLUMN] = (
        df_results[MODEL_COLUMN]
            .str.replace("Recommender", "")
            .str.replace("Cycling", "")
            .str.replace("Impressions Discounting", "")
            .str.replace("KNNCF", "KNN CF")
            .str.replace("Item Weighted Profile", "")
            .str.replace("User Weighted Profile", "")
            .str.replace("Impressions as User Profiles", "")
            .str.replace("Folded", "")
            .str.replace("ABLATION UIM FREQUENCY", "")
            .str.replace("Ablation", "")
            .str.strip()
    )

    df_results[MODEL_TYPE_COLUMN] = (
        df_results[MODEL_COLUMN]
            .str.replace("asymmetric", "")
            .str.replace("cosine", "")
            .str.replace("dice", "")
            .str.replace("jaccard", "")
            .str.replace("tversky", "")

            .str.replace("CF", "")

            .str.replace("AsySVD", "")
            .str.replace("BPR", "")
            .str.replace("FunkSVD", "")
            .str.replace("ElasticNet", "")

            .str.replace("ItemKNN", "")
            .str.replace("UserKNN", "")

            .str.replace("PureSVD", "")
            .str.replace("NMF", "")
            .str.replace("IALS", "")
            .str.replace("MF", "")
            .str.replace("MF AsySVD", "")
            .str.replace("MF BPR", "")
            .str.replace("MF FunkSVD", "")

            .str.replace("P3alpha", "")
            .str.replace("RP3beta", "")

            .str.replace("SLIM", "")
            .str.replace("SLIM BPR", "")
            .str.replace("SLIM ElasticNet", "")

            .str.replace("EASE R", "")
            .str.replace("FM", "")
            .str.replace("Light", "")
            .str.replace("LightFM", "")
            .str.replace("MultVAE", "")

            .str.replace("GlobalEffects", "")
            .str.replace("Random", "")
            .str.replace("TopPop", "")

            .str.replace("Last Impressions", "")
            .str.replace("Frequency & Recency", "")
            .str.replace("Recency", "")
            .str.replace("Item Weighted Profile Folded", "Impressions as User Profiles")
            .str.replace("User Weighted Profile Folded", "Impressions as User Profiles")
            .str.replace("Impressions as User Profiles Folded", "Impressions as User Profiles")

            .str.strip()
    )
    df_results[MODEL_TYPE_COLUMN] = df_results[MODEL_TYPE_COLUMN].where(
        df_results[MODEL_TYPE_COLUMN] != "", "Baseline"
    )

    df_results[ORDER_COLUMN] = df_results[MODEL_TYPE_COLUMN].apply(
        _model_orders
    )

    if results_name == "accuracy-metrics":
        df_results = df_results.sort_values(
            by=[CUTOFF_COLUMN, MODEL_BASE_COLUMN, ORDER_COLUMN],
            ascending=True,
            inplace=False,
            ignore_index=False,
        )

        df_results_pivoted = df_results.pivot(
            index=[MODEL_BASE_COLUMN, CUTOFF_COLUMN],
            columns=[MODEL_TYPE_COLUMN],
            values=["NDCG", "F1", "PRECISION", "RECALL", "DIVERSITY_MEAN_INTER_LIST", "COVERAGE_ITEM", "DIVERSITY_GINI"]
        )
    elif "times" == results_name:
        df_results = df_results.sort_values(
            by=[MODEL_BASE_COLUMN, ORDER_COLUMN],
            ascending=True,
            inplace=False,
            ignore_index=False,
        )

        df_results_pivoted = df_results.pivot(
            index=[MODEL_BASE_COLUMN],
            columns=[MODEL_TYPE_COLUMN],
            values=["Train Time", "Recommendation Time", "Recommendation Throughput"]
        )
    else:
        df_results = df_results.sort_values(
            by=[MODEL_BASE_COLUMN, ORDER_COLUMN],
            ascending=True,
            inplace=False,
            ignore_index=False,
        )
        df_results_pivoted = df_results

    with pd.option_context("max_colwidth", 1000):
        df_results_pivoted.to_csv(
            path_or_buf=os.path.join(folder_path_csv, f"pivot-{results_name}.csv"),
            index=True,
            header=True,
            encoding="utf-8",
            na_rep="-",
        )
        df_results.to_csv(
            path_or_buf=os.path.join(folder_path_csv, f"{results_name}.csv"),
            index=True,
            header=True,
            encoding="utf-8",
            na_rep="-",
        )

        df_results_pivoted.to_latex(
            buf=os.path.join(folder_path_latex, f"pivot-{results_name}.tex"),
            index=True,
            header=True,
            escape=False,
            float_format="{:.4f}".format,
            encoding="utf-8",
            na_rep="-",
            longtable=True,
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
    baseline_experiment_cases_interface: commons.ExperimentCasesInterface,
    impressions_heuristics_experiment_cases_interface: commons.ExperimentCasesInterface,
    re_ranking_experiment_cases_interface: commons.ExperimentCasesInterface,
    ablation_re_ranking_experiment_cases_interface: commons.ExperimentCasesInterface,
    user_profiles_experiment_cases_interface: commons.ExperimentCasesInterface,
) -> None:
    printed_experiments: set[tuple[commons.Benchmarks, commons.EHyperParameterTuningParameters]] = set()

    baseline_benchmarks = baseline_experiment_cases_interface.to_use_benchmarks
    baseline_hyper_parameters = baseline_experiment_cases_interface.to_use_hyper_parameter_tuning_parameters

    for benchmark, hyper_parameters in itertools.product(baseline_benchmarks, baseline_hyper_parameters):
        if (benchmark, hyper_parameters) in printed_experiments:
            continue
        else:
            printed_experiments.add((benchmark, hyper_parameters))

        experiment_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
            benchmark
        ]
        experiment_hyper_parameters = commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
            hyper_parameters
        ]

        data_reader = commons.get_reader_from_benchmark(
            benchmark_config=experiment_benchmark.config,
            benchmark=experiment_benchmark.benchmark,
        )

        dataset = data_reader.dataset
        interaction_data_splits = dataset.get_urm_splits(
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy,
        )

        urm_test = interaction_data_splits.sp_urm_test
        num_test_users = cast(
            int,
            np.sum(
                np.ediff1d(urm_test.indptr) >= 1
            )
        )

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

        results_baselines = _print_baselines_metrics(
            baseline_experiment_cases_interface=baseline_experiment_cases_interface,
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
            heuristics_experiment_cases_interface=impressions_heuristics_experiment_cases_interface,
            experiment_hyper_parameters=experiment_hyper_parameters,
            experiment_benchmark=experiment_benchmark,
            num_test_users=num_test_users,
            accuracy_metrics_list=ACCURACY_METRICS_LIST,
            beyond_accuracy_metrics_list=BEYOND_ACCURACY_METRICS_LIST,
            all_metrics_list=ALL_METRICS_LIST,
            cutoffs_list=RESULT_EXPORT_CUTOFFS,
            export_experiments_folder_path=folder_path_export_latex
        )

        results_re_ranking = _print_impressions_re_ranking_metrics(
            re_ranking_experiment_cases_interface=re_ranking_experiment_cases_interface,
            baseline_experiment_cases_interface=baseline_experiment_cases_interface,
            baseline_experiment_benchmark=experiment_benchmark,
            baseline_experiment_hyper_parameters=experiment_hyper_parameters,
            interaction_data_splits=interaction_data_splits,
            num_test_users=num_test_users,
            accuracy_metrics_list=ACCURACY_METRICS_LIST,
            beyond_accuracy_metrics_list=BEYOND_ACCURACY_METRICS_LIST,
            all_metrics_list=ALL_METRICS_LIST,
            cutoffs_list=RESULT_EXPORT_CUTOFFS,
            knn_similarity_list=knn_similarity_list,
            export_experiments_folder_path=folder_path_export_latex,
        )

        results_ablation_re_ranking = _print_ablation_impressions_re_ranking_metrics(
            ablation_re_ranking_experiment_cases_interface=ablation_re_ranking_experiment_cases_interface,
            baseline_experiment_cases_interface=baseline_experiment_cases_interface,
            baseline_experiment_benchmark=experiment_benchmark,
            baseline_experiment_hyper_parameters=experiment_hyper_parameters,
            interaction_data_splits=interaction_data_splits, num_test_users=num_test_users,
            accuracy_metrics_list=ACCURACY_METRICS_LIST, beyond_accuracy_metrics_list=BEYOND_ACCURACY_METRICS_LIST,
            all_metrics_list=ALL_METRICS_LIST, cutoffs_list=RESULT_EXPORT_CUTOFFS,
            knn_similarity_list=knn_similarity_list, export_experiments_folder_path=folder_path_export_latex)

        results_user_profiles = _print_impressions_user_profiles_metrics(
            user_profiles_experiment_cases_interface=user_profiles_experiment_cases_interface,
            baseline_experiment_cases_interface=baseline_experiment_cases_interface,
            baseline_experiment_hyper_parameters=experiment_hyper_parameters,
            baseline_experiment_benchmark=experiment_benchmark,
            interaction_data_splits=interaction_data_splits,
            num_test_users=num_test_users,
            accuracy_metrics_list=ACCURACY_METRICS_LIST,
            beyond_accuracy_metrics_list=BEYOND_ACCURACY_METRICS_LIST,
            all_metrics_list=ALL_METRICS_LIST,
            cutoffs_list=RESULT_EXPORT_CUTOFFS,
            knn_similarity_list=knn_similarity_list,
            export_experiments_folder_path=folder_path_export_latex,
        )

        _results_to_pandas(
            df_baselines=results_baselines.df_results,
            df_heuristics=results_heuristics.df_results,
            df_re_ranking=results_re_ranking.df_results,
            df_ablation_re_ranking=results_ablation_re_ranking.df_results,
            df_user_profiles=results_user_profiles.df_results,
            results_name="accuracy-metrics",
            folder_path_latex=folder_path_export_latex,
            folder_path_csv=folder_path_export_csv,
            folder_path_parquet=folder_path_export_parquet,
        )

        _results_to_pandas(
            df_baselines=results_baselines.df_times,
            df_heuristics=results_heuristics.df_times,
            df_re_ranking=results_re_ranking.df_times,
            df_ablation_re_ranking=results_ablation_re_ranking.df_times,
            df_user_profiles=results_user_profiles.df_times,
            results_name="times",
            folder_path_latex=folder_path_export_latex,
            folder_path_csv=folder_path_export_csv,
            folder_path_parquet=folder_path_export_parquet,
        )

        _results_to_pandas(
            df_baselines=results_baselines.df_hyper_params,
            df_heuristics=results_heuristics.df_hyper_params,
            df_re_ranking=results_re_ranking.df_hyper_params,
            df_ablation_re_ranking=results_ablation_re_ranking.df_hyper_params,
            df_user_profiles=results_user_profiles.df_hyper_params,
            results_name="hyper-parameters",
            folder_path_latex=folder_path_export_latex,
            folder_path_csv=folder_path_export_csv,
            folder_path_parquet=folder_path_export_parquet,
        )

        logger.info(
            f"Successfully finished exporting accuracy and beyond-accuracy results to LaTeX"
        )
