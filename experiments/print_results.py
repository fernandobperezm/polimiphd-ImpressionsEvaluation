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
    "latex",
    "{benchmark}",
    "{evaluation_strategy}",
    "",
)
ARTICLE_ACCURACY_METRICS_BASELINES_LATEX_DIR = os.path.join(
    BASE_FOLDER,
    "article-accuracy_and_beyond_accuracy",
    "",
)
ACCURACY_METRICS_BASELINES_LATEX_DIR = os.path.join(
    BASE_FOLDER,
    "accuracy_and_beyond_accuracy",
    "",
)

commons.FOLDERS.add(BASE_FOLDER)
commons.FOLDERS.add(ACCURACY_METRICS_BASELINES_LATEX_DIR)
commons.FOLDERS.add(ARTICLE_ACCURACY_METRICS_BASELINES_LATEX_DIR)

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
#                               Utility to load an already-tuned recommender                                #
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

    if try_folded_recommender and can_recommender_be_folded:
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

            if isinstance(loaded_recommender, FoldedMatrixFactorizationRecommender):
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
                loaded_baseline_recommender = mock_trained_recommender(
                    experiment_recommender=baseline_experiment_recommender,
                    experiment_benchmark=baseline_experiment_benchmark,
                    experiment_hyper_parameter_tuning_parameters=baseline_experiment_hyper_parameters,
                    experiment_model_dir=baseline_experiments_folder_path,
                    data_splits=interaction_data_splits,
                    similarity=similarity,
                    model_type=baselines.TrainedRecommenderType.TRAIN_VALIDATION,
                    try_folded_recommender=True,
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
                loaded_baseline_recommender = mock_trained_recommender(
                    experiment_recommender=baseline_experiment_recommender,
                    experiment_benchmark=baseline_experiment_benchmark,
                    experiment_hyper_parameter_tuning_parameters=baseline_experiment_hyper_parameters,
                    experiment_model_dir=baseline_experiments_folder_path,
                    data_splits=interaction_data_splits,
                    similarity=similarity,
                    model_type=baselines.TrainedRecommenderType.TRAIN_VALIDATION,
                    try_folded_recommender=True,
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


def print_results(
    baseline_experiment_cases_interface: commons.ExperimentCasesInterface,
    impressions_heuristics_experiment_cases_interface: commons.ExperimentCasesInterface,
    re_ranking_experiment_cases_interface: commons.ExperimentCasesInterface,
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

        export_experiments_folder_path = ACCURACY_METRICS_BASELINES_LATEX_DIR.format(
            benchmark=experiment_benchmark.benchmark.value,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
        )
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
            export_experiments_folder_path=export_experiments_folder_path,
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
            export_experiments_folder_path=export_experiments_folder_path
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
            export_experiments_folder_path=export_experiments_folder_path,
        )

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
            export_experiments_folder_path=export_experiments_folder_path,
        )

        results_metrics: pd.DataFrame = pd.concat(
            [
                results_baselines.df_results,
                results_heuristics.df_results,
                results_re_ranking.df_results,
                results_user_profiles.df_results,
            ],
            axis=0,
            ignore_index=False,  # The index is the list of recommender names.
        )

        results_times: pd.DataFrame = pd.concat(
            [
                results_baselines.df_times,
                results_heuristics.df_times,
                results_re_ranking.df_times,
                results_user_profiles.df_times,
            ],
            axis=0,
            ignore_index=False,
        )

        results_hyper_parameters: pd.DataFrame = pd.concat(
            [
                results_baselines.df_hyper_params,
                results_heuristics.df_hyper_params,
                results_re_ranking.df_hyper_params,
                results_user_profiles.df_hyper_params,
            ],
            axis=0,
            ignore_index=False,
        )

        with pd.option_context("max_colwidth", 1000):
            results_metrics.to_latex(
                buf=os.path.join(export_experiments_folder_path, "accuracy-metrics.tex"),
                index=True,
                header=True,
                escape=False,
                float_format="{:.4f}".format,
                encoding="utf-8",
                na_rep="-",
            )
            results_times.to_latex(
                buf=os.path.join(export_experiments_folder_path, "times.tex"),
                index=True,
                header=True,
                escape=False,
                float_format="{:.5f}".format,
                encoding="utf-8",
                na_rep="-",
            )
            results_hyper_parameters.to_latex(
                buf=os.path.join(export_experiments_folder_path, "hyper-parameters.tex"),
                index=True,
                header=True,
                escape=False,
                float_format="{:.4f}".format,
                encoding="utf-8",
                na_rep="-",
            )

        logger.info(
            f"Successfully finished exporting accuracy and beyond-accuracy results to LaTeX"
        )
