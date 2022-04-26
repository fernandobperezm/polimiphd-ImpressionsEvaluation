import itertools
import os
import uuid
from typing import Type, Optional, Sequence, cast

import Recommenders.Recommender_import_list as recommenders
import attrs
import numpy as np
import scipy.sparse as sp
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender, \
    BaseUserSimilarityMatrixRecommender, BaseSimilarityMatrixRecommender
from recsys_framework_extensions.dask import DaskInterface
from recsys_framework_extensions.data.mixins import InteractionsDataSplits
from recsys_framework_extensions.logging import get_logger
from recsys_framework_extensions.plotting import generate_accuracy_and_beyond_metrics_latex

import experiments.commons as commons
from experiments.baselines import load_trained_recommender, TrainedRecommenderType, mock_trained_recommender
from impression_recommenders.user_profile.weighted import UserWeightedUserProfileRecommender, \
    ItemWeightedUserProfileRecommender, BaseWeightedUserProfileRecommender

logger = get_logger(__name__)


####################################################################################################
####################################################################################################
#                                FOLDERS VARIABLES                            #
####################################################################################################
####################################################################################################
BASE_FOLDER = os.path.join(
    commons.RESULTS_EXPERIMENTS_DIR,
    "user_profiles",
    "{benchmark}",
    "{evaluation_strategy}",
    "",
)
ARTICLE_ACCURACY_METRICS_BASELINES_LATEX_DIR = os.path.join(
    BASE_FOLDER,
    "latex",
    "article-accuracy_and_beyond_accuracy",
    "",
)
ACCURACY_METRICS_BASELINES_LATEX_DIR = os.path.join(
    BASE_FOLDER,
    "latex",
    "accuracy_and_beyond_accuracy",
    "",
)
HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR = os.path.join(
    BASE_FOLDER,
    "experiments",
    ""
)

commons.FOLDERS.add(BASE_FOLDER)
commons.FOLDERS.add(ACCURACY_METRICS_BASELINES_LATEX_DIR)
commons.FOLDERS.add(ARTICLE_ACCURACY_METRICS_BASELINES_LATEX_DIR)
commons.FOLDERS.add(HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR)

####################################################################################################
####################################################################################################
#                                REPRODUCIBILITY VARIABLES                            #
####################################################################################################
####################################################################################################
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
#                               Hyper-parameter tuning of Baselines                                #
####################################################################################################
####################################################################################################
def _run_impressions_user_profiles_hyper_parameter_tuning(
    experiment_case_user_profile: commons.ExperimentCase,
    experiment_case_baseline: commons.ExperimentCase,
    experiment_baseline_similarity: Optional[str],
) -> None:
    """TODO: fernando-debugger| complete.
    """

    experiment_user_profiles_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
        experiment_case_user_profile.benchmark
    ]
    experiment_user_profiles_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
        experiment_case_user_profile.recommender
    ]
    experiment_user_profiles_hyper_parameters = commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
        experiment_case_user_profile.hyper_parameter_tuning_parameters
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

    assert experiment_user_profiles_recommender.search_hyper_parameters is not None

    benchmark_reader = commons.get_reader_from_benchmark(
        benchmark_config=experiment_user_profiles_benchmark.config,
        benchmark=experiment_user_profiles_benchmark.benchmark,
    )
    
    dataset = benchmark_reader.dataset

    interactions_data_splits = dataset.get_urm_splits(
        evaluation_strategy=experiment_user_profiles_hyper_parameters.evaluation_strategy,
    )
    impressions_data_splits = dataset.get_uim_splits(
        evaluation_strategy=experiment_user_profiles_hyper_parameters.evaluation_strategy,
    )

    baseline_recommender_trained_train = load_trained_recommender(
        experiment_benchmark=experiment_baseline_benchmark,
        experiment_hyper_parameter_tuning_parameters=experiment_baseline_hyper_parameters,
        experiment_recommender=experiment_baseline_recommender,
        similarity=experiment_baseline_similarity,
        data_splits=interactions_data_splits,
        model_type=TrainedRecommenderType.TRAIN,
        try_folded_recommender=True,
    )

    baseline_recommender_trained_train_validation = load_trained_recommender(
        experiment_benchmark=experiment_baseline_benchmark,
        experiment_hyper_parameter_tuning_parameters=experiment_baseline_hyper_parameters,
        experiment_recommender=experiment_baseline_recommender,
        similarity=experiment_baseline_similarity,
        data_splits=interactions_data_splits,
        model_type=TrainedRecommenderType.TRAIN_VALIDATION,
        try_folded_recommender=True,
    )

    requires_user_similarity = issubclass(experiment_user_profiles_recommender.recommender, UserWeightedUserProfileRecommender)
    requires_item_similarity = issubclass(experiment_user_profiles_recommender.recommender, ItemWeightedUserProfileRecommender)

    recommender_has_user_similarity = (
        isinstance(baseline_recommender_trained_train, BaseUserSimilarityMatrixRecommender)
        and isinstance(baseline_recommender_trained_train_validation, BaseUserSimilarityMatrixRecommender)
    )
    recommender_has_item_similarity = (
        isinstance(baseline_recommender_trained_train, BaseItemSimilarityMatrixRecommender)
        and isinstance(baseline_recommender_trained_train_validation, BaseItemSimilarityMatrixRecommender)
    )

    if requires_user_similarity and not recommender_has_user_similarity:
        # We require a recommender that can be folded. In case we did not receive it, we return
        # to gracefully say that this case finished (as there is nothing to search).
        logger.warning(
            f"Early-returning from {_run_impressions_user_profiles_hyper_parameter_tuning.__name__} as the loaded recommender "
            f"({baseline_recommender_trained_train.RECOMMENDER_NAME} the one requested in the hyper-parameter search) "
            f"cannot load a User-User similarity recommender for the recommender "
            f"{experiment_baseline_recommender.recommender}, i.e., to be an instance of {BaseUserSimilarityMatrixRecommender}. "
            f"\n This is not an issue, as not all recommenders cannot be folded-in. This means that there is nothing "
            f"to search here."
        )
        return

    if requires_item_similarity and not recommender_has_item_similarity:
        # We require a recommender that can be folded. In case we did not receive it, we return
        # to gracefully say that this case finished (as there is nothing to search).
        logger.warning(
            f"Early-returning from {_run_impressions_user_profiles_hyper_parameter_tuning.__name__} as the loaded recommender "
            f"({baseline_recommender_trained_train.RECOMMENDER_NAME} the one requested in the hyper-parameter search) "
            f"cannot load a Item-Item similarity recommender for the recommender "
            f"{experiment_baseline_recommender.recommender}, i.e., to be an instance of {BaseItemSimilarityMatrixRecommender}. "
            f"\n This is not an issue, as not all recommenders cannot be folded-in. This means that there is nothing "
            f"to search here."
        )
        return

    assert baseline_recommender_trained_train.RECOMMENDER_NAME == baseline_recommender_trained_train_validation.RECOMMENDER_NAME

    experiments_folder_path = HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR.format(
        benchmark=experiment_user_profiles_benchmark.benchmark.value,
        evaluation_strategy=experiment_user_profiles_hyper_parameters.evaluation_strategy.value,
    )
    experiment_file_name_root = (
        f"{experiment_user_profiles_recommender.recommender.RECOMMENDER_NAME}"
        f"_{baseline_recommender_trained_train.RECOMMENDER_NAME}"
    )

    import random
    import numpy as np

    random.seed(experiment_user_profiles_hyper_parameters.reproducibility_seed)
    np.random.seed(experiment_user_profiles_hyper_parameters.reproducibility_seed)

    evaluators = commons.get_evaluators(
        data_splits=interactions_data_splits,
        experiment_hyper_parameter_tuning_parameters=experiment_user_profiles_hyper_parameters,
    )

    logger_info = {
        "recommender": experiment_user_profiles_recommender.recommender.RECOMMENDER_NAME,
        "dataset": experiment_user_profiles_benchmark.benchmark.value,
        "urm_test_shape": interactions_data_splits.sp_urm_test.shape,
        "urm_train_shape": interactions_data_splits.sp_urm_train.shape,
        "urm_validation_shape": interactions_data_splits.sp_urm_validation.shape,
        "urm_train_and_validation_shape": interactions_data_splits.sp_urm_train_validation.shape,
        "hyper_parameter_tuning_parameters": experiment_user_profiles_hyper_parameters.__repr__(),
    }

    logger.info(
        f"Hyper-parameter tuning arguments:"
        f"\n\t* {logger_info}"
    )

    recommender_init_validation_args_kwargs = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[],
        CONSTRUCTOR_KEYWORD_ARGS={
            "urm_train": interactions_data_splits.sp_urm_train,
            "uim_train": impressions_data_splits.sp_uim_train,
            "trained_recommender": baseline_recommender_trained_train,
        },
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS={},
    )

    recommender_init_test_args_kwargs = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[],
        CONSTRUCTOR_KEYWORD_ARGS={
            "urm_train": interactions_data_splits.sp_urm_train_validation,
            "uim_train": impressions_data_splits.sp_uim_train_validation,
            "trained_recommender": baseline_recommender_trained_train_validation,
        },
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS={},
    )

    hyper_parameter_search_space = attrs.asdict(
        experiment_user_profiles_recommender.search_hyper_parameters()
    )

    search_bayesian_skopt = SearchBayesianSkopt(
        recommender_class=experiment_user_profiles_recommender.recommender,
        evaluator_validation=evaluators.validation,
        evaluator_test=evaluators.test,
        verbose=True,
    )
    search_bayesian_skopt.search(
        cutoff_to_optimize=experiment_user_profiles_hyper_parameters.cutoff_to_optimize,

        evaluate_on_test=experiment_user_profiles_hyper_parameters.evaluate_on_test,

        hyperparameter_search_space=hyper_parameter_search_space,

        max_total_time=experiment_user_profiles_hyper_parameters.max_total_time,
        metric_to_optimize=experiment_user_profiles_hyper_parameters.metric_to_optimize,

        n_cases=experiment_user_profiles_hyper_parameters.num_cases,
        n_random_starts=experiment_user_profiles_hyper_parameters.num_random_starts,

        output_file_name_root=experiment_file_name_root,
        output_folder_path=experiments_folder_path,

        recommender_input_args=recommender_init_validation_args_kwargs,
        recommender_input_args_last_test=recommender_init_test_args_kwargs,
        resume_from_saved=experiment_user_profiles_hyper_parameters.resume_from_saved,

        save_metadata=experiment_user_profiles_hyper_parameters.save_metadata,
        save_model=experiment_user_profiles_hyper_parameters.save_model,

        terminate_on_memory_error=experiment_user_profiles_hyper_parameters.terminate_on_memory_error,
    )


def run_impressions_user_profiles_experiments(
    dask_interface: DaskInterface,
    user_profiles_experiment_cases_interface: commons.ExperimentCasesInterface,
    baseline_experiment_cases_interface: commons.ExperimentCasesInterface,
) -> None:
    for experiment_case_user_profiles in user_profiles_experiment_cases_interface.experiment_cases:
        for experiment_case_baseline in baseline_experiment_cases_interface.experiment_cases:
            if (
                experiment_case_user_profiles.benchmark != experiment_case_baseline.benchmark
                and experiment_case_user_profiles.hyper_parameter_tuning_parameters != experiment_case_baseline.hyper_parameter_tuning_parameters
            ):
                continue

            user_profiles_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
                experiment_case_user_profiles.benchmark
            ]
            user_profiles_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
                experiment_case_user_profiles.recommender
            ]

            baseline_recommender= commons.MAPPER_AVAILABLE_RECOMMENDERS[
                experiment_case_baseline.recommender
            ]
            baseline_hyper_parameters = commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
                experiment_case_baseline.hyper_parameter_tuning_parameters
            ]

            similarities: Sequence[Optional[commons.T_SIMILARITY_TYPE]] = [None]
            if baseline_recommender.recommender in [
                recommenders.ItemKNNCFRecommender, recommenders.UserKNNCFRecommender
            ]:
                similarities = baseline_hyper_parameters.knn_similarity_types

            for similarity in similarities:
                dask_interface.submit_job(
                    job_key=(
                        f"_run_impressions_heuristics_hyper_parameter_tuning"
                        f"|{user_profiles_benchmark.benchmark.value}"
                        f"|{user_profiles_recommender.recommender.RECOMMENDER_NAME}"
                        f"|{baseline_recommender.recommender.RECOMMENDER_NAME}"
                        f"|{similarity}"
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
                        "similarity": similarity,
                        "benchmark": user_profiles_benchmark.benchmark.value,
                    },
                    method=_run_impressions_user_profiles_hyper_parameter_tuning,
                    method_kwargs={
                        "experiment_case_user_profile": experiment_case_user_profiles,
                        "experiment_case_baseline": experiment_case_baseline,
                        "experiment_baseline_similarity": similarity,
                    }
                )
