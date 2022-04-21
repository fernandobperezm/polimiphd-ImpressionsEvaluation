import os
import uuid
from typing import Type, Optional, Sequence

import Recommenders.Recommender_import_list as recommenders
import attrs
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender, \
    BaseSimilarityMatrixRecommender, BaseUserSimilarityMatrixRecommender
from recsys_framework_extensions.dask import DaskInterface
from recsys_framework_extensions.data.mixins import InteractionsDataSplits
from recsys_framework_extensions.logging import get_logger

import experiments.commons as commons
# from experiments.baselines import load_trained_recommender
from experiments.baselines import load_trained_recommender, TrainedRecommenderType, load_trained_folded_recommender
from impression_recommenders.user_profile.folding import FoldedMatrixFactorizationRecommender
from impression_recommenders.user_profile.weighted import UserWeightedUserProfileRecommender, \
    ItemWeightedUserProfileRecommender

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
    "article",
    "accuracy_and_beyond_accuracy",
)
ACCURACY_METRICS_BASELINES_LATEX_DIR = os.path.join(
    BASE_FOLDER,
    "latex",
    "accuracy_and_beyond_accuracy",
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
ARTICLE_KNN_SIMILARITY_LIST = [
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
#                               Get Folded-In Recommender                                #
####################################################################################################
####################################################################################################
def _load_similarity_recommender(
    experiment: commons.Experiment,
    experiment_recommender: commons.ExperimentRecommender,
    experiment_user_profiles_recommender: commons.ExperimentRecommender,
    data_splits: InteractionsDataSplits,
    similarity: Optional[str],
    model_type: TrainedRecommenderType,
) -> BaseItemSimilarityMatrixRecommender:

    if issubclass(experiment_user_profiles_recommender.recommender, UserWeightedUserProfileRecommender):
        if issubclass(experiment_recommender.recommender, BaseUserSimilarityMatrixRecommender):
            func = load_trained_recommender
        else:
            raise ValueError(
                f"Not possible to convert recommender {experiment_recommender.recommender} to a subclass of "
                f"{BaseUserSimilarityMatrixRecommender}. Valid recommender are those that are subclasses of "
                f"{BaseUserSimilarityMatrixRecommender}."
            )
    elif issubclass(experiment_user_profiles_recommender.recommender, ItemWeightedUserProfileRecommender):
        if issubclass(experiment_recommender.recommender, BaseMatrixFactorizationRecommender):
            func = load_trained_folded_recommender
        elif issubclass(experiment_recommender.recommender, BaseItemSimilarityMatrixRecommender):
            func = load_trained_recommender
        else:
            raise ValueError(
                f"Not possible to convert recommender {experiment_recommender.recommender} to a subclass of "
                f"{BaseItemSimilarityMatrixRecommender}. Valid recommender are those that are subclasses of "
                f"{BaseMatrixFactorizationRecommender} as they can be folded-in or subclasses of "
                f"{BaseItemSimilarityMatrixRecommender}."
            )
    else:
        raise ValueError(
            f"The recommender class {experiment_user_profiles_recommender.recommender} is not valid in this "
            f"function. Valid recommender classes are {UserWeightedUserProfileRecommender} & "
            f"{ItemWeightedUserProfileRecommender}."
        )

    return func(
        experiment=experiment,
        experiment_recommender=experiment_recommender,
        data_splits=data_splits,
        similarity=similarity,
        model_type=model_type,
    )


####################################################################################################
####################################################################################################
#                               Hyper-parameter tuning of Baselines                                #
####################################################################################################
####################################################################################################
def _run_impressions_user_profiles_hyper_parameter_tuning(
    experiment_user_profiles: commons.Experiment,
    experiment_user_profiles_recommender: commons.ExperimentRecommender,
    experiment_baseline: commons.Experiment,
    experiment_baseline_recommender: commons.ExperimentRecommender,
    experiment_baseline_similarity: Optional[str],
) -> None:
    """TODO: fernando-debugger| complete.
    """
    assert experiment_user_profiles_recommender.search_hyper_parameters is not None

    benchmark_reader = commons.get_reader_from_benchmark(
        benchmark_config=experiment_user_profiles.benchmark.config,
        benchmark=experiment_user_profiles.benchmark.benchmark,
    )
    
    dataset = benchmark_reader.dataset

    interactions_data_splits = dataset.get_urm_splits(
        evaluation_strategy=experiment_user_profiles.hyper_parameter_tuning_parameters.evaluation_strategy,
    )
    impressions_data_splits = dataset.get_uim_splits(
        evaluation_strategy=experiment_user_profiles.hyper_parameter_tuning_parameters.evaluation_strategy,
    )

    baseline_recommender_trained_train = _load_similarity_recommender(
        experiment=experiment_baseline,
        experiment_recommender=experiment_baseline_recommender,
        experiment_user_profiles_recommender=experiment_user_profiles_recommender,
        data_splits=interactions_data_splits,
        similarity=experiment_baseline_similarity,
        model_type=TrainedRecommenderType.TRAIN,
    )

    baseline_recommender_trained_train_validation = _load_similarity_recommender(
        experiment=experiment_baseline,
        experiment_recommender=experiment_baseline_recommender,
        experiment_user_profiles_recommender=experiment_user_profiles_recommender,
        data_splits=interactions_data_splits,
        similarity=experiment_baseline_similarity,
        model_type=TrainedRecommenderType.TRAIN_VALIDATION,
    )

    assert baseline_recommender_trained_train.RECOMMENDER_NAME == baseline_recommender_trained_train_validation.RECOMMENDER_NAME

    experiments_folder_path = HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR.format(
        benchmark=experiment_user_profiles.benchmark.benchmark.value,
        evaluation_strategy=experiment_user_profiles.hyper_parameter_tuning_parameters.evaluation_strategy.value,
    )
    experiment_file_name_root = (
        f"{experiment_user_profiles_recommender.recommender.RECOMMENDER_NAME}"
        f"_{baseline_recommender_trained_train.RECOMMENDER_NAME}"
    )

    import random
    import numpy as np

    random.seed(experiment_user_profiles.hyper_parameter_tuning_parameters.reproducibility_seed)
    np.random.seed(experiment_user_profiles.hyper_parameter_tuning_parameters.reproducibility_seed)

    evaluators = commons.get_evaluators(
        experiment=experiment_user_profiles,
        data_splits=interactions_data_splits,
    )

    logger_info = {
        "recommender": experiment_user_profiles_recommender.recommender.RECOMMENDER_NAME,
        "dataset": experiment_user_profiles.benchmark.benchmark.value,
        "urm_test_shape": interactions_data_splits.sp_urm_test.shape,
        "urm_train_shape": interactions_data_splits.sp_urm_train.shape,
        "urm_validation_shape": interactions_data_splits.sp_urm_validation.shape,
        "urm_train_and_validation_shape": interactions_data_splits.sp_urm_train_validation.shape,
        "hyper_parameter_tuning_parameters": experiment_user_profiles.hyper_parameter_tuning_parameters.__repr__(),
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
        cutoff_to_optimize=experiment_user_profiles.hyper_parameter_tuning_parameters.cutoff_to_optimize,

        evaluate_on_test=experiment_user_profiles.hyper_parameter_tuning_parameters.evaluate_on_test,

        hyperparameter_search_space=hyper_parameter_search_space,

        max_total_time=experiment_user_profiles.hyper_parameter_tuning_parameters.max_total_time,
        metric_to_optimize=experiment_user_profiles.hyper_parameter_tuning_parameters.metric_to_optimize,

        n_cases=experiment_user_profiles.hyper_parameter_tuning_parameters.num_cases,
        n_random_starts=experiment_user_profiles.hyper_parameter_tuning_parameters.num_random_starts,

        output_file_name_root=experiment_file_name_root,
        output_folder_path=experiments_folder_path,

        recommender_input_args=recommender_init_validation_args_kwargs,
        recommender_input_args_last_test=recommender_init_test_args_kwargs,
        resume_from_saved=experiment_user_profiles.hyper_parameter_tuning_parameters.resume_from_saved,

        save_metadata=experiment_user_profiles.hyper_parameter_tuning_parameters.save_metadata,
        save_model=experiment_user_profiles.hyper_parameter_tuning_parameters.save_model,

        terminate_on_memory_error=experiment_user_profiles.hyper_parameter_tuning_parameters.terminate_on_memory_error,
    )


def run_impressions_user_profiles_experiments(
    dask_interface: DaskInterface,
    user_profiles_experiment_cases_interface: commons.ExperimentCasesInterface,
    baseline_experiment_cases_interface: commons.ExperimentCasesInterface,
) -> None:
    for experiment_user_profiles in user_profiles_experiment_cases_interface.experiments:
        for experiment_user_profiles_recommender in experiment_user_profiles.recommenders:
            for experiment_baseline in baseline_experiment_cases_interface.experiments:
                if (
                    experiment_user_profiles.benchmark != experiment_baseline.benchmark
                    and experiment_user_profiles.hyper_parameter_tuning_parameters.evaluation_strategy != experiment_baseline.hyper_parameter_tuning_parameters.evaluation_strategy
                ):
                    continue

                for experiment_baseline_recommender in experiment_baseline.recommenders:
                    similarities: Sequence[Optional[commons.T_SIMILARITY_TYPE]] = [None]
                    if experiment_baseline_recommender.recommender in [recommenders.ItemKNNCFRecommender,
                                                                       recommenders.UserKNNCFRecommender]:
                        similarities = experiment_baseline.hyper_parameter_tuning_parameters.knn_similarity_types

                    for similarity in similarities:
                        dask_interface.submit_job(
                            job_key=(
                                f"_run_impressions_heuristics_hyper_parameter_tuning"
                                f"|{experiment_user_profiles.benchmark.benchmark.value}"
                                f"|{experiment_user_profiles_recommender.recommender.RECOMMENDER_NAME}"
                                f"|{experiment_baseline_recommender.recommender.RECOMMENDER_NAME}"
                                f"|{similarity}"
                                f"|{uuid.uuid4()}"
                            ),
                            job_priority=experiment_user_profiles.benchmark.priority * experiment_user_profiles_recommender.priority,
                            job_info={
                                "recommender": experiment_user_profiles_recommender.recommender.RECOMMENDER_NAME,
                                "baseline": experiment_baseline_recommender.recommender.RECOMMENDER_NAME,
                                "similarity": similarity,
                                "benchmark": experiment_user_profiles.benchmark.benchmark.value,
                            },
                            method=_run_impressions_user_profiles_hyper_parameter_tuning,
                            method_kwargs={
                                "experiment_user_profiles": experiment_user_profiles,
                                "experiment_user_profiles_recommender": experiment_user_profiles_recommender,
                                "experiment_baseline": experiment_baseline,
                                "experiment_baseline_recommender": experiment_baseline_recommender,
                                "experiment_baseline_similarity": similarity,
                            }
                        )

####################################################################################################
####################################################################################################
#             Reproducibility study: Results exporting          #
####################################################################################################
####################################################################################################
# def _print_hyper_parameter_tuning_accuracy_and_beyond_accuracy_metrics(
#     urm: sp.csr_matrix,
#     benchmark: CFGANBenchmarks,
#     num_test_users: int,
#     accuracy_metrics_list: list[str],
#     beyond_accuracy_metrics_list: list[str],
#     all_metrics_list: list[str],
#     cutoffs_list: list[int],
#     base_algorithm_list: list[Type[BaseRecommender]],
#     knn_similarity_list: list[str],
#     export_experiments_folder_path: str,
# ) -> None:
#
#     experiments_folder_path = HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR.format(
#         benchmark=benchmark.value
#     )
#
#     other_algorithm_list: list[Optional[CFGANRecommenderEarlyStopping]] = []
#     for cfgan_mode, cfgan_mask_type in cfgan_hyper_parameter_search_settings():
#         recommender = CFGANRecommenderEarlyStopping(
#             urm_train=urm,
#             num_training_item_weights_to_save=0
#         )
#
#         recommender.RECOMMENDER_NAME = recommender.get_recommender_name(
#             cfgan_mode=cfgan_mode,
#             cfgan_mask_type=cfgan_mask_type
#         )
#
#         other_algorithm_list.append(recommender)
#     other_algorithm_list.append(None)
#
#     generate_accuracy_and_beyond_metrics_latex(
#         experiments_folder_path=experiments_folder_path,
#         export_experiments_folder_path=export_experiments_folder_path,
#         num_test_users=num_test_users,
#         base_algorithm_list=base_algorithm_list,
#         knn_similarity_list=knn_similarity_list,
#         other_algorithm_list=other_algorithm_list,
#         accuracy_metrics_list=accuracy_metrics_list,
#         beyond_accuracy_metrics_list=beyond_accuracy_metrics_list,
#         all_metrics_list=all_metrics_list,
#         cutoffs_list=cutoffs_list,
#         icm_names=None
#     )
#
#
# def print_reproducibility_results(
#     experiments_interface: commons.ExperimentCasesInterface,
# ) -> None:
#     for dataset in experiments_interface.datasets:
#         urm = dataset.urm_train + dataset.urm_validation
#
#         num_test_users: int = np.sum(np.ediff1d(dataset.urm_test.indptr) >= 1)
#
#         # Print all baselines, cfgan, g-cfgan, similarities, and accuracy and beyond accuracy metrics
#         export_experiments_folder_path = ACCURACY_METRICS_BASELINES_LATEX_DIR.format(
#             benchmark=dataset.benchmark.value,
#         )
#         _print_hyper_parameter_tuning_accuracy_and_beyond_accuracy_metrics(
#             urm=urm,
#             benchmark=dataset.benchmark,
#             num_test_users=num_test_users,
#             accuracy_metrics_list=ACCURACY_METRICS_LIST,
#             beyond_accuracy_metrics_list=BEYOND_ACCURACY_METRICS_LIST,
#             all_metrics_list=ALL_METRICS_LIST,
#             cutoffs_list=RESULT_EXPORT_CUTOFFS,
#             base_algorithm_list=ARTICLE_BASELINES,
#             knn_similarity_list=KNN_SIMILARITY_LIST,
#             export_experiments_folder_path=export_experiments_folder_path
#         )
#
#         export_experiments_folder_path = ARTICLE_ACCURACY_METRICS_BASELINES_LATEX_DIR.format(
#             benchmark=dataset.benchmark.value,
#         )
#         # Print article baselines, cfgan, similarities, and accuracy and beyond-accuracy metrics.
#         _print_hyper_parameter_tuning_accuracy_and_beyond_accuracy_metrics(
#             urm=urm,
#             benchmark=dataset.benchmark,
#             num_test_users=num_test_users,
#             accuracy_metrics_list=ARTICLE_ACCURACY_METRICS_LIST,
#             beyond_accuracy_metrics_list=ARTICLE_BEYOND_ACCURACY_METRICS_LIST,
#             all_metrics_list=ARTICLE_ALL_METRICS_LIST,
#             cutoffs_list=ARTICLE_CUTOFF,
#             base_algorithm_list=ARTICLE_BASELINES,
#             knn_similarity_list=ARTICLE_KNN_SIMILARITY_LIST,
#             export_experiments_folder_path=export_experiments_folder_path
#         )
#
#         logger.info(
#             f"Successfully finished exporting accuracy and beyond-accuracy results to LaTeX"
#         )