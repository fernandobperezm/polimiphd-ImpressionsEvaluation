import os
import uuid
from typing import Type, Optional, Sequence

import Recommenders.Recommender_import_list as recommenders
import attrs
from Evaluation.Evaluator import EvaluatorHoldout
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.SearchSingleCase import SearchSingleCase
from Recommenders.BaseRecommender import BaseRecommender
from recsys_framework_extensions.dask import DaskInterface
from recsys_framework_extensions.evaluation import exclude_from_evaluation
from recsys_framework_extensions.hyper_parameter_search import run_hyper_parameter_search_collaborative
from recsys_framework_extensions.logging import get_logger

import experiments.commons as commons
# from experiments.baselines import load_trained_recommender
from experiments.baselines import load_trained_recommender, TrainedRecommenderType

logger = get_logger(__name__)


####################################################################################################
####################################################################################################
#                                FOLDERS VARIABLES                            #
####################################################################################################
####################################################################################################
BASE_FOLDER = os.path.join(
    commons.RESULTS_EXPERIMENTS_DIR,
    "re_ranking",
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
#                               Hyper-parameter tuning of Baselines                                #
####################################################################################################
####################################################################################################
def _run_impressions_re_ranking_hyper_parameter_tuning(
    experiment_re_ranking: commons.Experiment,
    experiment_re_ranking_recommender: commons.ExperimentRecommender,
    experiment_baseline: commons.Experiment,
    experiment_baseline_recommender: commons.ExperimentRecommender,
    experiment_baseline_similarity: Optional[str],
) -> None:
    """TODO: fernando-debugger| complete.
    """
    assert experiment_re_ranking_recommender.search_hyper_parameters is not None

    benchmark_reader = commons.get_reader_from_benchmark(
        benchmark_config=experiment_re_ranking.benchmark.config,
        benchmark=experiment_re_ranking.benchmark.benchmark,
    )
    
    dataset = benchmark_reader.dataset

    interactions_data_splits = dataset.get_urm_splits(
        evaluation_strategy=experiment_re_ranking.hyper_parameter_tuning_parameters.evaluation_strategy
    )

    impressions_feature_frequency_train = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking.benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking.hyper_parameter_tuning_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_FREQUENCY,
            impressions_feature_column=commons.ImpressionsFeatureColumnsFrequency.FREQUENCY,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
        )
    )
    impressions_feature_frequency_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking.benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking.hyper_parameter_tuning_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_FREQUENCY,
            impressions_feature_column=commons.ImpressionsFeatureColumnsFrequency.FREQUENCY,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    impressions_feature_position_train = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking.benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking.hyper_parameter_tuning_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_POSITION,
            impressions_feature_column=commons.ImpressionsFeatureColumnsPosition.POSITION,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
        )
    )
    impressions_feature_position_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking.benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking.hyper_parameter_tuning_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_POSITION,
            impressions_feature_column=commons.ImpressionsFeatureColumnsPosition.POSITION,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    impressions_feature_timestamp_train = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking.benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking.hyper_parameter_tuning_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_TIMESTAMP,
            impressions_feature_column=commons.ImpressionsFeatureColumnsTimestamp.TIMESTAMP,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
        )
    )
    impressions_feature_timestamp_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking.benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking.hyper_parameter_tuning_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_TIMESTAMP,
            impressions_feature_column=commons.ImpressionsFeatureColumnsTimestamp.TIMESTAMP,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    baseline_recommender_trained_train = load_trained_recommender(
        experiment=experiment_baseline,
        experiment_recommender=experiment_baseline_recommender,
        data_splits=interactions_data_splits,
        similarity=experiment_baseline_similarity,
        model_type=TrainedRecommenderType.TRAIN,
    )

    baseline_recommender_trained_train_validation = load_trained_recommender(
        experiment=experiment_baseline,
        experiment_recommender=experiment_baseline_recommender,
        data_splits=interactions_data_splits,
        similarity=experiment_baseline_similarity,
        model_type=TrainedRecommenderType.TRAIN_VALIDATION,
    )

    assert baseline_recommender_trained_train.RECOMMENDER_NAME == baseline_recommender_trained_train_validation.RECOMMENDER_NAME

    experiments_folder_path = HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR.format(
        benchmark=experiment_re_ranking.benchmark.benchmark.value,
        evaluation_strategy=experiment_re_ranking.hyper_parameter_tuning_parameters.evaluation_strategy.value,
    )
    experiment_file_name_root = (
        f"{experiment_re_ranking_recommender.recommender.RECOMMENDER_NAME}"
        f"_{baseline_recommender_trained_train.RECOMMENDER_NAME}"
    )

    import random
    import numpy as np

    random.seed(experiment_re_ranking.hyper_parameter_tuning_parameters.reproducibility_seed)
    np.random.seed(experiment_re_ranking.hyper_parameter_tuning_parameters.reproducibility_seed)

    evaluators = commons.get_evaluators(
        experiment=experiment_re_ranking,
        data_splits=interactions_data_splits,
    )

    logger_info = {
        "recommender": experiment_re_ranking_recommender.recommender.RECOMMENDER_NAME,
        "dataset": experiment_re_ranking.benchmark.benchmark.value,
        "urm_test_shape": interactions_data_splits.sp_urm_test.shape,
        "urm_train_shape": interactions_data_splits.sp_urm_train.shape,
        "urm_validation_shape": interactions_data_splits.sp_urm_validation.shape,
        "urm_train_and_validation_shape": interactions_data_splits.sp_urm_train_validation.shape,
        "hyper_parameter_tuning_parameters": experiment_re_ranking.hyper_parameter_tuning_parameters.__repr__(),
    }

    logger.info(
        f"Hyper-parameter tuning arguments:"
        f"\n\t* {logger_info}"
    )

    recommender_init_validation_args_kwargs = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[],
        CONSTRUCTOR_KEYWORD_ARGS={
            "urm_train": interactions_data_splits.sp_urm_train,
            "uim_frequency": impressions_feature_frequency_train,
            "uim_position": impressions_feature_position_train,
            "uim_timestamp": impressions_feature_timestamp_train,
            "seed": experiment_re_ranking.hyper_parameter_tuning_parameters.reproducibility_seed,
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
            "uim_frequency": impressions_feature_frequency_train_validation,
            "uim_position": impressions_feature_position_train_validation,
            "uim_timestamp": impressions_feature_timestamp_train_validation,
            "seed": experiment_re_ranking.hyper_parameter_tuning_parameters.reproducibility_seed,
            "trained_recommender": baseline_recommender_trained_train_validation,
        },
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS={},
    )

    hyper_parameter_search_space = attrs.asdict(
        experiment_re_ranking_recommender.search_hyper_parameters()
    )

    search_bayesian_skopt = SearchBayesianSkopt(
        recommender_class=experiment_re_ranking_recommender.recommender,
        evaluator_validation=evaluators.validation,
        evaluator_test=evaluators.test,
        verbose=True,
    )
    search_bayesian_skopt.search(
        cutoff_to_optimize=experiment_re_ranking.hyper_parameter_tuning_parameters.cutoff_to_optimize,

        evaluate_on_test=experiment_re_ranking.hyper_parameter_tuning_parameters.evaluate_on_test,

        hyperparameter_search_space=hyper_parameter_search_space,

        max_total_time=experiment_re_ranking.hyper_parameter_tuning_parameters.max_total_time,
        metric_to_optimize=experiment_re_ranking.hyper_parameter_tuning_parameters.metric_to_optimize,

        n_cases=experiment_re_ranking.hyper_parameter_tuning_parameters.num_cases,
        n_random_starts=experiment_re_ranking.hyper_parameter_tuning_parameters.num_random_starts,

        output_file_name_root=experiment_file_name_root,
        output_folder_path=experiments_folder_path,

        recommender_input_args=recommender_init_validation_args_kwargs,
        recommender_input_args_last_test=recommender_init_test_args_kwargs,
        resume_from_saved=experiment_re_ranking.hyper_parameter_tuning_parameters.resume_from_saved,

        save_metadata=experiment_re_ranking.hyper_parameter_tuning_parameters.save_metadata,
        save_model=experiment_re_ranking.hyper_parameter_tuning_parameters.save_model,

        terminate_on_memory_error=experiment_re_ranking.hyper_parameter_tuning_parameters.terminate_on_memory_error,
    )


def run_impressions_re_ranking_experiments(
    dask_interface: DaskInterface,
    re_ranking_experiment_cases_interface: commons.ExperimentCasesInterface,
    baseline_experiment_cases_interface: commons.ExperimentCasesInterface,
) -> None:
    for experiment_reranking in re_ranking_experiment_cases_interface.experiments:
        for experiment_re_ranking_recommender in experiment_reranking.recommenders:
            for experiment_baseline in baseline_experiment_cases_interface.experiments:
                if (
                    experiment_reranking.benchmark != experiment_baseline.benchmark
                    and experiment_reranking.hyper_parameter_tuning_parameters.evaluation_strategy != experiment_baseline.hyper_parameter_tuning_parameters.evaluation_strategy
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
                                f"|{experiment_reranking.benchmark.benchmark.value}"
                                f"|{experiment_re_ranking_recommender.recommender.RECOMMENDER_NAME}"
                                f"|{experiment_baseline_recommender.recommender.RECOMMENDER_NAME}"
                                f"|{similarity}"
                                f"|{uuid.uuid4()}"
                            ),
                            job_priority=experiment_reranking.benchmark.priority * experiment_re_ranking_recommender.priority,
                            job_info={
                                "recommender": experiment_re_ranking_recommender.recommender.RECOMMENDER_NAME,
                                "baseline": experiment_baseline_recommender.recommender.RECOMMENDER_NAME,
                                "similarity": similarity,
                                "benchmark": experiment_reranking.benchmark.benchmark.value,
                            },
                            method=_run_impressions_re_ranking_hyper_parameter_tuning,
                            method_kwargs={
                                "experiment_re_ranking": experiment_reranking,
                                "experiment_re_ranking_recommender": experiment_re_ranking_recommender,
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
