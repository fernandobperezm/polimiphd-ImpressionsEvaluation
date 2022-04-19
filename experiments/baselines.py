import os
import uuid
from typing import Type

import Recommenders.Recommender_import_list as recommenders
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.BaseRecommender import BaseRecommender
from recsys_framework_extensions.dask import DaskInterface
from recsys_framework_extensions.evaluation import exclude_from_evaluation
from recsys_framework_extensions.hyper_parameter_search import run_hyper_parameter_search_collaborative
from recsys_framework_extensions.logging import get_logger

import experiments.commons as commons

logger = get_logger(__name__)


####################################################################################################
####################################################################################################
#                                FOLDERS VARIABLES                            #
####################################################################################################
####################################################################################################
BASE_FOLDER = os.path.join(
    commons.RESULTS_EXPERIMENTS_DIR,
    "evaluation",
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
def _run_baselines_hyper_parameter_tuning(
    experiment: commons.Experiment,
    experiment_recommender: commons.ExperimentRecommender,
) -> None:
    """TODO: fernando-debugger| complete.
    """
    benchmark_reader = commons.get_reader_from_benchmark(
        benchmark_config=experiment.benchmark.config,
        benchmark=experiment.benchmark.benchmark,
    )
    
    dataset = benchmark_reader.dataset

    data_splits = dataset.get_urm_splits(
        evaluation_strategy=experiment.hyper_parameter_tuning_parameters.evaluation_strategy
    )

    import random
    import numpy as np

    logger.debug(
        f"Modules - {random=} - {np=}"
    )

    random.seed(experiment.hyper_parameter_tuning_parameters.reproducibility_seed)
    np.random.seed(experiment.hyper_parameter_tuning_parameters.reproducibility_seed)

    experiments_folder_path = HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR.format(
        benchmark=experiment.benchmark.benchmark.value,
        evaluation_strategy=experiment.hyper_parameter_tuning_parameters.evaluation_strategy.value,
    )

    if experiment.hyper_parameter_tuning_parameters.evaluation_ignore_users is None:
        users_to_exclude_validation = None
    else:
        users_to_exclude_validation = exclude_from_evaluation(
            urm_test=data_splits.sp_urm_test,
            frac_to_exclude=experiment.hyper_parameter_tuning_parameters.evaluation_ignore_users,
            type_to_exclude="users",
            seed=experiment.hyper_parameter_tuning_parameters.reproducibility_seed,
        )

    if experiment.hyper_parameter_tuning_parameters.evaluation_ignore_items is None:
        items_to_exclude_validation = None
    else:
        items_to_exclude_validation = exclude_from_evaluation(
            urm_test=data_splits.sp_urm_test,
            frac_to_exclude=experiment.hyper_parameter_tuning_parameters.evaluation_ignore_items,
            type_to_exclude="items",
            seed=experiment.hyper_parameter_tuning_parameters.reproducibility_seed,
        )

    evaluator_validation = EvaluatorHoldout(
        data_splits.sp_urm_validation,
        cutoff_list=experiment.hyper_parameter_tuning_parameters.evaluation_cutoffs,
        exclude_seen=experiment.hyper_parameter_tuning_parameters.evaluation_exclude_seen,
        min_ratings_per_user=experiment.hyper_parameter_tuning_parameters.evaluation_min_ratings_per_user,
        verbose=True,
        ignore_users=users_to_exclude_validation,
        ignore_items=items_to_exclude_validation,
    )
    evaluator_validation_early_stopping = EvaluatorHoldout(
        data_splits.sp_urm_validation,
        # The example uses the hyper-param benchmark_config instead of the evaluation cutoff.
        cutoff_list=[experiment.hyper_parameter_tuning_parameters.cutoff_to_optimize],
        exclude_seen=experiment.hyper_parameter_tuning_parameters.evaluation_exclude_seen,
        min_ratings_per_user=experiment.hyper_parameter_tuning_parameters.evaluation_min_ratings_per_user,
        verbose=True,
        ignore_users=users_to_exclude_validation,
        ignore_items=items_to_exclude_validation,
    )
    evaluator_test = EvaluatorHoldout(
        data_splits.sp_urm_test,
        cutoff_list=experiment.hyper_parameter_tuning_parameters.evaluation_cutoffs,
        exclude_seen=experiment.hyper_parameter_tuning_parameters.evaluation_exclude_seen,
        min_ratings_per_user=experiment.hyper_parameter_tuning_parameters.evaluation_min_ratings_per_user,
        verbose=True,
        ignore_users=None,  # Always consider all users in the test set.
        ignore_items=None,  # Always consider all items in the test set.
    )

    logger_info = {
        "recommender": experiment_recommender.recommender.RECOMMENDER_NAME,
        "dataset": experiment.benchmark.benchmark.value,
        "urm_test_shape": data_splits.sp_urm_test.shape,
        "urm_train_shape": data_splits.sp_urm_train.shape,
        "urm_validation_shape": data_splits.sp_urm_validation.shape,
        "urm_train_and_validation_shape": data_splits.sp_urm_train_validation.shape,
        "hyper_parameter_tuning_parameters": experiment.hyper_parameter_tuning_parameters.__repr__(),
    }

    logger.info(
        f"Hyper-parameter tuning arguments:"
        f"\n\t* {logger_info}"
    )
    run_hyper_parameter_search_collaborative(
        allow_weighting=True,
        allow_bias_URM=False,
        allow_dropout_MF=False,

        cutoff_to_optimize=experiment.hyper_parameter_tuning_parameters.cutoff_to_optimize,

        evaluator_test=evaluator_test,
        evaluator_validation=evaluator_validation,
        evaluator_validation_earlystopping=evaluator_validation_early_stopping,
        evaluate_on_test=experiment.hyper_parameter_tuning_parameters.evaluate_on_test,

        max_total_time=experiment.hyper_parameter_tuning_parameters.max_total_time,
        metric_to_optimize=experiment.hyper_parameter_tuning_parameters.metric_to_optimize,

        n_cases=experiment.hyper_parameter_tuning_parameters.num_cases,
        n_random_starts=experiment.hyper_parameter_tuning_parameters.num_random_starts,

        output_folder_path=experiments_folder_path,

        parallelizeKNN=False,

        recommender_class=experiment_recommender.recommender,
        resume_from_saved=experiment.hyper_parameter_tuning_parameters.resume_from_saved,

        similarity_type_list=experiment.hyper_parameter_tuning_parameters.knn_similarity_types,

        URM_train=data_splits.sp_urm_train,
        URM_train_last_test=data_splits.sp_urm_train_validation,
    )


def run_baselines_experiments(
    dask_interface: DaskInterface,
    experiment_cases_interface: commons.ExperimentCasesInterface,
) -> None:
    for experiment in experiment_cases_interface.experiments:
        for experiment_recommender in experiment.recommenders:

            dask_interface.submit_job(
                job_key=(
                    f"_run_baselines_hyper_parameter_tuning"
                    f"|{experiment.benchmark.benchmark.value}"
                    f"|{experiment_recommender.recommender.RECOMMENDER_NAME}"
                    f"|{uuid.uuid4()}"
                ),
                job_priority=experiment.benchmark.priority * experiment_recommender.priority,
                job_info={
                    "recommender": experiment_recommender.recommender.RECOMMENDER_NAME,
                    "benchmark": experiment.benchmark.benchmark.value,
                },
                method=_run_baselines_hyper_parameter_tuning,
                method_kwargs={
                    "experiment": experiment,
                    "experiment_recommender": experiment_recommender,
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
