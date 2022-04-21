import os
import uuid
from enum import Enum
from typing import Type, Optional, Any, cast

import Recommenders.Recommender_import_list as recommenders
import attrs
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Recommenders.BaseRecommender import BaseRecommender
from recsys_framework_extensions.dask import DaskInterface
from recsys_framework_extensions.data.io import DataIO
from recsys_framework_extensions.data.mixins import InteractionsDataSplits
from recsys_framework_extensions.hyper_parameter_search import run_hyper_parameter_search_collaborative
from recsys_framework_extensions.logging import get_logger

import experiments.commons as commons
from impression_recommenders.user_profile.folding import FoldedMatrixFactorizationRecommender

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
#                               Utility to load an already-tuned recommender                                #
####################################################################################################
####################################################################################################
def load_best_hyper_parameters(
    recommender_class: Type[BaseRecommender],
    benchmark: commons.Benchmarks,
    hyper_parameter_tuning_parameters: commons.HyperParameterTuningParameters,
    similarity: Optional[str],
) -> dict[Any, Any]:
    tuned_recommender_folder_path = HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR.format(
        benchmark=benchmark.value,
        evaluation_strategy=hyper_parameter_tuning_parameters.evaluation_strategy.value,
    )

    if recommender_class in [recommenders.ItemKNNCFRecommender, recommenders.UserKNNCFRecommender]:
        assert similarity is not None and similarity != ""
        tuned_recommender_file_name = (
            f"{recommender_class.RECOMMENDER_NAME}_{similarity}_metadata"
        )

    else:
        tuned_recommender_file_name = (
            f"{recommender_class.RECOMMENDER_NAME}_metadata"
        )

    tuned_recommender_metadata = DataIO.s_load_data(
        folder_path=tuned_recommender_folder_path,
        file_name=tuned_recommender_file_name,
    )

    return tuned_recommender_metadata["hyperparameters_best"]


class TrainedRecommenderType(Enum):
    TRAIN = "TRAIN"
    TRAIN_VALIDATION = "TRAIN_VALIDATION"


def load_trained_recommender(
    experiment: commons.Experiment,
    experiment_recommender: commons.ExperimentRecommender,
    data_splits: InteractionsDataSplits,
    similarity: Optional[str],
    model_type: TrainedRecommenderType,
) -> BaseRecommender:

    if experiment_recommender.recommender in [recommenders.ItemKNNCFRecommender, recommenders.UserKNNCFRecommender]:
        assert similarity is not None
        assert similarity in experiment.hyper_parameter_tuning_parameters.knn_similarity_types

        recommender_name = f"{experiment_recommender.recommender.RECOMMENDER_NAME}_{similarity}"
    else:
        recommender_name = f"{experiment_recommender.recommender.RECOMMENDER_NAME}"

    if TrainedRecommenderType.TRAIN == model_type:
        urm_train = data_splits.sp_urm_train
        file_name_postfix = "best_model.zip"

    elif TrainedRecommenderType.TRAIN_VALIDATION == model_type:
        urm_train = data_splits.sp_urm_train_validation
        file_name_postfix = "best_model_last.zip"

    else:
        raise ValueError(
            f"{load_trained_recommender.__name__} failed because it received an invalid instance of the "
            f"enum {TrainedRecommenderType} (received value {model_type}). Valid values are "
            f"{list(TrainedRecommenderType)}")

    folder_path = HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR.format(
        benchmark=experiment.benchmark.benchmark.value,
        evaluation_strategy=experiment.hyper_parameter_tuning_parameters.evaluation_strategy.value,
    )

    recommender_instance = experiment_recommender.recommender(
        URM_train=urm_train.copy(),
    )
    recommender_instance.load_model(
        folder_path=folder_path,
        file_name=f"{recommender_name}_{file_name_postfix}",
    )
    recommender_instance.RECOMMENDER_NAME = recommender_name

    return recommender_instance


def load_trained_folded_recommender(
    experiment: commons.Experiment,
    experiment_recommender: commons.ExperimentRecommender,
    data_splits: InteractionsDataSplits,
    similarity: Optional[str],
    model_type: TrainedRecommenderType,
) -> FoldedMatrixFactorizationRecommender:
    if experiment_recommender.recommender in [recommenders.ItemKNNCFRecommender, recommenders.UserKNNCFRecommender]:
        raise ValueError(f"Cannot load a folded recommender trained on MatrixSimilarity.")

    if TrainedRecommenderType.TRAIN == model_type:
        urm_train = data_splits.sp_urm_train
        file_name_postfix = "best_model.zip"

    elif TrainedRecommenderType.TRAIN_VALIDATION == model_type:
        urm_train = data_splits.sp_urm_train_validation
        file_name_postfix = "best_model_last.zip"

    else:
        raise ValueError(
            f"{load_trained_folded_recommender.__name__} failed because it received an invalid instance of the "
            f"enum {TrainedRecommenderType} (received value {model_type}). Valid values are "
            f"{list(TrainedRecommenderType)}")

    folder_path = HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR.format(
        benchmark=experiment.benchmark.benchmark.value,
        evaluation_strategy=experiment.hyper_parameter_tuning_parameters.evaluation_strategy.value,
    )

    trained_recommender = load_trained_recommender(
        experiment=experiment,
        experiment_recommender=experiment_recommender,
        data_splits=data_splits,
        similarity=similarity,
        model_type=model_type,
    )
    trained_recommender = cast(BaseMatrixFactorizationRecommender, trained_recommender)

    file_name_prefix = FoldedMatrixFactorizationRecommender.RECOMMENDER_NAME.replace("Recommender", "")
    file_name_prefix = f"{file_name_prefix}_{experiment_recommender.recommender.RECOMMENDER_NAME}"
    recommender_instance = FoldedMatrixFactorizationRecommender(
        urm_train=urm_train.copy(),
        trained_recommender=trained_recommender,
    )
    recommender_instance.load_model(
        folder_path=folder_path,
        file_name=f"{file_name_prefix}_{file_name_postfix}",
    )

    return recommender_instance


####################################################################################################
####################################################################################################
#                               Hyper-parameter tuning of Baselines                                #
####################################################################################################
####################################################################################################
def _run_baselines_folded_hyper_parameter_tuning(
    experiment: commons.Experiment,
    experiment_recommender: commons.ExperimentRecommender,
    recommender_folded: commons.ExperimentRecommender,
) -> None:
    """TODO: fernando-debugger| complete.
    """
    if not issubclass(
        experiment_recommender.recommender, BaseMatrixFactorizationRecommender
    ):
        return

    benchmark_reader = commons.get_reader_from_benchmark(
        benchmark_config=experiment.benchmark.config,
        benchmark=experiment.benchmark.benchmark,
    )

    dataset = benchmark_reader.dataset

    interactions_data_splits = dataset.get_urm_splits(
        evaluation_strategy=experiment.hyper_parameter_tuning_parameters.evaluation_strategy,
    )

    baseline_recommender_trained_train = load_trained_recommender(
        experiment=experiment,
        experiment_recommender=experiment_recommender,
        data_splits=interactions_data_splits,
        similarity=None,
        model_type=TrainedRecommenderType.TRAIN,
    )

    baseline_recommender_trained_train_validation = load_trained_recommender(
        experiment=experiment,
        experiment_recommender=experiment_recommender,
        data_splits=interactions_data_splits,
        similarity=None,
        model_type=TrainedRecommenderType.TRAIN_VALIDATION,
    )

    assert baseline_recommender_trained_train.RECOMMENDER_NAME == baseline_recommender_trained_train_validation.RECOMMENDER_NAME

    experiments_folder_path = HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR.format(
        benchmark=experiment.benchmark.benchmark.value,
        evaluation_strategy=experiment.hyper_parameter_tuning_parameters.evaluation_strategy.value,
    )
    experiment_file_name_root = (
        f"FoldedMatrixFactorization_{baseline_recommender_trained_train.RECOMMENDER_NAME}"
    )

    import random
    import numpy as np

    random.seed(experiment.hyper_parameter_tuning_parameters.reproducibility_seed)
    np.random.seed(experiment.hyper_parameter_tuning_parameters.reproducibility_seed)

    evaluators = commons.get_evaluators(
        experiment=experiment,
        data_splits=interactions_data_splits,
    )

    logger_info = {
        "recommender": experiment_recommender.recommender.RECOMMENDER_NAME,
        "dataset": experiment.benchmark.benchmark.value,
        "urm_test_shape": interactions_data_splits.sp_urm_test.shape,
        "urm_train_shape": interactions_data_splits.sp_urm_train.shape,
        "urm_validation_shape": interactions_data_splits.sp_urm_validation.shape,
        "urm_train_and_validation_shape": interactions_data_splits.sp_urm_train_validation.shape,
        "hyper_parameter_tuning_parameters": experiment.hyper_parameter_tuning_parameters,
    }

    logger.info(
        f"Hyper-parameter tuning arguments:"
        f"\n\t* {logger_info}"
    )

    recommender_init_validation_args_kwargs = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[],
        CONSTRUCTOR_KEYWORD_ARGS={
            "urm_train": interactions_data_splits.sp_urm_train,
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
            "trained_recommender": baseline_recommender_trained_train_validation,
        },
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS={},
    )

    hyper_parameter_search_space = attrs.asdict(
        recommender_folded.search_hyper_parameters()
    )

    search_bayesian_skopt = SearchBayesianSkopt(
        recommender_class=recommender_folded.recommender,
        evaluator_validation=evaluators.validation,
        evaluator_test=evaluators.test,
        verbose=True,
    )
    search_bayesian_skopt.search(
        cutoff_to_optimize=experiment.hyper_parameter_tuning_parameters.cutoff_to_optimize,

        evaluate_on_test=experiment.hyper_parameter_tuning_parameters.evaluate_on_test,

        hyperparameter_search_space=hyper_parameter_search_space,

        max_total_time=experiment.hyper_parameter_tuning_parameters.max_total_time,
        metric_to_optimize=experiment.hyper_parameter_tuning_parameters.metric_to_optimize,

        n_cases=experiment.hyper_parameter_tuning_parameters.num_cases,
        n_random_starts=experiment.hyper_parameter_tuning_parameters.num_random_starts,

        output_file_name_root=experiment_file_name_root,
        output_folder_path=experiments_folder_path,

        recommender_input_args=recommender_init_validation_args_kwargs,
        recommender_input_args_last_test=recommender_init_test_args_kwargs,
        resume_from_saved=experiment.hyper_parameter_tuning_parameters.resume_from_saved,

        save_metadata=experiment.hyper_parameter_tuning_parameters.save_metadata,
        save_model=experiment.hyper_parameter_tuning_parameters.save_model,

        terminate_on_memory_error=experiment.hyper_parameter_tuning_parameters.terminate_on_memory_error,
    )


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
    evaluators = commons.get_evaluators(
        experiment=experiment,
        data_splits=data_splits,
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

        evaluator_test=evaluators.test,
        evaluator_validation=evaluators.validation,
        evaluator_validation_earlystopping=evaluators.validation_early_stopping,
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

        save_model=experiment.hyper_parameter_tuning_parameters.save_model,
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


def run_baselines_folded(
    dask_interface: DaskInterface,
    recommender_folded: commons.ExperimentRecommender,
    experiment_cases_interface: commons.ExperimentCasesInterface,
) -> None:
    for experiment in experiment_cases_interface.experiments:
        for experiment_recommender in experiment.recommenders:
            dask_interface.submit_job(
                job_key=(
                    f"_run_baselines_folded_hyper_parameter_tuning"
                    f"|{experiment.benchmark.benchmark.value}"
                    f"|{experiment_recommender.recommender.RECOMMENDER_NAME}"
                    f"|{recommender_folded.recommender.RECOMMENDER_NAME}"
                    f"|{uuid.uuid4()}"
                ),
                job_priority=experiment.benchmark.priority * experiment_recommender.priority,
                job_info={
                    "recommender": experiment_recommender.recommender.RECOMMENDER_NAME,
                    "recommender_folded": recommender_folded.recommender.RECOMMENDER_NAME,
                    "benchmark": experiment.benchmark.benchmark.value,
                },
                method=_run_baselines_folded_hyper_parameter_tuning,
                method_kwargs={
                    "experiment": experiment,
                    "experiment_recommender": experiment_recommender,
                    "recommender_folded": recommender_folded,
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
