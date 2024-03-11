import os
import uuid
from enum import Enum
from typing import Type, Optional, Any, cast, Sequence

import Recommenders.Recommender_import_list as recommenders
import attrs
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from Recommenders.BaseMatrixFactorizationRecommender import (
    BaseMatrixFactorizationRecommender,
)
from Recommenders.BaseRecommender import BaseRecommender
from recsys_framework_extensions.dask import DaskInterface
from recsys_framework_extensions.data.io import DataIO
from recsys_framework_extensions.data.mixins import InteractionsDataSplits
from recsys_framework_extensions.hyper_parameter_search import (
    run_hyper_parameter_search_collaborative,
)
import logging

import impressions_evaluation.experiments.commons as commons
from impressions_evaluation import configure_logger
from impressions_evaluation.impression_recommenders.user_profile.folding import (
    FoldedMatrixFactorizationRecommender,
)

logger = logging.getLogger(__name__)


####################################################################################################
####################################################################################################
#                                FOLDERS VARIABLES                            #
####################################################################################################
####################################################################################################
DIR_TRAINED_MODELS_BASELINES = os.path.join(
    commons.DIR_TRAINED_MODELS,
    "{benchmark}",
    "{evaluation_strategy}",
    # "baselines",
    "",
)

commons.FOLDERS.add(DIR_TRAINED_MODELS_BASELINES)


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
    """
    Loads the dictionary of best hyper-parameters for a given recommender. Currently, not used and untested.
    """

    tuned_recommender_folder_path = DIR_TRAINED_MODELS_BASELINES.format(
        benchmark=benchmark.value,
        evaluation_strategy=hyper_parameter_tuning_parameters.evaluation_strategy.value,
    )

    if recommender_class in [
        recommenders.ItemKNNCFRecommender,
        recommenders.UserKNNCFRecommender,
    ]:
        assert similarity is not None and similarity != ""
        tuned_recommender_file_name = (
            f"{recommender_class.RECOMMENDER_NAME}_{similarity}_metadata"
        )

    else:
        tuned_recommender_file_name = f"{recommender_class.RECOMMENDER_NAME}_metadata"

    tuned_recommender_metadata = DataIO.s_load_data(
        folder_path=tuned_recommender_folder_path,
        file_name=tuned_recommender_file_name,
    )

    return tuned_recommender_metadata["hyperparameters_best"]


class TrainedRecommenderType(Enum):
    TRAIN = "TRAIN"
    TRAIN_VALIDATION = "TRAIN_VALIDATION"


def load_trained_recommender(
    experiment_recommender: commons.ExperimentRecommender,
    experiment_benchmark: commons.ExperimentBenchmark,
    experiment_hyper_parameter_tuning_parameters: commons.HyperParameterTuningParameters,
    data_splits: InteractionsDataSplits,
    similarity: Optional[str],
    model_type: TrainedRecommenderType,
    try_folded_recommender: bool,
) -> Optional[BaseRecommender]:
    """Loads to memory an already-trained recommender.

    This function loads the requested recommender (`experiment_recommender`) on disk. It can load a folded-in
    or the original version of the recommender. If the recommender cannot be loaded, then it returns None.
    """
    if TrainedRecommenderType.TRAIN == model_type:
        urm_train = data_splits.sp_urm_train.copy()
        file_name_postfix = "best_model"
    elif TrainedRecommenderType.TRAIN_VALIDATION == model_type:
        urm_train = data_splits.sp_urm_train_validation.copy()
        file_name_postfix = "best_model_last"
    else:
        raise ValueError(
            f"{load_trained_recommender.__name__} failed because it received an invalid instance of the "
            f"enum {TrainedRecommenderType} (received value {model_type}). Valid values are "
            f"{list(TrainedRecommenderType)}"
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

    folder_path = DIR_TRAINED_MODELS_BASELINES.format(
        benchmark=experiment_benchmark.benchmark.value,
        evaluation_strategy=experiment_hyper_parameter_tuning_parameters.evaluation_strategy.value,
    )

    try:
        trained_recommender_instance = experiment_recommender.recommender(
            URM_train=urm_train,
        )
        trained_recommender_instance.load_model(
            folder_path=folder_path,
            file_name=f"{recommender_name}_{file_name_postfix}",
        )
        trained_recommender_instance.RECOMMENDER_NAME = recommender_name
    except:
        logger.warning(
            f"Could not load the recommender {recommender_name} for the benchmark {experiment_benchmark.benchmark} "
            f"model type {model_type} similarity {similarity} and the following hyper-parameters"
            f" {experiment_hyper_parameter_tuning_parameters} "
        )
        return None

    if try_folded_recommender:
        can_recommender_be_folded = (
            FoldedMatrixFactorizationRecommender.can_recommender_be_folded(
                recommender_instance=trained_recommender_instance,
            )
        )

        if can_recommender_be_folded:
            trained_recommender_instance = cast(
                BaseMatrixFactorizationRecommender,
                trained_recommender_instance,
            )

            file_name_prefix = (
                FoldedMatrixFactorizationRecommender.RECOMMENDER_NAME.replace(
                    "Recommender", ""
                )
            )
            file_name_prefix = f"{file_name_prefix}_{experiment_recommender.recommender.RECOMMENDER_NAME}"

            try:
                trained_folded_recommender_instance = (
                    FoldedMatrixFactorizationRecommender(
                        urm_train=urm_train.copy(),
                        trained_recommender=trained_recommender_instance,
                    )
                )
                trained_folded_recommender_instance.load_model(
                    folder_path=folder_path,
                    file_name=f"{file_name_prefix}_{file_name_postfix}",
                )
            except:
                logger.warning(
                    f"Could not load the the folded recommender {file_name_prefix} for the benchmark"
                    f" {experiment_benchmark.benchmark} model type {model_type} similarity {similarity} and the"
                    f" following hyper-parameters {experiment_hyper_parameter_tuning_parameters} "
                )
                return None

            return trained_folded_recommender_instance
        else:
            return None
    else:
        return trained_recommender_instance


def _run_baselines_folded_hyper_parameter_tuning(
    experiment_case: commons.ExperimentCase,
) -> None:
    """
    Runs in a dask worker the hyper-parameter tuning of folded recommender. This method return early if the base
    recommender is not tuned or cannot be folded, e.g., if its a similarity-based recommender.

    This method should not be called from outside.
    """
    experiment_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
        experiment_case.benchmark
    ]
    experiment_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
        experiment_case.recommender
    ]
    experiment_folded = commons.MAPPER_AVAILABLE_RECOMMENDERS[
        commons.RecommenderFolded.FOLDED
    ]
    experiment_hyper_parameter_tuning_parameters = (
        commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
            experiment_case.hyper_parameter_tuning_parameters
        ]
    )

    benchmark_reader = commons.get_reader_from_benchmark(
        benchmark_config=experiment_benchmark.config,
        benchmark=experiment_benchmark.benchmark,
    )

    dataset = benchmark_reader.dataset

    interactions_data_splits = dataset.get_urm_splits(
        evaluation_strategy=experiment_hyper_parameter_tuning_parameters.evaluation_strategy,
    )

    baseline_recommender_trained_train = load_trained_recommender(
        experiment_benchmark=experiment_benchmark,
        experiment_recommender=experiment_recommender,
        experiment_hyper_parameter_tuning_parameters=experiment_hyper_parameter_tuning_parameters,
        data_splits=interactions_data_splits,
        similarity=None,  # No similarity is used in folded recommenders.
        model_type=TrainedRecommenderType.TRAIN,
        try_folded_recommender=False,
    )

    baseline_recommender_trained_train_validation = load_trained_recommender(
        experiment_benchmark=experiment_benchmark,
        experiment_recommender=experiment_recommender,
        experiment_hyper_parameter_tuning_parameters=experiment_hyper_parameter_tuning_parameters,
        data_splits=interactions_data_splits,
        similarity=None,  # No similarity is used in folded recommenders.
        model_type=TrainedRecommenderType.TRAIN_VALIDATION,
        try_folded_recommender=False,
    )

    if (
        baseline_recommender_trained_train is None
        or baseline_recommender_trained_train_validation is None
    ):
        # We require a recommender that is already optimized.
        logger.warning(
            f"Early-returning from {_run_baselines_folded_hyper_parameter_tuning.__name__}. Could not load trained recommenders for {experiment_recommender.recommender} with the benchmark {experiment_benchmark.benchmark}. "
        )
        return

    if not FoldedMatrixFactorizationRecommender.can_recommender_be_folded(
        recommender_instance=baseline_recommender_trained_train
    ):
        # We require a recommender that can be folded. In case we did not receive it, we return
        # to gracefully say that this case finished (as there is nothing to search).
        logger.warning(
            f"Early-returning from {_run_baselines_folded_hyper_parameter_tuning.__name__} as the loaded recommender "
            f"({baseline_recommender_trained_train.RECOMMENDER_NAME} the one requested in the hyper-parameter search) "
            f"cannot be folded-in, i.e., does not inherit from {BaseMatrixFactorizationRecommender.__name__}."
            f"\n This is not an issue, as not all recommenders cannot be folded-in. This means that there is nothing "
            f"to search here."
        )
        return

    if not FoldedMatrixFactorizationRecommender.can_recommender_be_folded(
        recommender_instance=baseline_recommender_trained_train_validation
    ):
        # We require a recommender that can be folded. In case we did not receive it, we return
        # to gracefully say that this case finished (as there is nothing to search).
        logger.warning(
            f"Early-returning from {_run_baselines_folded_hyper_parameter_tuning.__name__} as the loaded recommender "
            f"({baseline_recommender_trained_train_validation.RECOMMENDER_NAME} the one requested in the "
            f"hyper-parametersearch) cannot be folded-in, i.e., does not inherit from "
            f"{BaseMatrixFactorizationRecommender.__name__}."
            f"\n This is not an issue, as not all recommenders cannot be folded-in. This means that there is nothing "
            f"to search here."
        )
        return

    assert (
        baseline_recommender_trained_train.RECOMMENDER_NAME
        == baseline_recommender_trained_train_validation.RECOMMENDER_NAME
    )

    experiments_folder_path = DIR_TRAINED_MODELS_BASELINES.format(
        benchmark=experiment_benchmark.benchmark.value,
        evaluation_strategy=experiment_hyper_parameter_tuning_parameters.evaluation_strategy.value,
    )
    experiment_file_name_root = f"FoldedMatrixFactorization_{baseline_recommender_trained_train.RECOMMENDER_NAME}"

    import random
    import numpy as np

    random.seed(experiment_hyper_parameter_tuning_parameters.reproducibility_seed)
    np.random.seed(experiment_hyper_parameter_tuning_parameters.reproducibility_seed)

    evaluators = commons.get_evaluators(
        experiment_hyper_parameter_tuning_parameters=experiment_hyper_parameter_tuning_parameters,
        data_splits=interactions_data_splits,
    )

    logger_info = {
        "recommender": experiment_recommender.recommender.RECOMMENDER_NAME,
        "dataset": experiment_benchmark.benchmark.value,
        "urm_test_shape": interactions_data_splits.sp_urm_test.shape,
        "urm_train_shape": interactions_data_splits.sp_urm_train.shape,
        "urm_validation_shape": interactions_data_splits.sp_urm_validation.shape,
        "urm_train_and_validation_shape": interactions_data_splits.sp_urm_train_validation.shape,
        "hyper_parameter_tuning_parameters": experiment_hyper_parameter_tuning_parameters,
    }

    logger.info(f"Hyper-parameter tuning arguments:" f"\n\t* {logger_info}")

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
        experiment_folded.search_hyper_parameters()
    )

    search_bayesian_skopt = SearchBayesianSkopt(
        recommender_class=experiment_folded.recommender,
        evaluator_validation=evaluators.validation,
        evaluator_test=evaluators.test,
        verbose=True,
    )
    search_bayesian_skopt.search(
        cutoff_to_optimize=experiment_hyper_parameter_tuning_parameters.cutoff_to_optimize,
        evaluate_on_test=experiment_hyper_parameter_tuning_parameters.evaluate_on_test,
        hyperparameter_search_space=hyper_parameter_search_space,
        max_total_time=experiment_hyper_parameter_tuning_parameters.max_total_time,
        metric_to_optimize=experiment_hyper_parameter_tuning_parameters.metric_to_optimize,
        n_cases=experiment_hyper_parameter_tuning_parameters.num_cases,
        n_random_starts=experiment_hyper_parameter_tuning_parameters.num_random_starts,
        output_file_name_root=experiment_file_name_root,
        output_folder_path=experiments_folder_path,
        recommender_input_args=recommender_init_validation_args_kwargs,
        recommender_input_args_last_test=recommender_init_test_args_kwargs,
        resume_from_saved=experiment_hyper_parameter_tuning_parameters.resume_from_saved,
        save_metadata=experiment_hyper_parameter_tuning_parameters.save_metadata,
        save_model=experiment_hyper_parameter_tuning_parameters.save_model,
        terminate_on_memory_error=experiment_hyper_parameter_tuning_parameters.terminate_on_memory_error,
    )

    # loaded_trained_recommender = load_trained_recommender(
    #     experiment_recommender=experiment_recommender,
    #     experiment_benchmark=experiment_benchmark,
    #     experiment_hyper_parameter_tuning_parameters=experiment_hyper_parameter_tuning_parameters,
    #     data_splits=interactions_data_splits,
    #     similarity=similarity,
    #     model_type=TrainedRecommenderType.TRAIN_VALIDATION,
    #     try_folded_recommender=True,  # We just searched the hyper-parameters of a folded-recommender.
    # )
    #
    # if loaded_trained_recommender is None:
    #     return
    #
    # path_trained_models = DIR_TRAINED_MODELS_BASELINES.format(
    #     benchmark=experiment_benchmark.benchmark.value,
    #     evaluation_strategy=experiment_hyper_parameter_tuning_parameters.evaluation_strategy.value,
    # )
    # experiment_file_name_root = (
    #     f"{loaded_trained_recommender.RECOMMENDER_NAME}_best_model_last"
    # )

    # evaluators.test.compute_recommender_confidence_intervals(
    #     recommender=loaded_trained_recommender,
    #     recommender_name=experiment_file_name_root,
    #     folder_export_results=path_trained_models,
    # )


def _run_baselines_hyper_parameter_tuning(
    experiment_case: commons.ExperimentCase,
) -> None:
    """
    Runs in a dask worker the hyper-parameter tuning of a base recommender.

    This method should not be called from outside.
    """
    import logging

    configure_logger()

    logger = logging.getLogger(__name__)
    logger.info(
        "Running hyper-parameter tuning of a baseline, function %(function)s."
        " Received arguments: benchmark=%(benchmark)s - recommender=%(recommender)s - hyper_parameters=%(hyper_parameters)s",
        {
            "function": _run_baselines_hyper_parameter_tuning.__name__,
            "benchmark": experiment_case.benchmark,
            "recommender": experiment_case.recommender,
            "hyper_parameters": experiment_case.hyper_parameter_tuning_parameters,
        },
    )

    experiment_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
        experiment_case.benchmark
    ]
    experiment_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
        experiment_case.recommender
    ]
    experiment_hyper_parameter_tuning_parameters = (
        commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
            experiment_case.hyper_parameter_tuning_parameters
        ]
    )

    benchmark_reader = commons.get_reader_from_benchmark(
        benchmark_config=experiment_benchmark.config,
        benchmark=experiment_benchmark.benchmark,
    )

    dataset = benchmark_reader.dataset

    data_splits = dataset.get_urm_splits(
        evaluation_strategy=experiment_hyper_parameter_tuning_parameters.evaluation_strategy
    )

    import random
    import numpy as np

    random.seed(experiment_hyper_parameter_tuning_parameters.reproducibility_seed)
    np.random.seed(experiment_hyper_parameter_tuning_parameters.reproducibility_seed)

    experiments_folder_path = DIR_TRAINED_MODELS_BASELINES.format(
        benchmark=experiment_benchmark.benchmark.value,
        evaluation_strategy=experiment_hyper_parameter_tuning_parameters.evaluation_strategy.value,
    )
    evaluators = commons.get_evaluators(
        experiment_hyper_parameter_tuning_parameters=experiment_hyper_parameter_tuning_parameters,
        data_splits=data_splits,
    )

    logger_info = {
        "recommender": experiment_recommender.recommender.RECOMMENDER_NAME,
        "dataset": experiment_benchmark.benchmark.value,
        "urm_test_shape": data_splits.sp_urm_test.shape,
        "urm_train_shape": data_splits.sp_urm_train.shape,
        "urm_validation_shape": data_splits.sp_urm_validation.shape,
        "urm_train_and_validation_shape": data_splits.sp_urm_train_validation.shape,
        "hyper_parameter_tuning_parameters": repr(
            experiment_hyper_parameter_tuning_parameters
        ),
    }

    logger.info(
        "Hyper-parameter tuning arguments: \n\t* %(logger_info)s",
        {"logger_info": logger_info},
    )
    run_hyper_parameter_search_collaborative(
        allow_weighting=True,
        allow_bias_URM=False,
        allow_dropout_MF=False,
        cutoff_to_optimize=experiment_hyper_parameter_tuning_parameters.cutoff_to_optimize,
        evaluator_test=evaluators.test,
        evaluator_validation=evaluators.validation,
        evaluator_validation_earlystopping=evaluators.validation_early_stopping,
        evaluate_on_test=experiment_hyper_parameter_tuning_parameters.evaluate_on_test,
        max_total_time=experiment_hyper_parameter_tuning_parameters.max_total_time,
        metric_to_optimize=experiment_hyper_parameter_tuning_parameters.metric_to_optimize,
        n_cases=experiment_hyper_parameter_tuning_parameters.num_cases,
        n_random_starts=experiment_hyper_parameter_tuning_parameters.num_random_starts,
        output_folder_path=experiments_folder_path,
        parallelizeKNN=False,
        recommender_class=experiment_recommender.recommender,
        resume_from_saved=experiment_hyper_parameter_tuning_parameters.resume_from_saved,
        similarity_type_list=experiment_hyper_parameter_tuning_parameters.knn_similarity_types,
        URM_train=data_splits.sp_urm_train,
        URM_train_last_test=data_splits.sp_urm_train_validation,
        save_model=experiment_hyper_parameter_tuning_parameters.save_model,
    )


def run_baselines_experiments(
    dask_interface: DaskInterface,
    experiment_cases_interface: commons.ExperimentCasesInterface,
) -> None:
    """
    Public method that tells Dask to run the hyper-parameter tuning of recommenders. This function instructs Dask to
    execute the hyper-parameter tuning of each recommender in a separate worker. Processes should be always preferred
    instead of threads, as the hyper-parameter tuning loops are not thread-safe. Validations are not done in case of
    debug.
    """
    for experiment_case in experiment_cases_interface.experiment_cases:
        experiment_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
            experiment_case.benchmark
        ]
        experiment_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
            experiment_case.recommender
        ]

        # TODO: FERNANDO-DEBUGGER. REMOVE WHEN DEBUGGED.
        # try:
        #     _run_baselines_hyper_parameter_tuning(
        #         experiment_case=experiment_case,
        #     )
        # except Exception as e:
        #     import pdb

        #     pdb.set_trace()

        #     print("FUNCTION FAILED. INSPECT")
        #     print(e)

        # TODO: FERNANDO-DEBUGGER. INSTANTIATE AGAIN DASK.
        dask_interface.submit_job(
            job_key=(
                f"_run_baselines_hyper_parameter_tuning"
                f"|{experiment_benchmark.benchmark.value}"
                f"|{experiment_recommender.recommender.RECOMMENDER_NAME}"
                f"|{uuid.uuid4()}"
            ),
            job_priority=experiment_benchmark.priority
            + experiment_recommender.priority,
            job_info={
                "recommender": experiment_recommender.recommender.RECOMMENDER_NAME,
                "benchmark": experiment_benchmark.benchmark.value,
            },
            method=_run_baselines_hyper_parameter_tuning,
            method_kwargs={
                "experiment_case": experiment_case,
            },
        )


def run_baselines_folded(
    dask_interface: DaskInterface,
    experiment_cases_interface: commons.ExperimentCasesInterface,
) -> None:
    """
    Public method that tells Dask to run the hyper-parameter tuning of folded recommenders. This function instructs
    Dask to execute the hyper-parameter tuning of each recommender in a separate worker. Processes should be always
    preferred instead of threads, as the hyper-parameter tuning loops are not thread-safe. Validations are not done
    in case of debug.
    """
    for experiment_case in experiment_cases_interface.experiment_cases:
        experiment_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
            experiment_case.benchmark
        ]
        experiment_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
            experiment_case.recommender
        ]
        experiment_folded = commons.MAPPER_AVAILABLE_RECOMMENDERS[
            commons.RecommenderFolded.FOLDED
        ]

        dask_interface.submit_job(
            job_key=(
                f"_run_baselines_folded_hyper_parameter_tuning"
                f"|{experiment_benchmark.benchmark.value}"
                f"|{experiment_recommender.recommender.RECOMMENDER_NAME}"
                f"|{experiment_folded.recommender.RECOMMENDER_NAME}"
                f"|{uuid.uuid4()}"
            ),
            job_priority=experiment_benchmark.priority
            + experiment_recommender.priority,
            job_info={
                "recommender": experiment_recommender.recommender.RECOMMENDER_NAME,
                "recommender_folded": experiment_folded.recommender.RECOMMENDER_NAME,
                "benchmark": experiment_benchmark.benchmark.value,
            },
            method=_run_baselines_folded_hyper_parameter_tuning,
            method_kwargs={
                "experiment_case": experiment_case,
            },
        )
