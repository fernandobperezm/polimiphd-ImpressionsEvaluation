import logging
import os
import uuid
from enum import Enum
from typing import Type, Optional, Any

import Recommenders.Recommender_import_list as recommenders
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from Recommenders.BaseRecommender import BaseRecommender
from recsys_framework_extensions.dask import DaskInterface
from recsys_framework_extensions.data.io import DataIO
from recsys_framework_extensions.data.mixins import InteractionsDataSplits
from recsys_framework_extensions.hyper_parameter_search import (
    run_hyper_parameter_search_collaborative,
)

import impressions_evaluation.experiments.commons as commons
from impressions_evaluation import configure_logger

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
    *,
    experiment_recommender: commons.ExperimentRecommender,
    experiment_benchmark: commons.ExperimentBenchmark,
    experiment_hyper_parameter_tuning_parameters: commons.HyperParameterTuningParameters,
    data_splits: InteractionsDataSplits,
    similarity: Optional[str],
    model_type: TrainedRecommenderType,
) -> Optional[BaseRecommender]:
    """Loads to memory an already-trained recommender.

    This function loads the requested recommender (`experiment_recommender`) on disk. If the recommender cannot be loaded, then it returns None.
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

        return trained_recommender_instance
    except:
        logger.warning(
            "Could not load the recommender %(recommender_name)s for the benchmark %(benchmark)s model type %(model_type)s similarity %(similarity)s and the following hyper-parameters %(hyper_parameters)s",
            {
                "recommender_name": recommender_name,
                "benchmark": experiment_benchmark.benchmark,
                "model_type": model_type,
                "similarity": similarity,
                "hyper_parameters": experiment_hyper_parameter_tuning_parameters,
            },
        )
        return None


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
