import os
import uuid

import attrs
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.SearchSingleCase import SearchSingleCase
from recsys_framework_extensions.dask import DaskInterface
from recsys_framework_extensions.logging import get_logger

import experiments.commons as commons
from impression_recommenders.heuristics.frequency_and_recency import FrequencyRecencyRecommender, RecencyRecommender

logger = get_logger(__name__)


####################################################################################################
####################################################################################################
#                                FOLDERS VARIABLES                            #
####################################################################################################
####################################################################################################
BASE_FOLDER = os.path.join(
    commons.RESULTS_EXPERIMENTS_DIR,
    "heuristics",
    "{benchmark}",
    "{evaluation_strategy}",
    "",
)
HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR = os.path.join(
    BASE_FOLDER,
    "experiments",
    ""
)

commons.FOLDERS.add(BASE_FOLDER)
commons.FOLDERS.add(HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR)


####################################################################################################
####################################################################################################
#                               Hyper-parameter tuning of Heuristic                                #
####################################################################################################
####################################################################################################
def _run_impressions_heuristics_hyper_parameter_tuning(
    experiment_case: commons.ExperimentCase,
) -> None:
    """
    Runs in a dask worker the hyper-parameter tuning of a time-aware impression recommender.

    This method should not be called from outside.
    """
    
    experiment_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[experiment_case.benchmark]
    experiment_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[experiment_case.recommender]
    experiment_hyper_parameter_tuning_parameters = commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
        experiment_case.hyper_parameter_tuning_parameters
    ]

    assert experiment_recommender.search_hyper_parameters is not None

    benchmark_reader = commons.get_reader_from_benchmark(
        benchmark_config=experiment_benchmark.config,
        benchmark=experiment_benchmark.benchmark,
    )
    
    dataset = benchmark_reader.dataset

    interactions_data_splits = dataset.get_urm_splits(
        evaluation_strategy=experiment_hyper_parameter_tuning_parameters.evaluation_strategy
    )

    impressions_feature_frequency_train = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_benchmark.benchmark,
            evaluation_strategy=experiment_hyper_parameter_tuning_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_FREQUENCY,
            impressions_feature_column=commons.ImpressionsFeatureColumnsFrequency.FREQUENCY,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
        )
    )
    impressions_feature_frequency_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_benchmark.benchmark,
            evaluation_strategy=experiment_hyper_parameter_tuning_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_FREQUENCY,
            impressions_feature_column=commons.ImpressionsFeatureColumnsFrequency.FREQUENCY,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    impressions_feature_position_train = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_benchmark.benchmark,
            evaluation_strategy=experiment_hyper_parameter_tuning_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_POSITION,
            impressions_feature_column=commons.ImpressionsFeatureColumnsPosition.POSITION,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
        )
    )
    impressions_feature_position_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_benchmark.benchmark,
            evaluation_strategy=experiment_hyper_parameter_tuning_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_POSITION,
            impressions_feature_column=commons.ImpressionsFeatureColumnsPosition.POSITION,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    impressions_feature_timestamp_train = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_benchmark.benchmark,
            evaluation_strategy=experiment_hyper_parameter_tuning_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_TIMESTAMP,
            impressions_feature_column=commons.ImpressionsFeatureColumnsTimestamp.TIMESTAMP,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
        )
    )
    impressions_feature_timestamp_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_benchmark.benchmark,
            evaluation_strategy=experiment_hyper_parameter_tuning_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_TIMESTAMP,
            impressions_feature_column=commons.ImpressionsFeatureColumnsTimestamp.TIMESTAMP,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    experiments_folder_path = HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR.format(
        benchmark=experiment_benchmark.benchmark.value,
        evaluation_strategy=experiment_hyper_parameter_tuning_parameters.evaluation_strategy.value,
    )
    experiment_file_name_root = experiment_recommender.recommender.RECOMMENDER_NAME

    import random
    import numpy as np

    random.seed(experiment_hyper_parameter_tuning_parameters.reproducibility_seed)
    np.random.seed(experiment_hyper_parameter_tuning_parameters.reproducibility_seed)

    evaluators = commons.get_evaluators(
        data_splits=interactions_data_splits,
        experiment_hyper_parameter_tuning_parameters=experiment_hyper_parameter_tuning_parameters,
    )

    recommender_init_validation_args_kwargs = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[],
        CONSTRUCTOR_KEYWORD_ARGS={
            "urm_train": interactions_data_splits.sp_urm_train,
            "uim_frequency": impressions_feature_frequency_train,
            "uim_position": impressions_feature_position_train,
            "uim_timestamp": impressions_feature_timestamp_train,
            "seed": experiment_hyper_parameter_tuning_parameters.reproducibility_seed,
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
            "seed": experiment_hyper_parameter_tuning_parameters.reproducibility_seed,
        },
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS={},
    )

    hyper_parameter_search_space = attrs.asdict(
        experiment_recommender.search_hyper_parameters()
    )

    logger_info = {
        "recommender": experiment_recommender.recommender.RECOMMENDER_NAME,
        "dataset": experiment_benchmark.benchmark.value,
        "urm_test_shape": interactions_data_splits.sp_urm_test.shape,
        "urm_train_shape": interactions_data_splits.sp_urm_train.shape,
        "urm_validation_shape": interactions_data_splits.sp_urm_validation.shape,
        "urm_train_and_validation_shape": interactions_data_splits.sp_urm_train_validation.shape,
        "hyper_parameter_tuning_parameters": repr(experiment_hyper_parameter_tuning_parameters),
        "hyper_parameter_search_space": hyper_parameter_search_space,
    }

    logger.info(
        f"Hyper-parameter tuning arguments:"
        f"\n\t* {logger_info}"
    )

    if experiment_recommender.recommender in [FrequencyRecencyRecommender, RecencyRecommender]:
        search_bayesian_skopt = SearchBayesianSkopt(
            recommender_class=experiment_recommender.recommender,
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
    else:
        search_single_case = SearchSingleCase(
            recommender_class=experiment_recommender.recommender,
            evaluator_validation=evaluators.validation,
            evaluator_test=evaluators.test,
            verbose=True,
        )

        search_single_case.search(
            cutoff_to_optimize=experiment_hyper_parameter_tuning_parameters.cutoff_to_optimize,

            evaluate_on_test=experiment_hyper_parameter_tuning_parameters.evaluate_on_test,

            fit_hyperparameters_values={},

            metric_to_optimize=experiment_hyper_parameter_tuning_parameters.metric_to_optimize,

            output_file_name_root=experiment_file_name_root,
            output_folder_path=experiments_folder_path,

            recommender_input_args=recommender_init_validation_args_kwargs,
            recommender_input_args_last_test=recommender_init_test_args_kwargs,
            resume_from_saved=experiment_hyper_parameter_tuning_parameters.resume_from_saved,

            save_metadata=experiment_hyper_parameter_tuning_parameters.save_metadata,
            save_model=experiment_hyper_parameter_tuning_parameters.save_model,

            terminate_on_memory_error=experiment_hyper_parameter_tuning_parameters.terminate_on_memory_error,
        )


def run_impressions_heuristics_experiments(
    dask_interface: DaskInterface,
    experiment_cases_interface: commons.ExperimentCasesInterface,
) -> None:
    """
    Public method that instructs dask to run in dask workers the hyper-parameter tuning of time-aware recommenders.
    Processes are always preferred than threads as the hyper-parameter tuning loop is probably not thread-safe.
    """
    for experiment_case in experiment_cases_interface.experiment_cases:
        experiment_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[experiment_case.benchmark]
        experiment_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[experiment_case.recommender]

        dask_interface.submit_job(
            job_key=(
                f"_run_impressions_heuristics_hyper_parameter_tuning"
                f"|{experiment_benchmark.benchmark.value}"
                f"|{experiment_recommender.recommender.RECOMMENDER_NAME}"
                f"|{uuid.uuid4()}"
            ),
            job_priority=experiment_benchmark.priority * experiment_recommender.priority,
            job_info={
                "recommender": experiment_recommender.recommender.RECOMMENDER_NAME,
                "benchmark": experiment_benchmark.benchmark.value,
            },
            method=_run_impressions_heuristics_hyper_parameter_tuning,
            method_kwargs={
                "experiment_case": experiment_case,
            }
        )
