import logging
import os
import uuid

import attrs
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from recsys_framework_extensions.dask import DaskInterface
from recsys_framework_extensions.hyper_parameter_search import (
    SearchBayesianSkopt,
)

from impressions_evaluation.experiments import commons
from impressions_evaluation.experiments.baselines import DIR_TRAINED_MODELS_BASELINES

logger = logging.getLogger(__name__)


DIR_TRAINED_MODELS_IMPRESSION_AWARE = os.path.join(
    commons.DIR_TRAINED_MODELS,
    "{benchmark}",
    "{evaluation_strategy}",
    "",
)


def _run_collaborative_filtering_hyper_parameter_tuning(
    experiment_case: commons.ExperimentCase,
) -> None:
    experiment_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
        experiment_case.benchmark
    ]
    # experiment_recommender = _MAPPER_COLLABORATIVE_RECOMMENDERS[
    experiment_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
        experiment_case.recommender
    ]
    experiment_hyper_parameter_tuning_parameters = (
        commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
            experiment_case.hyper_parameter_tuning_parameters
        ]
    )

    assert experiment_recommender.search_hyper_parameters is not None

    benchmark_reader = commons.get_reader_from_benchmark(
        benchmark_config=experiment_benchmark.config,
        benchmark=experiment_benchmark.benchmark,
    )

    dataset = benchmark_reader.dataset

    interactions_data_splits = dataset.get_urm_splits(
        evaluation_strategy=experiment_hyper_parameter_tuning_parameters.evaluation_strategy
    )

    experiments_folder_path = DIR_TRAINED_MODELS_BASELINES.format(
        benchmark=experiment_benchmark.benchmark.value,
        evaluation_strategy=experiment_hyper_parameter_tuning_parameters.evaluation_strategy.value,
    )
    experiment_file_name_root = experiment_recommender.recommender.RECOMMENDER_NAME

    import random
    import numpy as np
    import torch

    random.seed(experiment_hyper_parameter_tuning_parameters.reproducibility_seed)
    np.random.seed(experiment_hyper_parameter_tuning_parameters.reproducibility_seed)
    torch.manual_seed(experiment_hyper_parameter_tuning_parameters.reproducibility_seed)

    evaluators = commons.get_evaluators(
        data_splits=interactions_data_splits,
        experiment_hyper_parameter_tuning_parameters=experiment_hyper_parameter_tuning_parameters,
    )

    fit_kwargs = (
        attrs.asdict(experiment_recommender.fit_keyword_parameters())
        if experiment_recommender.fit_keyword_parameters is not None
        else {}
    )

    early_stopping_kwargs = (
        attrs.asdict(
            commons.get_early_stopping_configuration(
                evaluators=evaluators,
                hyper_parameter_tuning_parameters=experiment_hyper_parameter_tuning_parameters,
            )
        )
        if experiment_recommender.do_early_stopping
        else {}
    )

    recommender_init_validation_args_kwargs = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[],
        CONSTRUCTOR_KEYWORD_ARGS={
            "urm_train": interactions_data_splits.sp_urm_train,
            "use_gpu": experiment_recommender.use_gpu,
        },
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS=fit_kwargs,
        EARLYSTOPPING_KEYWORD_ARGS=early_stopping_kwargs,
    )

    recommender_init_test_args_kwargs = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[],
        CONSTRUCTOR_KEYWORD_ARGS={
            "urm_train": interactions_data_splits.sp_urm_train_validation,
            "use_gpu": experiment_recommender.use_gpu,
        },
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS=fit_kwargs,
        EARLYSTOPPING_KEYWORD_ARGS={},
    )

    hyper_parameter_search_space = attrs.asdict(
        experiment_recommender.search_hyper_parameters()
    )

    logger_info = {
        "recommender": experiment_recommender.recommender.RECOMMENDER_NAME,
        "use_gpu": experiment_recommender.use_gpu,
        "dataset": experiment_benchmark.benchmark.value,
        "urm_test_shape": interactions_data_splits.sp_urm_test.shape,
        "urm_train_shape": interactions_data_splits.sp_urm_train.shape,
        "urm_validation_shape": interactions_data_splits.sp_urm_validation.shape,
        "urm_train_and_validation_shape": interactions_data_splits.sp_urm_train_validation.shape,
        "fit_kwargs": fit_kwargs,
        "early_stopping_kwargs": early_stopping_kwargs,
        "hyper_parameter_tuning_parameters": repr(
            experiment_hyper_parameter_tuning_parameters
        ),
        "hyper_parameter_search_space": hyper_parameter_search_space,
    }

    logger.info(
        "Hyper-parameter tuning arguments:\n\t* %(logger_info)s",
        {"logger_info": logger_info},
    )

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


def _run_pure_impressions_hyper_parameter_tuning(
    experiment_case: commons.ExperimentCase,
) -> None:
    experiment_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
        experiment_case.benchmark
    ]
    # experiment_recommender = _MAPPER_IMPRESSIONS_RECOMMENDERS[
    experiment_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
        experiment_case.recommender
    ]
    experiment_hyper_parameter_tuning_parameters = (
        commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
            experiment_case.hyper_parameter_tuning_parameters
        ]
    )

    assert experiment_recommender.search_hyper_parameters is not None

    benchmark_reader = commons.get_reader_from_benchmark(
        benchmark_config=experiment_benchmark.config,
        benchmark=experiment_benchmark.benchmark,
    )

    dataset = benchmark_reader.dataset

    interactions_data_splits = dataset.get_urm_splits(
        evaluation_strategy=experiment_hyper_parameter_tuning_parameters.evaluation_strategy
    )
    impressions_data_splits = dataset.get_uim_splits(
        evaluation_strategy=experiment_hyper_parameter_tuning_parameters.evaluation_strategy
    )

    experiments_folder_path = DIR_TRAINED_MODELS_IMPRESSION_AWARE.format(
        benchmark=experiment_benchmark.benchmark.value,
        evaluation_strategy=experiment_hyper_parameter_tuning_parameters.evaluation_strategy.value,
    )
    experiment_file_name_root = experiment_recommender.recommender.RECOMMENDER_NAME

    import random
    import numpy as np
    import torch

    random.seed(experiment_hyper_parameter_tuning_parameters.reproducibility_seed)
    np.random.seed(experiment_hyper_parameter_tuning_parameters.reproducibility_seed)
    torch.manual_seed(experiment_hyper_parameter_tuning_parameters.reproducibility_seed)

    evaluators = commons.get_evaluators(
        data_splits=interactions_data_splits,
        experiment_hyper_parameter_tuning_parameters=experiment_hyper_parameter_tuning_parameters,
    )

    early_stopping_kwargs = (
        attrs.asdict(
            commons.get_early_stopping_configuration(
                evaluators=evaluators,
                hyper_parameter_tuning_parameters=experiment_hyper_parameter_tuning_parameters,
            )
        )
        if experiment_recommender.do_early_stopping
        else {}
    )

    recommender_init_validation_args_kwargs = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[],
        CONSTRUCTOR_KEYWORD_ARGS={
            "urm_train": interactions_data_splits.sp_urm_train,
            "uim_train": impressions_data_splits.sp_uim_train,
            "use_gpu": experiment_recommender.use_gpu,
        },
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS=early_stopping_kwargs,
    )

    recommender_init_test_args_kwargs = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[],
        CONSTRUCTOR_KEYWORD_ARGS={
            "urm_train": interactions_data_splits.sp_urm_train_validation,
            "uim_train": impressions_data_splits.sp_uim_train_validation,
            "use_gpu": experiment_recommender.use_gpu,
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
        "use_gpu": experiment_recommender.use_gpu,
        "dataset": experiment_benchmark.benchmark.value,
        "urm_test_shape": interactions_data_splits.sp_urm_test.shape,
        "urm_train_shape": interactions_data_splits.sp_urm_train.shape,
        "urm_validation_shape": interactions_data_splits.sp_urm_validation.shape,
        "urm_train_and_validation_shape": interactions_data_splits.sp_urm_train_validation.shape,
        "hyper_parameter_tuning_parameters": repr(
            experiment_hyper_parameter_tuning_parameters
        ),
        "hyper_parameter_search_space": hyper_parameter_search_space,
    }

    logger.info(
        "Hyper-parameter tuning arguments:\n\t* %(logger_info)s",
        {"logger_info": logger_info},
    )

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


def _run_frequency_impressions_hyper_parameter_tuning(
    experiment_case: commons.ExperimentCase,
) -> None:
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

    assert experiment_recommender.search_hyper_parameters is not None

    benchmark_reader = commons.get_reader_from_benchmark(
        benchmark_config=experiment_benchmark.config,
        benchmark=experiment_benchmark.benchmark,
    )

    dataset = benchmark_reader.dataset

    interactions_data_splits = dataset.get_urm_splits(
        evaluation_strategy=experiment_hyper_parameter_tuning_parameters.evaluation_strategy
    )
    impressions_data_splits = dataset.get_uim_splits(
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

    experiments_folder_path = DIR_TRAINED_MODELS_IMPRESSION_AWARE.format(
        benchmark=experiment_benchmark.benchmark.value,
        evaluation_strategy=experiment_hyper_parameter_tuning_parameters.evaluation_strategy.value,
    )
    experiment_file_name_root = experiment_recommender.recommender.RECOMMENDER_NAME

    import random
    import numpy as np
    import torch

    random.seed(experiment_hyper_parameter_tuning_parameters.reproducibility_seed)
    np.random.seed(experiment_hyper_parameter_tuning_parameters.reproducibility_seed)
    torch.manual_seed(experiment_hyper_parameter_tuning_parameters.reproducibility_seed)

    evaluators = commons.get_evaluators(
        data_splits=interactions_data_splits,
        experiment_hyper_parameter_tuning_parameters=experiment_hyper_parameter_tuning_parameters,
    )

    early_stopping_kwargs = (
        attrs.asdict(
            commons.get_early_stopping_configuration(
                evaluators=evaluators,
                hyper_parameter_tuning_parameters=experiment_hyper_parameter_tuning_parameters,
            )
        )
        if experiment_recommender.do_early_stopping
        else {}
    )

    recommender_init_validation_args_kwargs = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[],
        CONSTRUCTOR_KEYWORD_ARGS={
            "urm_train": interactions_data_splits.sp_urm_train,
            "uim_train": impressions_data_splits.sp_uim_train,
            "uim_frequency": impressions_feature_frequency_train,
            "use_gpu": experiment_recommender.use_gpu,
        },
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS=early_stopping_kwargs,
    )

    recommender_init_test_args_kwargs = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[],
        CONSTRUCTOR_KEYWORD_ARGS={
            "urm_train": interactions_data_splits.sp_urm_train_validation,
            "uim_train": impressions_data_splits.sp_uim_train_validation,
            "uim_frequency": impressions_feature_frequency_train_validation,
            "use_gpu": experiment_recommender.use_gpu,
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
        "use_gpu": experiment_recommender.use_gpu,
        "dataset": experiment_benchmark.benchmark.value,
        "urm_test_shape": interactions_data_splits.sp_urm_test.shape,
        "urm_train_shape": interactions_data_splits.sp_urm_train.shape,
        "urm_validation_shape": interactions_data_splits.sp_urm_validation.shape,
        "urm_train_and_validation_shape": interactions_data_splits.sp_urm_train_validation.shape,
        "hyper_parameter_tuning_parameters": repr(
            experiment_hyper_parameter_tuning_parameters
        ),
        "early_stopping_kwargs": early_stopping_kwargs,
        "hyper_parameter_search_space": hyper_parameter_search_space,
    }

    logger.info(
        "Hyper-parameter tuning arguments:\n\t* %(logger_info)s",
        {"logger_info": logger_info},
    )

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


def run_experiments_dask(
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

        training_function = experiment_case.training_function

        dask_interface.submit_job(
            job_key=(
                f"{training_function.__name__}"
                f"|{experiment_benchmark.benchmark.value}"
                f"|{experiment_recommender.recommender.RECOMMENDER_NAME}"
                f"|{uuid.uuid4()}"
            ),
            job_priority=experiment_benchmark.priority
            * experiment_recommender.priority,
            job_info={
                "recommender": experiment_recommender.recommender.RECOMMENDER_NAME,
                "benchmark": experiment_benchmark.benchmark.value,
            },
            method=training_function,
            method_kwargs={
                "experiment_case": experiment_case,
            },
        )


def run_experiments_sequentially(
    experiment_cases_interface: commons.ExperimentCasesInterface,
) -> None:
    """ """
    experiment_cases = sorted(
        experiment_cases_interface.experiment_cases,
        key=lambda ex: commons.MAPPER_AVAILABLE_BENCHMARKS[ex.benchmark].priority
        + commons.MAPPER_AVAILABLE_RECOMMENDERS[ex.recommender].priority,
        reverse=True,
    )

    for experiment_case in experiment_cases:
        logger.info(
            "Training recommender: \n %(experiment_case)s",
            {"experiment_case": attrs.asdict(experiment_case)},
        )
        experiment_case.training_function(
            experiment_case=experiment_case,
        )
