import itertools
import os
import uuid
from typing import Type, Optional, Sequence, cast, Union

import Recommenders.Recommender_import_list as recommenders
import attrs
import numpy as np
import scipy.sparse as sp
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from recsys_framework_extensions.dask import DaskInterface
from recsys_framework_extensions.data.mixins import InteractionsDataSplits
from recsys_framework_extensions.logging import get_logger
from recsys_framework_extensions.plotting import generate_accuracy_and_beyond_metrics_latex


import experiments.commons as commons
from experiments.baselines import load_trained_recommender, TrainedRecommenderType
from impression_recommenders.re_ranking.cycling import CyclingRecommender
from impression_recommenders.re_ranking.impressions_discounting import ImpressionsDiscountingRecommender


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
def _run_impressions_re_ranking_hyper_parameter_tuning(
    experiment_case_reranking: commons.ExperimentCase,
    experiment_case_baseline: commons.ExperimentCase,
    experiment_baseline_similarity: Optional[str],
) -> None:
    """TODO: fernando-debugger| complete.
    """

    experiment_re_ranking_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
        experiment_case_reranking.benchmark
    ]
    experiment_re_ranking_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
        experiment_case_reranking.recommender
    ]
    experiment_re_ranking_hyper_parameters = commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
        experiment_case_reranking.hyper_parameter_tuning_parameters
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

    assert experiment_re_ranking_recommender.search_hyper_parameters is not None

    benchmark_reader = commons.get_reader_from_benchmark(
        benchmark_config=experiment_re_ranking_benchmark.config,
        benchmark=experiment_re_ranking_benchmark.benchmark,
    )
    
    dataset = benchmark_reader.dataset

    interactions_data_splits = dataset.get_urm_splits(
        evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy
    )

    impressions_feature_frequency_train = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking_benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_FREQUENCY,
            impressions_feature_column=commons.ImpressionsFeatureColumnsFrequency.FREQUENCY,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
        )
    )
    impressions_feature_frequency_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking_benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_FREQUENCY,
            impressions_feature_column=commons.ImpressionsFeatureColumnsFrequency.FREQUENCY,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    impressions_feature_position_train = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking_benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_POSITION,
            impressions_feature_column=commons.ImpressionsFeatureColumnsPosition.POSITION,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
        )
    )
    impressions_feature_position_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking_benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_POSITION,
            impressions_feature_column=commons.ImpressionsFeatureColumnsPosition.POSITION,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    impressions_feature_timestamp_train = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking_benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_TIMESTAMP,
            impressions_feature_column=commons.ImpressionsFeatureColumnsTimestamp.TIMESTAMP,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
        )
    )
    impressions_feature_timestamp_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking_benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_TIMESTAMP,
            impressions_feature_column=commons.ImpressionsFeatureColumnsTimestamp.TIMESTAMP,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    if commons.Benchmarks.FINNNoSlates == experiment_re_ranking_benchmark.benchmark:
        impressions_feature_last_seen_train = dataset.sparse_matrix_impression_feature(
            feature=commons.get_feature_key_by_benchmark(
                benchmark=experiment_re_ranking_benchmark.benchmark,
                evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
                impressions_feature=commons.ImpressionsFeatures.USER_ITEM_LAST_SEEN,
                impressions_feature_column=commons.ImpressionsFeatureColumnsLastSeen.EUCLIDEAN,
                impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
            )
        )
        impressions_feature_last_seen_train_validation = dataset.sparse_matrix_impression_feature(
            feature=commons.get_feature_key_by_benchmark(
                benchmark=experiment_re_ranking_benchmark.benchmark,
                evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
                impressions_feature=commons.ImpressionsFeatures.USER_ITEM_LAST_SEEN,
                impressions_feature_column=commons.ImpressionsFeatureColumnsLastSeen.EUCLIDEAN,
                impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
            )
        )

    else:
        impressions_feature_last_seen_train = dataset.sparse_matrix_impression_feature(
            feature=commons.get_feature_key_by_benchmark(
                benchmark=experiment_re_ranking_benchmark.benchmark,
                evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
                impressions_feature=commons.ImpressionsFeatures.USER_ITEM_LAST_SEEN,
                impressions_feature_column=commons.ImpressionsFeatureColumnsLastSeen.TOTAL_DAYS,
                impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
            )
        )
        impressions_feature_last_seen_train_validation = dataset.sparse_matrix_impression_feature(
            feature=commons.get_feature_key_by_benchmark(
                benchmark=experiment_re_ranking_benchmark.benchmark,
                evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
                impressions_feature=commons.ImpressionsFeatures.USER_ITEM_LAST_SEEN,
                impressions_feature_column=commons.ImpressionsFeatureColumnsLastSeen.TOTAL_DAYS,
                impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
            )
        )

    # The experiments will be done on all recommenders. However, in order for the impression methods to be comparable
    # within each other, all matrix factorization approaches will be folded-in. This is because the user.profile
    # experiments always need a similarity recommender.
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

    assert baseline_recommender_trained_train.RECOMMENDER_NAME == baseline_recommender_trained_train_validation.RECOMMENDER_NAME

    experiments_folder_path = HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR.format(
        benchmark=experiment_re_ranking_benchmark.benchmark.value,
        evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy.value,
    )
    experiment_file_name_root = (
        f"{experiment_re_ranking_recommender.recommender.RECOMMENDER_NAME}"
        f"_{baseline_recommender_trained_train.RECOMMENDER_NAME}"
    )

    import random
    import numpy as np

    random.seed(experiment_re_ranking_hyper_parameters.reproducibility_seed)
    np.random.seed(experiment_re_ranking_hyper_parameters.reproducibility_seed)

    evaluators = commons.get_evaluators(
        data_splits=interactions_data_splits,
        experiment_hyper_parameter_tuning_parameters=experiment_re_ranking_hyper_parameters,
    )

    logger_info = {
        "recommender": experiment_re_ranking_recommender.recommender.RECOMMENDER_NAME,
        "dataset": experiment_re_ranking_benchmark.benchmark.value,
        "urm_test_shape": interactions_data_splits.sp_urm_test.shape,
        "urm_train_shape": interactions_data_splits.sp_urm_train.shape,
        "urm_validation_shape": interactions_data_splits.sp_urm_validation.shape,
        "urm_train_and_validation_shape": interactions_data_splits.sp_urm_train_validation.shape,
        "hyper_parameter_tuning_parameters": experiment_re_ranking_hyper_parameters.__repr__(),
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
            "uim_last_seen": impressions_feature_last_seen_train,
            "seed": experiment_re_ranking_hyper_parameters.reproducibility_seed,
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
            "uim_last_seen": impressions_feature_last_seen_train_validation,
            "seed": experiment_re_ranking_hyper_parameters.reproducibility_seed,
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
        cutoff_to_optimize=experiment_re_ranking_hyper_parameters.cutoff_to_optimize,

        evaluate_on_test=experiment_re_ranking_hyper_parameters.evaluate_on_test,

        hyperparameter_search_space=hyper_parameter_search_space,

        max_total_time=experiment_re_ranking_hyper_parameters.max_total_time,
        metric_to_optimize=experiment_re_ranking_hyper_parameters.metric_to_optimize,

        n_cases=experiment_re_ranking_hyper_parameters.num_cases,
        n_random_starts=experiment_re_ranking_hyper_parameters.num_random_starts,

        output_file_name_root=experiment_file_name_root,
        output_folder_path=experiments_folder_path,

        recommender_input_args=recommender_init_validation_args_kwargs,
        recommender_input_args_last_test=recommender_init_test_args_kwargs,
        resume_from_saved=experiment_re_ranking_hyper_parameters.resume_from_saved,

        save_metadata=experiment_re_ranking_hyper_parameters.save_metadata,
        save_model=experiment_re_ranking_hyper_parameters.save_model,

        terminate_on_memory_error=experiment_re_ranking_hyper_parameters.terminate_on_memory_error,
    )


def run_impressions_re_ranking_experiments(
    dask_interface: DaskInterface,
    re_ranking_experiment_cases_interface: commons.ExperimentCasesInterface,
    baseline_experiment_cases_interface: commons.ExperimentCasesInterface,
) -> None:
    for experiment_case_reranking in re_ranking_experiment_cases_interface.experiment_cases:
        for experiment_case_baseline in baseline_experiment_cases_interface.experiment_cases:
            if (
                experiment_case_reranking.benchmark != experiment_case_baseline.benchmark
                and experiment_case_reranking.hyper_parameter_tuning_parameters != experiment_case_baseline.hyper_parameter_tuning_parameters
            ):
                continue

            re_ranking_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
                experiment_case_reranking.benchmark
            ]
            re_ranking_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
                experiment_case_reranking.recommender
            ]

            baseline_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
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
                        f"|{experiment_case_reranking.benchmark.value}"
                        f"|{re_ranking_recommender.recommender.RECOMMENDER_NAME}"
                        f"|{baseline_recommender.recommender.RECOMMENDER_NAME}"
                        f"|{similarity}"
                        f"|{uuid.uuid4()}"
                    ),
                    job_priority=(
                        re_ranking_benchmark.priority
                        * re_ranking_recommender.priority
                        * baseline_recommender.priority
                    ),
                    job_info={
                        "recommender": re_ranking_recommender.recommender.RECOMMENDER_NAME,
                        "baseline": baseline_recommender.recommender.RECOMMENDER_NAME,
                        "similarity": similarity,
                        "benchmark": re_ranking_benchmark.benchmark.value,
                    },
                    method=_run_impressions_re_ranking_hyper_parameter_tuning,
                    method_kwargs={
                        "experiment_case_reranking": experiment_case_reranking,
                        "experiment_case_baseline": experiment_case_baseline,
                        "experiment_baseline_similarity": similarity,
                    }
                )


####################################################################################################
####################################################################################################
#             Reproducibility study: Results exporting          #
####################################################################################################
####################################################################################################
def _print_impressions_re_ranking_metrics(
    re_ranking_experiment_recommenders: list[commons.ExperimentRecommender],
    baseline_experiment_recommenders: list[commons.ExperimentRecommender],
    baseline_experiment_benchmark: commons.ExperimentBenchmark,
    baseline_experiment_hyper_parameter_tuning_parameters: commons.HyperParameterTuningParameters,
    interaction_data_splits: InteractionsDataSplits,
    num_test_users: int,
    accuracy_metrics_list: list[str],
    beyond_accuracy_metrics_list: list[str],
    all_metrics_list: list[str],
    cutoffs_list: list[int],
    knn_similarity_list: list[commons.T_SIMILARITY_TYPE],
    export_experiments_folder_path: str,
) -> None:
    experiments_folder_path = HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR.format(
        benchmark=baseline_experiment_benchmark.benchmark.value,
        evaluation_strategy=baseline_experiment_hyper_parameter_tuning_parameters.evaluation_strategy.value,
    )

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
                loaded_baseline_recommender = load_trained_recommender(
                    experiment_recommender=baseline_experiment_recommender,
                    experiment_benchmark=baseline_experiment_benchmark,
                    experiment_hyper_parameter_tuning_parameters=baseline_experiment_hyper_parameter_tuning_parameters,
                    data_splits=interaction_data_splits,
                    similarity=similarity,
                    model_type=TrainedRecommenderType.TRAIN_VALIDATION,
                    try_folded_recommender=True,
                )

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

    generate_accuracy_and_beyond_metrics_latex(
        experiments_folder_path=experiments_folder_path,
        export_experiments_folder_path=export_experiments_folder_path,
        num_test_users=num_test_users,
        base_algorithm_list=base_algorithm_list,
        knn_similarity_list=knn_similarity_list,
        other_algorithm_list=[],
        accuracy_metrics_list=accuracy_metrics_list,
        beyond_accuracy_metrics_list=beyond_accuracy_metrics_list,
        all_metrics_list=all_metrics_list,
        cutoffs_list=cutoffs_list,
        icm_names=None
    )


def print_impressions_re_ranking_results(
    baseline_experiment_cases_interface: commons.ExperimentCasesInterface,
    re_ranking_experiment_cases_interface: commons.ExperimentCasesInterface,
) -> None:
    printed_experiments: set[tuple[commons.Benchmarks, commons.EHyperParameterTuningParameters]] = set()

    re_ranking_benchmarks = re_ranking_experiment_cases_interface.to_use_benchmarks
    re_ranking_hyper_parameters = re_ranking_experiment_cases_interface.to_use_hyper_parameter_tuning_parameters
    re_ranking_recommenders = re_ranking_experiment_cases_interface.to_use_recommenders
    re_ranking_experiment_recommenders = [
        commons.MAPPER_AVAILABLE_RECOMMENDERS[rec]
        for rec in re_ranking_recommenders
    ]

    baseline_recommenders = baseline_experiment_cases_interface.to_use_recommenders
    baseline_experiment_recommenders = [
        commons.MAPPER_AVAILABLE_RECOMMENDERS[rec]
        for rec in baseline_recommenders
    ]

    for benchmark, hyper_parameters in itertools.product(re_ranking_benchmarks, re_ranking_hyper_parameters):
        if (benchmark, hyper_parameters) in printed_experiments:
            continue

        printed_experiments.add(
            (benchmark, hyper_parameters)
        )

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
        _print_impressions_re_ranking_metrics(
            re_ranking_experiment_recommenders=re_ranking_experiment_recommenders,
            baseline_experiment_recommenders=baseline_experiment_recommenders,
            baseline_experiment_benchmark=experiment_benchmark,
            baseline_experiment_hyper_parameter_tuning_parameters=experiment_hyper_parameters,
            interaction_data_splits=interaction_data_splits,

            num_test_users=num_test_users,
            accuracy_metrics_list=ACCURACY_METRICS_LIST,
            beyond_accuracy_metrics_list=BEYOND_ACCURACY_METRICS_LIST,
            all_metrics_list=ALL_METRICS_LIST,
            cutoffs_list=RESULT_EXPORT_CUTOFFS,
            knn_similarity_list=experiment_hyper_parameters.knn_similarity_types,
            export_experiments_folder_path=export_experiments_folder_path,
        )

        export_experiments_folder_path = ARTICLE_ACCURACY_METRICS_BASELINES_LATEX_DIR.format(
            benchmark=experiment_benchmark.benchmark.value,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
        )
        _print_impressions_re_ranking_metrics(
            re_ranking_experiment_recommenders=re_ranking_experiment_recommenders,
            baseline_experiment_recommenders=baseline_experiment_recommenders,
            baseline_experiment_benchmark=experiment_benchmark,
            baseline_experiment_hyper_parameter_tuning_parameters=experiment_hyper_parameters,
            interaction_data_splits=interaction_data_splits,

            num_test_users=num_test_users,
            accuracy_metrics_list=ARTICLE_ACCURACY_METRICS_LIST,
            beyond_accuracy_metrics_list=ARTICLE_BEYOND_ACCURACY_METRICS_LIST,
            all_metrics_list=ARTICLE_ALL_METRICS_LIST,
            cutoffs_list=ARTICLE_CUTOFF,
            knn_similarity_list=ARTICLE_KNN_SIMILARITY_LIST,
            export_experiments_folder_path=export_experiments_folder_path,
        )

        logger.info(
            f"Successfully finished exporting accuracy and beyond-accuracy results to LaTeX"
        )
