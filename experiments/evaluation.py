import os
import uuid
from typing import Type, cast

import Recommenders.Recommender_import_list as recommenders
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.BaseRecommender import BaseRecommender
from recsys_framework_extensions.dask import DaskInterface
from recsys_framework_extensions.evaluation import EvaluationStrategy
from recsys_framework_extensions.hyper_parameter_search import run_hyper_parameter_search_collaborative
from recsys_framework_extensions.logging import get_logger

import experiments.commons as commons
from ContentWiseImpressionsReader import (
    ContentWiseImpressionsReader,
    PandasContentWiseImpressionsProcessData,
    ContentWiseImpressionsConfig,
    ContentWiseImpressionsRawData,
    PandasContentWiseImpressionsRawData,
)
from MINDReader import MINDSmallConfig, MINDReader

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
REPRODUCIBILITY_SEED = 1234567890

RESULT_EXPORT_CUTOFFS = [20]

HYPER_PARAMETER_SEARCH_CONFIG = {
    # Allocate a maximum of 14 days for hyperparameter search, measured in seconds (60 seconds, 60 minutes, 24 hours,
    # 14 days).
    "MAX_TOTAL_TIME": 60 * 60 * 24 * 14,
    "METRIC_TO_OPTIMIZE": "NDCG",
    "CUTOFF_TO_OPTIMIZE": 10,
    "NUM_CASES": 50,
    # 16 if NUM_CASES is 50.
    "NUM_RANDOM_STARTS": int(50 / 3),
    "KNN_SIMILARITY_TYPES": [
        "cosine",
        "dice",
        "jaccard",
        "asymmetric",
        "tversky",
    ],
    "EVALUATE_ON_TEST": "last",
}

EVALUATION_CONFIG = {
    "CUTOFFS": [5, 10, 20, 30, 40, 50, 100],
    "MIN_RATINGS_PER_USER": 1,
    "EXCLUDE_SEEN": True,
}

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
    # recommenders.Random,
    # recommenders.TopPop,
    recommenders.UserKNNCFRecommender,
    recommenders.ItemKNNCFRecommender,
    recommenders.RP3betaRecommender,
    recommenders.PureSVDRecommender,
    recommenders.NMFRecommender,
    recommenders.IALSRecommender,
    recommenders.SLIMElasticNetRecommender,
    #recommenders.SLIM_BPR_Cython,
    recommenders.MatrixFactorization_BPR_Cython,
    recommenders.LightFMCFRecommender,
    recommenders.MultVAERecommender,
    #recommenders.EASE_R_Recommender,
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
#             Reproducibility study: Hyper-parameter tuning of Baselines and CFGAN          #
####################################################################################################
####################################################################################################
def _run_baselines_hyper_parameter_tuning(
    benchmark: commons.Benchmarks,
    benchmark_config: object,
    evaluation_strategy: EvaluationStrategy,
    recommender: Type[BaseRecommender],
) -> None:
    """Run hyper-parameter tuning of baselines on different datasets.

    This method runs hyper parameter tuning of baselines on the original three datasets: Ciao,
    ML100K, and ML1M in their "original" forms, i.e., using the train/test splits reported in the
    original implementation. The baselines are the same as in the original paper of CFGAN plus
    other baselines, such as PureSVD and P3Alpha.
    """
    if commons.Benchmarks.ContentWiseImpressions == benchmark:
        benchmark_config = cast(
            ContentWiseImpressionsConfig,
            benchmark_config,
        )

        raw_reader = ContentWiseImpressionsRawData(
            config=benchmark_config
        )

        pandas_raw_data_reader = PandasContentWiseImpressionsRawData(
            config=benchmark_config,
            raw_data_loader=raw_reader,
        )

        pandas_processed_data_reader = PandasContentWiseImpressionsProcessData(
            config=benchmark_config,
            pandas_raw_data=pandas_raw_data_reader,
        )

        benchmark_reader = ContentWiseImpressionsReader(
            config=benchmark_config,
            processed_data_loader=pandas_processed_data_reader,
        )
    elif commons.Benchmarks.MINDSmall == benchmark:
        benchmark_config = cast(
            MINDSmallConfig,
            benchmark_config,
        )
        benchmark_reader = MINDReader(
            config=benchmark_config,
        )
    else:
        raise ValueError("error fernando-debugger")
    
    dataset = benchmark_reader.dataset

    urm_train, urm_validation, urm_test = dataset.get_urm_splits(
        evaluation_strategy=evaluation_strategy
    )

    urm_train_and_validation = urm_train + urm_validation

    import random
    import numpy as np

    random.seed(REPRODUCIBILITY_SEED)
    np.random.seed(REPRODUCIBILITY_SEED)

    experiments_folder_path = HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR.format(
        benchmark=benchmark.value,
        evaluation_strategy=evaluation_strategy.value,
    )

    evaluator_validation = EvaluatorHoldout(
        urm_validation,
        cutoff_list=EVALUATION_CONFIG["CUTOFFS"],
        exclude_seen=EVALUATION_CONFIG["EXCLUDE_SEEN"],
        min_ratings_per_user=EVALUATION_CONFIG["MIN_RATINGS_PER_USER"],
        verbose=True,
    )
    evaluator_validation_early_stopping = EvaluatorHoldout(
        urm_validation,
        # The example uses the hyper-param config instead of the evaluation cutoff.
        cutoff_list=[HYPER_PARAMETER_SEARCH_CONFIG["CUTOFF_TO_OPTIMIZE"]],
        exclude_seen=EVALUATION_CONFIG["EXCLUDE_SEEN"],
        min_ratings_per_user=EVALUATION_CONFIG["MIN_RATINGS_PER_USER"],
        verbose=True,
    )
    evaluator_test = EvaluatorHoldout(
        urm_test,
        cutoff_list=EVALUATION_CONFIG["CUTOFFS"],
        exclude_seen=EVALUATION_CONFIG["EXCLUDE_SEEN"],
        min_ratings_per_user=EVALUATION_CONFIG["MIN_RATINGS_PER_USER"],
        verbose=True,
    )

    logger_info = {
        "recommender": recommender.RECOMMENDER_NAME,
        "dataset": benchmark.value,
        "urm_test_shape": urm_test.shape,
        "urm_train_shape": urm_train.shape,
        "urm_validation_shape": urm_validation.shape,
        "urm_train_and_validation_shape": urm_train_and_validation.shape,
    }

    logger.info(
        f"Hyper-parameter tuning arguments:"
        f"\n\t* {logger_info}"
    )
    run_hyper_parameter_search_collaborative(
        recommender_class=recommender,

        URM_train=urm_train,
        URM_train_last_test=urm_train_and_validation,

        evaluator_test=evaluator_test,
        evaluator_validation=evaluator_validation,
        evaluator_validation_earlystopping=evaluator_validation_early_stopping,

        output_folder_path=experiments_folder_path,

        n_cases=HYPER_PARAMETER_SEARCH_CONFIG["NUM_CASES"],
        n_random_starts=HYPER_PARAMETER_SEARCH_CONFIG["NUM_RANDOM_STARTS"],
        max_total_time=HYPER_PARAMETER_SEARCH_CONFIG["MAX_TOTAL_TIME"],
        metric_to_optimize=HYPER_PARAMETER_SEARCH_CONFIG["METRIC_TO_OPTIMIZE"],
        cutoff_to_optimize=HYPER_PARAMETER_SEARCH_CONFIG["CUTOFF_TO_OPTIMIZE"],
        similarity_type_list=HYPER_PARAMETER_SEARCH_CONFIG["KNN_SIMILARITY_TYPES"],
        evaluate_on_test=HYPER_PARAMETER_SEARCH_CONFIG["EVALUATE_ON_TEST"],

        allow_weighting=True,
        allow_bias_URM=False,
        allow_dropout_MF=False,
        parallelizeKNN=False,
        resume_from_saved=True,
    )


# def _run_cfgan_with_early_stopping_hyper_parameter_tuning(
#     benchmark: CFGANBenchmarks,
#     urm_train: sp.csr_matrix,
#     urm_validation: sp.csr_matrix,
#     urm_test: sp.csr_matrix,
#     cfgan_mode: CFGANMode,
#     cfgan_mask_type: CFGANMaskType,
# ) -> None:
#     import random
#     import tensorflow as tf
#     import numpy as np
#
#     random.seed(REPRODUCIBILITY_SEED)
#     np.random.seed(REPRODUCIBILITY_SEED)
#     tf.random.set_seed(REPRODUCIBILITY_SEED)
#
#     recommender_class = CFGANRecommenderEarlyStopping
#
#     experiments_folder_path = HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR.format(
#         benchmark=benchmark.value
#     )
#
#     experiment_filename_root = recommender_class.get_recommender_name(
#         cfgan_mode=cfgan_mode,
#         cfgan_mask_type=cfgan_mask_type,
#     )
#
#     evaluator_validation = EvaluatorHoldout(
#         URM_test_list=urm_validation,
#         cutoff_list=VALIDATION_CUTOFFS,
#     )
#
#     evaluator_test = EvaluatorHoldout(
#         URM_test_list=urm_test,
#         cutoff_list=TEST_CUTOFFS,
#     )
#
#     parameter_search = SearchBayesianSkopt(
#         recommender_class=recommender_class,
#         evaluator_validation=evaluator_validation,
#         evaluator_test=evaluator_test
#     )
#
#     early_stopping_kwargs = {
#         "validation_every_n": 5,
#         "stop_on_validation": True,
#         "validation_metric": HYPER_PARAMETER_SEARCH_METRIC_TO_OPTIMIZE,
#         "lower_validations_allowed": 5,
#         "evaluator_object": evaluator_validation,
#         "algorithm_name": recommender_class.RECOMMENDER_NAME,
#     }
#
#     recommender_input_args = SearchInputRecommenderArgs(
#         CONSTRUCTOR_POSITIONAL_ARGS=[],
#         CONSTRUCTOR_KEYWORD_ARGS={
#             "urm_train": urm_train,
#             "num_training_item_weights_to_save": 0,
#         },
#         FIT_POSITIONAL_ARGS=[],
#         FIT_KEYWORD_ARGS={
#             **early_stopping_kwargs
#         }
#     )
#
#     # Evaluation on test set overrides FIT_KEYWORD_ARGS to only be the best hyper-parameters found on validation.
#     # Hence, it's better set FIT_KEYWORD_ARGS as an empty dict. This does not affect the early stopping as when
#     # evaluating on the test set it is not necessary.
#     recommender_input_args_last_test = SearchInputRecommenderArgs(
#         CONSTRUCTOR_POSITIONAL_ARGS=[],
#         CONSTRUCTOR_KEYWORD_ARGS={
#             "urm_train": urm_train + urm_validation,
#             "num_training_item_weights_to_save": 0,
#         },
#         FIT_POSITIONAL_ARGS=[],
#         FIT_KEYWORD_ARGS={}
#     )
#
#     # We test the same lows/highs hyper-parameters as the ones reported in the CFGAN paper.
#     # Our best parameters could differ
#     hyper_parameters = attr.asdict(
#         CFGANEarlyStoppingSearchHyperParameters(
#             benchmark=Categorical([benchmark]),
#             mode=Categorical([cfgan_mode]),
#
#             generator_steps=Integer(
#                 low=1,
#                 high=4,
#                 prior="uniform",
#                 base=10,
#             ),
#             generator_batch_size=Integer(
#                 low=32,
#                 high=256,
#                 prior="uniform",
#                 base=10,
#             ),
#             generator_regularization=Real(  # This one was only 0.0001
#                 low=1e-4,  # 0.0001
#                 high=1e-1,  # 0.1
#                 prior="log-uniform"
#             ),
#             generator_hidden_features=Integer(
#                 low=50,
#                 high=300,
#                 prior="uniform",
#                 base=10,
#             ),
#             generator_learning_rate=Real(
#                 low=1e-4,  # 0.0001,
#                 high=5e-3,  # 0.005,
#                 prior="log-uniform",
#                 base=10,
#             ),
#             generator_hidden_layers=Integer(
#                 low=1,
#                 high=5,
#                 prior="uniform",
#                 base=10,
#             ),
#             generator_optimizer=Categorical([CFGANOptimizer.ADAM]),
#             generator_activation=Categorical([CFGANActivation.SIGMOID]),
#
#             discriminator_steps=Integer(
#                 low=1,
#                 high=4,
#                 prior="uniform",
#                 base=10,
#             ),
#             discriminator_batch_size=Integer(
#                 low=32,
#                 high=256,
#                 prior="uniform",
#                 base=10,
#             ),
#             discriminator_regularization=Real(  # This one was only 0.001
#                 low=1e-4,  # 0.0001
#                 high=1e-1,  # 0.1
#                 prior="log-uniform",
#                 base=10,
#             ),
#             discriminator_hidden_features=Integer(
#                 low=50,
#                 high=300,
#                 prior="uniform",
#                 base=10,
#             ),
#             discriminator_learning_rate=Real(
#                 low=1e-4,  # 0.0001,
#                 high=5e-3,  # 0.005,
#                 prior="log-uniform",
#                 base=10,
#             ),
#             discriminator_hidden_layers=Integer(
#                 low=1,
#                 high=5,
#                 prior="uniform",
#                 base=10,
#             ),
#             discriminator_optimizer=Categorical([CFGANOptimizer.ADAM]),
#             discriminator_activation=Categorical([CFGANActivation.SIGMOID]),
#
#             mask_type=Categorical([cfgan_mask_type]),
#             ratio_zero_reconstruction=Integer(
#                 low=10,
#                 high=90,
#                 prior="uniform",
#                 base=10,
#             ),
#             ratio_partial_masking=Integer(
#                 low=10,
#                 high=90,
#                 prior="uniform",
#                 base=10,
#             ),
#             coefficient_zero_reconstruction=Real(
#                 low=1e-2,  # 0.01,
#                 high=5e-1,  # 0.5,
#                 prior="log-uniform",
#                 base=10,
#             )
#         )
#     )
#
#     logger_info = {
#         "recommender": recommender_class.RECOMMENDER_NAME,
#         "hyper_parameter_space": hyper_parameters,
#         "dataset": benchmark.value,
#         "validation_cutoffs": VALIDATION_CUTOFFS,
#         "test_cutoffs": TEST_CUTOFFS,
#         "urm_train_shape": urm_train.shape,
#         "urm_validation_shape": urm_validation.shape,
#         "urm_test_shape": urm_test.shape,
#         "urm_train_last_test_shape": (urm_train + urm_validation).shape,
#         "output_folder_path": experiments_folder_path,
#         "metric_to_optimize": HYPER_PARAMETER_SEARCH_METRIC_TO_OPTIMIZE,
#         "num_cases": NUM_CASES,
#         "num_random_starts": NUM_RANDOM_STARTS,
#     }
#
#     logger.info(
#         f"Scheduling Hyper-parameter search for {recommender_class.RECOMMENDER_NAME} with "
#         f"search options and hyper-parameters {logger_info}"
#     )
#
#     parameter_search.search(
#         recommender_input_args=recommender_input_args,
#         recommender_input_args_last_test=recommender_input_args_last_test,
#         parameter_search_space=hyper_parameters,
#         n_cases=NUM_CASES,
#         n_random_starts=NUM_RANDOM_STARTS,
#         output_folder_path=experiments_folder_path,
#         output_file_name_root=experiment_filename_root,
#         metric_to_optimize=HYPER_PARAMETER_SEARCH_METRIC_TO_OPTIMIZE,
#         resume_from_saved=True,
#         save_metadata=True,
#         seed=REPRODUCIBILITY_SEED,
#     )


def run_evaluation_experiments(
    include_baselines: bool,
    dask_interface: DaskInterface,
    dataset_interface: commons.DatasetInterface,
) -> None:
    for dataset in dataset_interface.datasets:
        if include_baselines:
            for recommender in ARTICLE_BASELINES:
                dask_interface.submit_job(
                    job_key=(
                        f"_run_baselines_hyper_parameter_tuning"
                        f"|{dataset.benchmark.value}"
                        f"|{recommender.RECOMMENDER_NAME}"
                        f"|{uuid.uuid4()}"
                    ),
                    job_priority=dataset.priority,
                    job_info={
                        "recommender": recommender.RECOMMENDER_NAME,
                        "benchmark": dataset.benchmark.value,
                    },
                    method=_run_baselines_hyper_parameter_tuning,
                    method_kwargs={
                        "benchmark": dataset.benchmark,
                        "benchmark_config": dataset.config,
                        "evaluation_strategy": dataset.evaluation_strategy,
                        "recommender": recommender,
                    }
                )


####################################################################################################
####################################################################################################
#             Evaluation: Dataset Analysis          #
####################################################################################################
####################################################################################################
def plot_popularity_of_datasets(
    dataset_interface: commons.DatasetInterface,
) -> None:
    for dataset in dataset_interface.datasets:

        from Utils.plot_popularity import plot_popularity_bias
        dataset_reader = dataset.reader_class(
            config=dataset.config,
        )
        loaded_dataset = dataset_reader.dataset

        output_folder = BASE_FOLDER.format(
            benchmark=dataset.benchmark.value,
            evaluation_strategy=dataset.evaluation_strategy.value,
        )

        # Interactions plotting.
        urm_train, urm_validation, urm_test = loaded_dataset.get_urm_splits(
            evaluation_strategy=dataset.evaluation_strategy,
        )
        plot_popularity_bias(
            URM_object_list=[urm_train, urm_validation, urm_test],
            URM_name_list=["Train", "Validation", "Test"],
            output_img_path=os.path.join(
                output_folder, "interactions"
            ),
            sort_on_all=False,
        )

        plot_popularity_bias(
            URM_object_list=[urm_train, urm_validation, urm_test],
            URM_name_list=["Train", "Validation", "Test"],
            output_img_path=os.path.join(
                output_folder, "interactions_sorted"
            ),
            sort_on_all=True,
        )

        # Impressions plotting
        uim_train, uim_validation, uim_test = loaded_dataset.get_uim_splits(
            evaluation_strategy=dataset.evaluation_strategy,
        )
        plot_popularity_bias(
            URM_object_list=[uim_train, uim_validation, uim_test],
            URM_name_list=["Train", "Validation", "Test"],
            output_img_path=os.path.join(
                output_folder, "impressions"
            ),
            sort_on_all=False,
        )

        plot_popularity_bias(
            URM_object_list=[uim_train, uim_validation, uim_test],
            URM_name_list=["Train", "Validation", "Test"],
            output_img_path=os.path.join(
                output_folder, "impressions_sorted"
            ),
            sort_on_all=True,
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
#     dataset_interface: commons.DatasetInterface,
# ) -> None:
#     for dataset in dataset_interface.datasets:
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
