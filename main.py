#!/usr/bin/env python3
from __future__ import annotations

from recsys_framework_extensions.recommenders.base import SearchHyperParametersBaseRecommender
from tap import Tap

from experiments.heuristics import run_impressions_heuristics_experiments
from experiments.re_ranking import run_impressions_re_ranking_experiments
from experiments.user_profiles import run_impressions_user_profiles_experiments
from impression_recommenders.heuristics.latest_impressions import LastImpressionsRecommender, \
    SearchHyperParametersLastImpressionsRecommender
from impression_recommenders.heuristics.frequency_and_recency import FrequencyRecencyRecommender, RecencyRecommender, \
    SearchHyperParametersFrequencyRecencyRecommender, SearchHyperParametersRecencyRecommender
from impression_recommenders.re_ranking.cycling import CyclingRecommender, SearchHyperParametersCyclingRecommender
from impression_recommenders.re_ranking.dithering import DitheringRecommender, SearchHyperParametersDitheringRecommender
from impression_recommenders.user_profile.folding import FoldedMatrixFactorizationRecommender, \
    SearchHyperParametersFoldedMatrixFactorizationRecommender
from impression_recommenders.user_profile.weighted import (
    UserWeightedUserProfileRecommender,
    ItemWeightedUserProfileRecommender, SearchHyperParametersWeightedUserProfileRecommender,
)

from FINNNoReader import FinnNoSlatesConfig
from MINDReader import MINDSmallConfig, MINDLargeConfig
from ContentWiseImpressionsReader import ContentWiseImpressionsConfig
from experiments.commons import (
    create_necessary_folders,
    ExperimentCasesInterface,
    Benchmarks,
    HyperParameterTuningParameters,
    Experiment,
    ExperimentBenchmark,
    ExperimentRecommender,
    plot_popularity_of_datasets,
    ensure_datasets_exist,
)
from experiments.baselines import run_baselines_experiments, run_baselines_folded
from recsys_framework_extensions.dask import configure_dask_cluster
from recsys_framework_extensions.logging import get_logger
import Recommenders.Recommender_import_list as recommenders


class ConsoleArguments(Tap):
    run_evaluation: bool = False
    """Run Hyper-parameter tuning of recommenders on the Ciao, ML100K, and ML1M datasets. Which recommenders are 
    tuned depend on the presence of the options --include_baselines and --include_cfgan.
    """

    create_datasets: bool = False
    """TODO: fernando-debugger."""

    include_baselines: bool = False
    """Include baselines in the hyper-parameter tuning"""

    include_folded: bool = False
    """Include baselines in the hyper-parameter tuning"""

    include_impressions_heuristics: bool = False
    """Include baselines in the hyper-parameter tuning"""

    include_impressions_reranking: bool = False
    """Include baselines in the hyper-parameter tuning"""

    include_impressions_profile: bool = False
    """Include baselines in the hyper-parameter tuning"""

    print_evaluation_results: bool = False
    """Print LaTeX tables containing the accuracy and beyond accuracy metrics of the hyper-parameter tuned 
    recommenders."""

    plot_popularity_of_datasets: bool = False
    """Creates plots depicting the popularity of each dataset split."""


####################################################################################################
####################################################################################################
#                                            MAIN                                                  #
####################################################################################################
####################################################################################################
if __name__ == '__main__':
    input_flags = ConsoleArguments().parse_args()

    logger = get_logger(__name__)

    dask_interface = configure_dask_cluster()

    common_hyper_parameter_tuning_parameters = HyperParameterTuningParameters()

    benchmark_finn_no = ExperimentBenchmark(
        benchmark=Benchmarks.FINNNoSlates,
        config=FinnNoSlatesConfig(),
        priority=10,
    )

    benchmark_mind_small = ExperimentBenchmark(
        benchmark=Benchmarks.MINDSmall,
        config=MINDSmallConfig(),
        priority=10,
    )

    benchmark_cw = ExperimentBenchmark(
        benchmark=Benchmarks.ContentWiseImpressions,
        config=ContentWiseImpressionsConfig(),
        priority=10,
    )

    recommender_random = ExperimentRecommender(
        recommender=recommenders.Random,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=30,
    )
    recommender_top_pop = ExperimentRecommender(
        recommender=recommenders.TopPop,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=30,
    )
    recommender_user_knn = ExperimentRecommender(
        recommender=recommenders.UserKNNCFRecommender,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=20,
    )
    recommender_item_knn = ExperimentRecommender(
        recommender=recommenders.ItemKNNCFRecommender,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=20,
    )
    recommender_pure_svd = ExperimentRecommender(
        recommender=recommenders.PureSVDRecommender,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=20,
    )
    recommender_nmf = ExperimentRecommender(
        recommender=recommenders.NMFRecommender,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=10,
    )
    recommender_rp3beta = ExperimentRecommender(
        recommender=recommenders.RP3betaRecommender,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=10,
    )
    recommender_mf_bpr = ExperimentRecommender(
        recommender=recommenders.MatrixFactorization_BPR_Cython,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=10,
    )
    recommender_slim_elasticnet = ExperimentRecommender(
        recommender=recommenders.SLIMElasticNetRecommender,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=5,
    )
    recommender_slim_bpr = ExperimentRecommender(
        recommender=recommenders.SLIM_BPR_Cython,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=4,
    )
    recommender_light_fm = ExperimentRecommender(
        recommender=recommenders.LightFMCFRecommender,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=4,
    )
    recommender_mult_vae = ExperimentRecommender(
        recommender=recommenders.MultVAERecommender,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=4,
    )
    recommender_ials = ExperimentRecommender(
        recommender=recommenders.IALSRecommender,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=1,
    )
    recommender_ease_r = ExperimentRecommender(
        recommender=recommenders.EASE_R_Recommender,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=1,
    )

    # Folded Recommender
    recommender_folded = ExperimentRecommender(
        recommender=FoldedMatrixFactorizationRecommender,
        search_hyper_parameters=SearchHyperParametersFoldedMatrixFactorizationRecommender,
        priority=10,
    )

    # Impression Recommenders
    recommender_impressions_last_impressions = ExperimentRecommender(
        recommender=LastImpressionsRecommender,
        search_hyper_parameters=SearchHyperParametersLastImpressionsRecommender,
        priority=10,
    )
    recommender_impressions_frequency_recency = ExperimentRecommender(
        recommender=FrequencyRecencyRecommender,
        search_hyper_parameters=SearchHyperParametersFrequencyRecencyRecommender,
        priority=10,
    )
    recommender_impressions_recency = ExperimentRecommender(
        recommender=RecencyRecommender,
        search_hyper_parameters=SearchHyperParametersRecencyRecommender,
        priority=10,
    )
    recommender_impressions_cycling = ExperimentRecommender(
        recommender=CyclingRecommender,
        search_hyper_parameters=SearchHyperParametersCyclingRecommender,
        priority=10,
    )
    recommender_impressions_dithering = ExperimentRecommender(
        recommender=DitheringRecommender,
        search_hyper_parameters=SearchHyperParametersDitheringRecommender,
        priority=10,
    )
    recommender_impressions_user_weighted = ExperimentRecommender(
        recommender=UserWeightedUserProfileRecommender,
        search_hyper_parameters=SearchHyperParametersWeightedUserProfileRecommender,
        priority=10,
    )
    recommender_impressions_item_weighted = ExperimentRecommender(
        recommender=ItemWeightedUserProfileRecommender,
        search_hyper_parameters=SearchHyperParametersWeightedUserProfileRecommender,
        priority=10,
    )

    experiments_baselines = [
        Experiment(
            hyper_parameter_tuning_parameters=common_hyper_parameter_tuning_parameters,
            benchmark=benchmark_cw,
            recommenders=[
                recommender_random,
                recommender_top_pop,
                #recommender_user_knn,
                #recommender_item_knn,
                recommender_pure_svd,
                #recommender_nmf,
                # recommender_rp3beta,
                # recommender_mf_bpr,
                # recommender_slim_elasticnet,
                # recommender_slim_bpr,
                # recommender_light_fm,
                # recommender_mult_vae,
                # recommender_ials,
                # recommender_ease_r,
            ],
        ),
        # Experiment(
        #     hyper_parameter_tuning_parameters=common_hyper_parameter_tuning_parameters,
        #     benchmark=benchmark_mind_small,
        #     recommenders=[
        #         recommender_random,
        #         recommender_top_pop,
        #         recommender_user_knn,
        #         recommender_item_knn,
        #         recommender_pure_svd,
        #         recommender_nmf,
        #         recommender_rp3beta,
        #         recommender_mf_bpr,
        #         recommender_slim_elasticnet,
        #         recommender_slim_bpr,
        #         recommender_light_fm,
        #         recommender_mult_vae,
        #         recommender_ials,
        #         recommender_ease_r,
        #     ],
        # ),
        # Experiment(
        #     hyper_parameter_tuning_parameters=common_hyper_parameter_tuning_parameters,
        #     benchmark=benchmark_finn_no,
        #     recommenders=[
        #         recommender_random,
        #         recommender_top_pop,
        #         recommender_user_knn,
        #         recommender_item_knn,
        #         recommender_pure_svd,
        #         recommender_nmf,
        #         recommender_rp3beta,
        #         recommender_mf_bpr,
        #         recommender_slim_elasticnet,
        #         recommender_slim_bpr,
        #         recommender_light_fm,
        #         recommender_mult_vae,
        #         recommender_ials,
        #         recommender_ease_r,
        #     ],
        # ),
    ]

    experiments_impressions_heuristics = [
        Experiment(
            hyper_parameter_tuning_parameters=common_hyper_parameter_tuning_parameters,
            benchmark=benchmark_cw,
            recommenders=[
                recommender_impressions_last_impressions,
                recommender_impressions_frequency_recency,
                recommender_impressions_recency,
            ],
        ),
        # Experiment(
        #     hyper_parameter_tuning_parameters=common_hyper_parameter_tuning_parameters,
        #     benchmark=benchmark_mind_small,
        #     recommenders=[
        #         recommender_impressions_last_impressions,
        #         recommender_impressions_frequency_recency,
        #         recommender_impressions_recency,
        #     ],
        # ),
        # Experiment(
        #     hyper_parameter_tuning_parameters=common_hyper_parameter_tuning_parameters,
        #     benchmark=benchmark_finn_no,
        #     recommenders=[
        #         recommender_impressions_last_impressions,
        #         recommender_impressions_frequency_recency,
        #         recommender_impressions_recency,
        #     ],
        # ),
    ]

    experiments_impressions_re_ranking = [
        Experiment(
            hyper_parameter_tuning_parameters=common_hyper_parameter_tuning_parameters,
            benchmark=benchmark_cw,
            recommenders=[
                recommender_impressions_cycling,
            ],
        ),
        # Experiment(
        #     hyper_parameter_tuning_parameters=common_hyper_parameter_tuning_parameters,
        #     benchmark=benchmark_mind_small,
        #     recommenders=[
        #         recommender_impressions_cycling,
        #         recommender_impressions_dithering,
        #     ],
        # ),
        # Experiment(
        #     hyper_parameter_tuning_parameters=common_hyper_parameter_tuning_parameters,
        #     benchmark=benchmark_finn_no,
        #     recommenders=[
        #         recommender_impressions_cycling,
        #         recommender_impressions_dithering,
        #     ],
        # ),
    ]

    experiments_impressions_user_profiles = [
        Experiment(
            hyper_parameter_tuning_parameters=common_hyper_parameter_tuning_parameters,
            benchmark=benchmark_cw,
            recommenders=[
                recommender_impressions_user_weighted,
                recommender_impressions_item_weighted,
            ],
        ),
        # Experiment(
        #     hyper_parameter_tuning_parameters=common_hyper_parameter_tuning_parameters,
        #     benchmark=benchmark_mind_small,
        #     recommenders=[
        #         recommender_impressions_user_weighted,
        #         recommender_impressions_item_weighted,
        #     ],
        # ),
        # Experiment(
        #     hyper_parameter_tuning_parameters=common_hyper_parameter_tuning_parameters,
        #     benchmark=benchmark_finn_no,
        #     recommenders=[
        #         recommender_impressions_user_weighted,
        #         recommender_impressions_item_weighted,
        #     ],
        # ),
    ]

    # Training statistics.
    # CW - UserKNN - 3 GB Training - 250 sec/it
    # CW - ItemKNN - 3 GB Training - 320 sec/it
    # CW - PureSVD - 2 GB Training - 170 sec/it
    # CW - EASE-R - 28 GB Training - 400 sec/it
    # MINDSmall - EASE_R - 16GB Training - 80sec/it
    # MINDLarge - EASE_R - 29.3GB Training - 450sec/it
    # FINNNoSlates - EASE R - 12.4TB Training - No Training.
    experiments_interface = ExperimentCasesInterface(
        experiments=experiments_baselines
    )

    experiments_impressions_heuristics_interface = ExperimentCasesInterface(
        experiments=experiments_impressions_heuristics,
    )

    experiments_impressions_re_ranking_interface = ExperimentCasesInterface(
        experiments=experiments_impressions_re_ranking,
    )

    experiments_impressions_user_profiles_interface = ExperimentCasesInterface(
        experiments=experiments_impressions_user_profiles,
    )

    create_necessary_folders(
        benchmarks=experiments_interface.benchmarks,
        evaluation_strategies=experiments_interface.evaluation_strategies,
    )

    if input_flags.create_datasets:
        ensure_datasets_exist(
            dataset_interface=experiments_interface,
        )

    if input_flags.include_baselines:
        run_baselines_experiments(
            dask_interface=dask_interface,
            experiment_cases_interface=experiments_interface,
        )

    if input_flags.include_folded:
        run_baselines_folded(
            dask_interface=dask_interface,
            recommender_folded=recommender_folded,
            experiment_cases_interface=experiments_interface,
        )

    if input_flags.include_impressions_heuristics:
        run_impressions_heuristics_experiments(
            dask_interface=dask_interface,
            experiment_cases_interface=experiments_impressions_heuristics_interface,
        )

    if input_flags.include_impressions_reranking:
        run_impressions_re_ranking_experiments(
            dask_interface=dask_interface,
            re_ranking_experiment_cases_interface=experiments_impressions_re_ranking_interface,
            baseline_experiment_cases_interface=experiments_interface,
        )

    if input_flags.include_impressions_profile:
        run_impressions_user_profiles_experiments(
            dask_interface=dask_interface,
            user_profiles_experiment_cases_interface=experiments_impressions_user_profiles_interface,
            baseline_experiment_cases_interface=experiments_interface,
        )

    dask_interface.wait_for_jobs()

    if input_flags.plot_popularity_of_datasets:
        plot_popularity_of_datasets(
            experiments_interface=experiments_interface,
        )

    # if input_flags.print_evaluation_results:
    #     print_reproducibility_results(
    #         experiments_interface=experiments_interface,
    #     )

    dask_interface.close()
