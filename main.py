#!/usr/bin/env python3
from __future__ import annotations

import Recommenders.Recommender_import_list as recommenders
from recsys_framework_extensions.dask import configure_dask_cluster
from recsys_framework_extensions.logging import get_logger
from recsys_framework_extensions.recommenders.base import SearchHyperParametersBaseRecommender
from tap import Tap

from ContentWiseImpressionsReader import ContentWiseImpressionsConfig
from FINNNoReader import FinnNoSlatesConfig
from MINDReader import MINDSmallConfig
from experiments.baselines import run_baselines_experiments, run_baselines_folded
from experiments.commons import (
    create_necessary_folders,
    ExperimentCasesInterface,
    Benchmarks,
    HyperParameterTuningParameters,
    Experiment,
    ExperimentBenchmark,
    ExperimentRecommender,
    plot_popularity_of_datasets,
    ensure_datasets_exist, RecommenderBaseline, RecommenderImpressions,
)
from experiments.heuristics import run_impressions_heuristics_experiments
from experiments.re_ranking import run_impressions_re_ranking_experiments
from experiments.user_profiles import run_impressions_user_profiles_experiments
from impression_recommenders.heuristics.frequency_and_recency import FrequencyRecencyRecommender, RecencyRecommender, \
    SearchHyperParametersFrequencyRecencyRecommender, SearchHyperParametersRecencyRecommender
from impression_recommenders.heuristics.latest_impressions import LastImpressionsRecommender, \
    SearchHyperParametersLastImpressionsRecommender
from impression_recommenders.re_ranking.cycling import CyclingRecommender, SearchHyperParametersCyclingRecommender
from impression_recommenders.user_profile.folding import FoldedMatrixFactorizationRecommender, \
    SearchHyperParametersFoldedMatrixFactorizationRecommender
from impression_recommenders.user_profile.weighted import (
    UserWeightedUserProfileRecommender,
    ItemWeightedUserProfileRecommender, SearchHyperParametersWeightedUserProfileRecommender,
)


class ConsoleArguments(Tap):
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

    send_email: bool = False
    """Send a notification email via GMAIL when experiments finish."""


####################################################################################################
####################################################################################################
#                                            MAIN                                                  #
####################################################################################################
####################################################################################################
_AVAILABLE_BENCHMARKS = {
    Benchmarks.ContentWiseImpressions: ExperimentBenchmark(
        benchmark=Benchmarks.ContentWiseImpressions,
        config=ContentWiseImpressionsConfig(),
        priority=10,
    ),

    Benchmarks.MINDSmall: ExperimentBenchmark(
        benchmark=Benchmarks.MINDSmall,
        config=MINDSmallConfig(),
        priority=10,
    ),

    Benchmarks.FINNNoSlates: ExperimentBenchmark(
        benchmark=Benchmarks.FINNNoSlates,
        config=FinnNoSlatesConfig(frac_users_to_keep=0.05),
        priority=10,
    ),
}

_AVAILABLE_RECOMMENDERS = {
    RecommenderBaseline.RANDOM: ExperimentRecommender(
        recommender=recommenders.Random,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=30,
    ),
    RecommenderBaseline.TOP_POPULAR: ExperimentRecommender(
        recommender=recommenders.TopPop,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=30,
    ),
    RecommenderBaseline.USER_KNN: ExperimentRecommender(
        recommender=recommenders.UserKNNCFRecommender,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=20,
    ),
    RecommenderBaseline.ITEM_KNN: ExperimentRecommender(
        recommender=recommenders.ItemKNNCFRecommender,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=20,
    ),
    RecommenderBaseline.PURE_SVD: ExperimentRecommender(
        recommender=recommenders.PureSVDRecommender,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=20,
    ),
    RecommenderBaseline.NMF: ExperimentRecommender(
        recommender=recommenders.NMFRecommender,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=10,
    ),
    RecommenderBaseline.RP3_BETA: ExperimentRecommender(
        recommender=recommenders.RP3betaRecommender,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=10,
    ),
    RecommenderBaseline.MF_BPR: ExperimentRecommender(
        recommender=recommenders.MatrixFactorization_BPR_Cython,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=10,
    ),
    RecommenderBaseline.SLIM_ELASTIC_NET: ExperimentRecommender(
        recommender=recommenders.SLIMElasticNetRecommender,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=5,
    ),
    RecommenderBaseline.SLIM_BPR: ExperimentRecommender(
        recommender=recommenders.SLIM_BPR_Cython,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=4,
    ),
    RecommenderBaseline.LIGHT_FM: ExperimentRecommender(
        recommender=recommenders.LightFMCFRecommender,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=4,
    ),
    RecommenderBaseline.MULT_VAE: ExperimentRecommender(
        recommender=recommenders.MultVAERecommender,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=4,
    ),
    RecommenderBaseline.IALS: ExperimentRecommender(
        recommender=recommenders.IALSRecommender,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=1,
    ),
    RecommenderBaseline.EASE_R: ExperimentRecommender(
        recommender=recommenders.EASE_R_Recommender,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=1,
    ),
    RecommenderBaseline.FOLDED: ExperimentRecommender(
        recommender=FoldedMatrixFactorizationRecommender,
        search_hyper_parameters=SearchHyperParametersFoldedMatrixFactorizationRecommender,
        priority=10,
    ),

    # IMPRESSIONS APPROACHES: HEURISTIC
    RecommenderImpressions.LAST_IMPRESSIONS: ExperimentRecommender(
        recommender=LastImpressionsRecommender,
        search_hyper_parameters=SearchHyperParametersLastImpressionsRecommender,
        priority=10,
    ),
    RecommenderImpressions.FREQUENCY_RECENCY: ExperimentRecommender(
        recommender=FrequencyRecencyRecommender,
        search_hyper_parameters=SearchHyperParametersFrequencyRecencyRecommender,
        priority=10,
    ),
    RecommenderImpressions.RECENCY: ExperimentRecommender(
        recommender=RecencyRecommender,
        search_hyper_parameters=SearchHyperParametersRecencyRecommender,
        priority=10,
    ),

    # IMPRESSIONS APPROACHES: RE RANKING
    RecommenderImpressions.CYCLING: ExperimentRecommender(
        recommender=CyclingRecommender,
        search_hyper_parameters=SearchHyperParametersCyclingRecommender,
        priority=10,
    ),

    # IMPRESSIONS APPROACHES: USER PROFILES
    RecommenderImpressions.USER_WEIGHTED_USER_PROFILE: ExperimentRecommender(
        recommender=UserWeightedUserProfileRecommender,
        search_hyper_parameters=SearchHyperParametersWeightedUserProfileRecommender,
        priority=10,
    ),
    RecommenderImpressions.ITEM_WEIGHTED_USER_PROFILE: ExperimentRecommender(
        recommender=ItemWeightedUserProfileRecommender,
        search_hyper_parameters=SearchHyperParametersWeightedUserProfileRecommender,
        priority=10,
    ),
}

_TO_USE_BENCHMARKS = [
    Benchmarks.ContentWiseImpressions,
    Benchmarks.MINDSmall,
    Benchmarks.FINNNoSlates,
]

_TO_USE_RECOMMENDERS_BASELINE = list(RecommenderBaseline)

_TO_USE_RECOMMENDERS_IMPRESSIONS_HEURISTICS = [
    RecommenderImpressions.LAST_IMPRESSIONS,
    RecommenderImpressions.FREQUENCY_RECENCY,
    RecommenderImpressions.RECENCY,
]

_TO_USE_RECOMMENDERS_IMPRESSIONS_RE_RANKING = [
    RecommenderImpressions.CYCLING,
]

_TO_USE_RECOMMENDERS_IMPRESSIONS_USER_PROFILES = [
    RecommenderImpressions.USER_WEIGHTED_USER_PROFILE,
    RecommenderImpressions.ITEM_WEIGHTED_USER_PROFILE,
]


if __name__ == '__main__':
    input_flags = ConsoleArguments().parse_args()

    logger = get_logger(__name__)

    dask_interface = configure_dask_cluster()

    common_hyper_parameter_tuning_parameters = HyperParameterTuningParameters()

    # Training statistics.
    # CW - UserKNN - 3 GB Training - 250 sec/it
    # CW - ItemKNN - 3 GB Training - 320 sec/it
    # CW - PureSVD - 2 GB Training - 170 sec/it
    # CW - EASE-R - 28 GB Training - 400 sec/it
    # MINDSmall - EASE_R - 16GB Training - 80sec/it
    # MINDLarge - EASE_R - 29.3GB Training - 450sec/it
    # FINNNoSlates - EASE R - 12.4TB Training - No Training.
    experiments_interface_baselines = ExperimentCasesInterface(
        experiments=[
            Experiment(
                hyper_parameter_tuning_parameters=common_hyper_parameter_tuning_parameters,
                benchmark=_AVAILABLE_BENCHMARKS[benchmark],
                recommenders=[
                    _AVAILABLE_RECOMMENDERS[recommender]
                    for recommender in _TO_USE_RECOMMENDERS_BASELINE
                ],
            )
            for benchmark in _TO_USE_BENCHMARKS
        ],
    )

    experiments_impressions_heuristics_interface = ExperimentCasesInterface(
        experiments=[
            Experiment(
                hyper_parameter_tuning_parameters=common_hyper_parameter_tuning_parameters,
                benchmark=_AVAILABLE_BENCHMARKS[benchmark],
                recommenders=[
                    _AVAILABLE_RECOMMENDERS[recommender]
                    for recommender in _TO_USE_RECOMMENDERS_IMPRESSIONS_HEURISTICS
                ],
            )
            for benchmark in _TO_USE_BENCHMARKS
        ],
    )

    experiments_impressions_re_ranking_interface = ExperimentCasesInterface(
        experiments=[
            Experiment(
                hyper_parameter_tuning_parameters=common_hyper_parameter_tuning_parameters,
                benchmark=_AVAILABLE_BENCHMARKS[benchmark],
                recommenders=[
                    _AVAILABLE_RECOMMENDERS[recommender]
                    for recommender in _TO_USE_RECOMMENDERS_IMPRESSIONS_RE_RANKING
                ],
            )
            for benchmark in _TO_USE_BENCHMARKS
        ],
    )

    experiments_impressions_user_profiles_interface = ExperimentCasesInterface(
        experiments=[
            Experiment(
                hyper_parameter_tuning_parameters=common_hyper_parameter_tuning_parameters,
                benchmark=_AVAILABLE_BENCHMARKS[benchmark],
                recommenders=[
                    _AVAILABLE_RECOMMENDERS[recommender]
                    for recommender in _TO_USE_RECOMMENDERS_IMPRESSIONS_RE_RANKING
                ],
            )
            for benchmark in _TO_USE_BENCHMARKS
        ],
    )

    create_necessary_folders(
        benchmarks=experiments_interface_baselines.benchmarks,
        evaluation_strategies=experiments_interface_baselines.evaluation_strategies,
    )

    if input_flags.create_datasets:
        ensure_datasets_exist(
            dataset_interface=experiments_interface_baselines,
        )

    if input_flags.include_baselines:
        run_baselines_experiments(
            dask_interface=dask_interface,
            experiment_cases_interface=experiments_interface_baselines,
        )

    if input_flags.include_folded:
        run_baselines_folded(
            dask_interface=dask_interface,
            recommender_folded=_AVAILABLE_RECOMMENDERS[RecommenderBaseline.FOLDED],
            experiment_cases_interface=experiments_interface_baselines,
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
            baseline_experiment_cases_interface=experiments_interface_baselines,
        )

    if input_flags.include_impressions_profile:
        run_impressions_user_profiles_experiments(
            dask_interface=dask_interface,
            user_profiles_experiment_cases_interface=experiments_impressions_user_profiles_interface,
            baseline_experiment_cases_interface=experiments_interface_baselines,
        )

    dask_interface.wait_for_jobs()

    if input_flags.plot_popularity_of_datasets:
        plot_popularity_of_datasets(
            experiments_interface=experiments_interface_baselines,
        )

    # if input_flags.print_evaluation_results:
    #     print_reproducibility_results(
    #         experiments_interface=experiments_interface,
    #     )

    if input_flags.send_email:
        from recsys_framework_extensions.data.io import ExtendedJSONEncoderDecoder
        from recsys_framework_extensions.email.gmail import GmailEmailNotifier
        import json

        GmailEmailNotifier.send_email(
            subject="[Impressions Datasets] Execution finished",
            body=f"""An execution with the following properties just finished:
            \n\t* Input Flags: {json.dumps(input_flags.as_dict(), default=ExtendedJSONEncoderDecoder.to_json)} 
            \n\t* Benchmarks: {json.dumps(_TO_USE_BENCHMARKS, default=ExtendedJSONEncoderDecoder.to_json)} 
            \n\t* Baselines: {json.dumps(_TO_USE_RECOMMENDERS_BASELINE, default=ExtendedJSONEncoderDecoder.to_json)} 
            \n\t* Impressions Heuristics: {json.dumps(_TO_USE_RECOMMENDERS_IMPRESSIONS_HEURISTICS, default=ExtendedJSONEncoderDecoder.to_json)} 
            \n\t* Impressions Re Ranking: {json.dumps(_TO_USE_RECOMMENDERS_IMPRESSIONS_RE_RANKING, default=ExtendedJSONEncoderDecoder.to_json)} 
            \n\t* Impressions User Profiles: {json.dumps(_TO_USE_RECOMMENDERS_IMPRESSIONS_USER_PROFILES, default=ExtendedJSONEncoderDecoder.to_json)} 
            """,
            sender="mistermaurera@gmail.com",
            receivers=[
                "fperezmaurera@gmail.com",
                "fernandobenjamin.perez@polimi.it",
            ],
        )

    dask_interface.close()
