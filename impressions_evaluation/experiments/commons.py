import itertools
import os
from enum import Enum
from typing import Type, Literal, Optional, cast, Union, TypeVar, Sequence, Callable

import Recommenders.Recommender_import_list as recommenders
import attrs
import scipy.sparse as sp
from Recommenders.BaseRecommender import BaseRecommender
from recsys_framework_extensions.data.io import attach_to_extended_json_decoder
from recsys_framework_extensions.data.mixins import InteractionsDataSplits
from recsys_framework_extensions.data.reader import DataReader
from recsys_framework_extensions.evaluation import (
    EvaluationStrategy,
    exclude_from_evaluation,
)
from recsys_framework_extensions.evaluation.Evaluator import ExtendedEvaluatorHoldout
from recsys_framework_extensions.recommenders.base import (
    SearchHyperParametersBaseRecommender,
    AbstractExtendedBaseRecommender,
    load_extended_recommender,
    load_recsys_framework_recommender,
)
from typing_extensions import ParamSpec

from impressions_evaluation.readers.ContentWiseImpressionsReader import (
    ContentWiseImpressionsReader,
    ContentWiseImpressionsConfig,
)
from impressions_evaluation.readers.FINNNoReader import (
    FINNNoSlateReader,
    FinnNoSlatesConfig,
)
from impressions_evaluation.readers.MINDReader import MINDReader, MINDSmallConfig
from impressions_evaluation.impression_recommenders.heuristics.frequency_and_recency import (
    FrequencyRecencyRecommender,
    RecencyRecommender,
    SearchHyperParametersFrequencyRecencyRecommender,
    SearchHyperParametersRecencyRecommender,
)
from impressions_evaluation.impression_recommenders.heuristics.latest_impressions import (
    LastImpressionsRecommender,
    SearchHyperParametersLastImpressionsRecommender,
)
from impressions_evaluation.impression_recommenders.re_ranking.cycling import (
    CyclingRecommender,
    SearchHyperParametersCyclingRecommender,
)
from impressions_evaluation.impression_recommenders.re_ranking.impressions_discounting import (
    ImpressionsDiscountingRecommender,
    SearchHyperParametersImpressionsDiscountingRecommender,
)
from impressions_evaluation.impression_recommenders.user_profile.folding import (
    FoldedMatrixFactorizationRecommender,
    SearchHyperParametersFoldedMatrixFactorizationRecommender,
)
from impressions_evaluation.impression_recommenders.user_profile.weighted import (
    UserWeightedUserProfileRecommender,
    ItemWeightedUserProfileRecommender,
    SearchHyperParametersWeightedUserProfileRecommender,
)


class ImpressionsFeatures(Enum):
    USER_ITEM_FREQUENCY = "user_item_frequency"
    USER_ITEM_LAST_SEEN = "user_item_last_seen"
    USER_ITEM_POSITION = "user_item_position"
    USER_ITEM_TIMESTAMP = "user_item_timestamp"


class ImpressionsFeatureColumnsFrequency(Enum):
    FREQUENCY = "feature-user_id-impressions-frequency"


class ImpressionsFeatureColumnsPosition(Enum):
    POSITION = "feature-user_id-impressions-position"


class ImpressionsFeatureColumnsTimestamp(Enum):
    TIMESTAMP = "feature-user_id-impressions-timestamp"


class ImpressionsFeaturesSplit(Enum):
    TRAIN = "train"
    TRAIN_VALIDATION = "train_validation"


class ImpressionsFeatureColumnsLastSeen(Enum):
    EUCLIDEAN = "feature_last_seen_euclidean"
    TOTAL_SECONDS = "feature_last_seen_total_seconds"
    TOTAL_MINUTES = "feature_last_seen_total_minutes"
    TOTAL_HOURS = "feature_last_seen_total_hours"
    TOTAL_DAYS = "feature_last_seen_total_days"
    TOTAL_WEEKS = "feature_last_seen_total_weeks"


class SignalAnalysisType(Enum):
    POSITIVE = "SIGNAL_ANALYSIS_POSITIVE"
    NEGATIVE = "SIGNAL_ANALYSIS_NEGATIVE"


@attach_to_extended_json_decoder
class Benchmarks(Enum):
    MINDSmall = "MINDSmall"
    FINNNoSlates = "FINNNoSlates"
    ContentWiseImpressions = "ContentWiseImpressions"


@attach_to_extended_json_decoder
class RecommenderBaseline(Enum):
    RANDOM = "RANDOM"
    TOP_POPULAR = "TOP_POPULAR"
    GLOBAL_EFFECTS = "GLOBAL_EFFECTS"
    USER_KNN = "USER_KNN"
    ITEM_KNN = "ITEM_KNN"
    ASYMMETRIC_SVD = "ASYMMETRIC_SVD"
    SVDpp = "SVDpp"
    PURE_SVD = "PURE_SVD"
    NMF = "NMF"
    IALS = "IALS"
    MF_BPR = "MF_BPR"
    P3_ALPHA = "P3_ALPHA"
    RP3_BETA = "RP3_BETA"
    SLIM_ELASTIC_NET = "SLIM_ELASTIC_NET"
    SLIM_BPR = "SLIM_BPR"
    LIGHT_FM = "LIGHT_FM"
    MULT_VAE = "MULT_VAE"
    EASE_R = "EASE_R"
    LIGHT_GCN = "LIGHT_GCN"


@attach_to_extended_json_decoder
class RecommenderFolded(Enum):
    FOLDED = "FOLDED"


@attach_to_extended_json_decoder
class RecommenderImpressions(Enum):
    LAST_IMPRESSIONS = "LAST_IMPRESSIONS"
    FREQUENCY_RECENCY = "FREQUENCY_RECENCY"
    RECENCY = "RECENCY"
    CYCLING = "CYCLING"
    IMPRESSIONS_DISCOUNTING = "IMPRESSIONS_DISCOUNTING"
    USER_WEIGHTED_USER_PROFILE = "USER_WEIGHTED_USER_PROFILE"
    ITEM_WEIGHTED_USER_PROFILE = "ITEM_WEIGHTED_USER_PROFILE"
    LIGHT_GCN_ONLY_IMPRESSIONS = "LIGHT_GCN_ONLY_IMPRESSIONS"
    LIGHT_GCN_DIRECTED_INTERACTIONS_IMPRESSIONS = (
        "LIGHT_GCN_DIRECTED_INTERACTIONS_IMPRESSIONS"
    )
    SOFT_FREQUENCY_CAPPING = "SOFT_FREQUENCY_CAPPING"


@attach_to_extended_json_decoder
class EHyperParameterTuningParameters(Enum):
    LEAVE_LAST_OUT_BAYESIAN_50_16 = "LEAVE_LAST_OUT_BAYESIAN_50_16"
    LEAVE_LAST_OUT_BAYESIAN_5_2 = "LEAVE_LAST_OUT_BAYESIAN_5_2"


@attach_to_extended_json_decoder
class TrainedRecommenderType(Enum):
    TRAIN = "TRAIN"
    TRAIN_VALIDATION = "TRAIN_VALIDATION"


T_METRIC = Literal["NDCG", "MAP"]
T_SIMILARITY_TYPE = Literal["cosine", "dice", "jaccard", "asymmetric", "tversky"]
T_EVALUATE_ON_TEST = Literal["best", "last"]
T_SAVE_MODEL = Literal["all", "best", "last"]


@attrs.define(frozen=True, kw_only=True)
class HyperParameterTuningParameters:
    """
    Class that contains all the configuration to run the hyper-parameter tuning of any recommender.
    """

    evaluation_strategy: EvaluationStrategy = attrs.field(
        default=EvaluationStrategy.LEAVE_LAST_K_OUT
    )
    reproducibility_seed: int = attrs.field(default=1234567890)
    max_total_time: int = attrs.field(default=60 * 60 * 24 * 14)
    metric_to_optimize: T_METRIC = attrs.field(default="NDCG")
    cutoff_to_optimize: int = attrs.field(default=10)
    num_cases: int = attrs.field(
        default=50, validator=[attrs.validators.instance_of(int)]
    )
    num_random_starts: int = attrs.field(
        default=16, validator=[attrs.validators.instance_of(int)]
    )
    knn_similarity_types: list[T_SIMILARITY_TYPE] = attrs.field(
        default=[
            "cosine",
            "dice",
            "jaccard",
            "asymmetric",
            "tversky",
        ]
    )
    resume_from_saved: bool = attrs.field(default=True)
    evaluate_on_test: T_EVALUATE_ON_TEST = attrs.field(default="last")
    evaluation_cutoffs: list[int] = attrs.field(default=[5, 10, 20, 30, 40, 50, 100])
    evaluation_min_ratings_per_user: int = attrs.field(default=1)
    evaluation_exclude_seen: bool = attrs.field(default=True)
    evaluation_percentage_ignore_users: Optional[float] = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            [
                attrs.validators.instance_of(float),
                attrs.validators.ge(0.0),
                attrs.validators.le(1.0),
            ]
        ),
    )
    evaluation_percentage_ignore_items: Optional[float] = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            [
                attrs.validators.instance_of(float),
                attrs.validators.ge(0.0),
                attrs.validators.le(1.0),
            ]
        ),
    )
    save_metadata: bool = attrs.field(
        default=True, validator=[attrs.validators.instance_of(bool)]
    )
    save_model: T_SAVE_MODEL = attrs.field(default="best")
    terminate_on_memory_error: bool = attrs.field(
        default=True, validator=[attrs.validators.instance_of(bool)]
    )


@attrs.define(frozen=True, kw_only=True)
class ExperimentBenchmark:
    benchmark: Benchmarks = attrs.field()
    config: object = attrs.field()
    priority: int = attrs.field()
    reader_class: Type[DataReader] = attrs.field(init=False)

    def __attrs_post_init__(self):
        object.__setattr__(
            self, "reader_class", MAPPER_AVAILABLE_DATA_READERS_CLASSES[self.benchmark]
        )


@attrs.define(frozen=True, kw_only=True)
class ExperimentRecommender:
    recommender: Type[
        Union[BaseRecommender, AbstractExtendedBaseRecommender]
    ] = attrs.field()
    search_hyper_parameters: Type[SearchHyperParametersBaseRecommender] = attrs.field()
    priority: int = attrs.field()


@attrs.define(frozen=True, kw_only=True)
class Experiment:
    hyper_parameter_tuning_parameters: HyperParameterTuningParameters = attrs.field()
    benchmark: ExperimentBenchmark = attrs.field()
    recommenders: list[ExperimentRecommender] = attrs.field()


T_RECOMMENDER = TypeVar(
    "T_RECOMMENDER", RecommenderBaseline, RecommenderImpressions, RecommenderFolded
)


@attrs.define(frozen=True, kw_only=True)
class ExperimentCase:
    benchmark: Benchmarks = attrs.field()
    hyper_parameter_tuning_parameters: EHyperParameterTuningParameters = attrs.field()
    recommender: T_RECOMMENDER = attrs.field()  # type: ignore
    training_function: Callable = attrs.field()


@attrs.define(frozen=True, kw_only=True)
class ExperimentCasesInterface:
    to_use_benchmarks: list[Benchmarks] = attrs.field()
    to_use_hyper_parameter_tuning_parameters: list[
        EHyperParameterTuningParameters
    ] = attrs.field()
    to_use_recommenders: list[T_RECOMMENDER] = attrs.field()  # type: ignore
    to_use_training_functions: list[Callable] = attrs.field(default=[])

    @property
    def experiment_cases(self) -> list[ExperimentCase]:
        list_cases = itertools.product(
            self.to_use_benchmarks,
            self.to_use_hyper_parameter_tuning_parameters,
            self.to_use_recommenders,
            self.to_use_training_functions,
        )

        return [
            ExperimentCase(
                benchmark=benchmark,
                hyper_parameter_tuning_parameters=hyper_parameter_tuning_parameters,
                recommender=recommender,
                training_function=training_function,
            )
            for benchmark, hyper_parameter_tuning_parameters, recommender, training_function in list_cases
        ]

    @property
    def benchmarks(self) -> list[Benchmarks]:
        return self.to_use_benchmarks

    @property
    def evaluation_strategies(self) -> list[EvaluationStrategy]:
        return list(EvaluationStrategy)


DIR_TRAINED_MODELS = os.path.join(
    os.getcwd(),
    "trained_models",
    "",
)


DIR_RESULTS_EXPORT = os.path.join(
    os.getcwd(),
    "result_experiments",
    "",
)

DIR_DATASET_POPULARITY = os.path.join(
    DIR_RESULTS_EXPORT,
    "dataset_popularity",
    "{benchmark}",
    "{evaluation_strategy}",
)

# Each module calls common.FOLDERS.add(<folder_name>) on this variable so they make aware the folder-creator function
# that their folders need to be created.
FOLDERS: set[str] = {
    DIR_TRAINED_MODELS,
    DIR_RESULTS_EXPORT,
    DIR_DATASET_POPULARITY,
}


####################################################################################################
####################################################################################################
#             MAPPERS FROM ENUMS TO INSTANCE OF EXPERIMENTS.
####################################################################################################
####################################################################################################
MAPPER_AVAILABLE_IMPRESSION_FEATURES: dict[
    Benchmarks, dict[ImpressionsFeatures, list]
] = {
    Benchmarks.ContentWiseImpressions: {
        ImpressionsFeatures.USER_ITEM_FREQUENCY: [
            ImpressionsFeatureColumnsFrequency.FREQUENCY
        ],
        ImpressionsFeatures.USER_ITEM_LAST_SEEN: [
            ImpressionsFeatureColumnsLastSeen.TOTAL_SECONDS,
            ImpressionsFeatureColumnsLastSeen.TOTAL_MINUTES,
            ImpressionsFeatureColumnsLastSeen.TOTAL_HOURS,
            ImpressionsFeatureColumnsLastSeen.TOTAL_DAYS,
            ImpressionsFeatureColumnsLastSeen.TOTAL_WEEKS,
        ],
        ImpressionsFeatures.USER_ITEM_POSITION: [
            ImpressionsFeatureColumnsPosition.POSITION
        ],
        ImpressionsFeatures.USER_ITEM_TIMESTAMP: [
            ImpressionsFeatureColumnsTimestamp.TIMESTAMP
        ],
    },
    Benchmarks.MINDSmall: {
        ImpressionsFeatures.USER_ITEM_FREQUENCY: [
            ImpressionsFeatureColumnsFrequency.FREQUENCY
        ],
        ImpressionsFeatures.USER_ITEM_LAST_SEEN: [
            ImpressionsFeatureColumnsLastSeen.TOTAL_SECONDS,
            ImpressionsFeatureColumnsLastSeen.TOTAL_MINUTES,
            ImpressionsFeatureColumnsLastSeen.TOTAL_HOURS,
            ImpressionsFeatureColumnsLastSeen.TOTAL_DAYS,
            ImpressionsFeatureColumnsLastSeen.TOTAL_WEEKS,
        ],
        ImpressionsFeatures.USER_ITEM_POSITION: [
            ImpressionsFeatureColumnsPosition.POSITION
        ],
        ImpressionsFeatures.USER_ITEM_TIMESTAMP: [
            ImpressionsFeatureColumnsTimestamp.TIMESTAMP
        ],
    },
    Benchmarks.FINNNoSlates: {
        ImpressionsFeatures.USER_ITEM_FREQUENCY: [
            ImpressionsFeatureColumnsFrequency.FREQUENCY
        ],
        ImpressionsFeatures.USER_ITEM_LAST_SEEN: [
            ImpressionsFeatureColumnsLastSeen.EUCLIDEAN,
        ],
        ImpressionsFeatures.USER_ITEM_POSITION: [
            ImpressionsFeatureColumnsPosition.POSITION
        ],
        ImpressionsFeatures.USER_ITEM_TIMESTAMP: [
            ImpressionsFeatureColumnsTimestamp.TIMESTAMP
        ],
    },
}

MAPPER_AVAILABLE_DATA_READERS_CLASSES = {
    Benchmarks.MINDSmall: MINDReader,
    Benchmarks.FINNNoSlates: FINNNoSlateReader,
    Benchmarks.ContentWiseImpressions: ContentWiseImpressionsReader,
}

MAPPER_AVAILABLE_BENCHMARKS = {
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

MAPPER_AVAILABLE_RECOMMENDERS = {
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
    RecommenderBaseline.GLOBAL_EFFECTS: ExperimentRecommender(
        recommender=recommenders.GlobalEffects,
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
    RecommenderBaseline.ASYMMETRIC_SVD: ExperimentRecommender(
        recommender=recommenders.MatrixFactorization_AsySVD_Cython,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=20,
    ),
    RecommenderBaseline.SVDpp: ExperimentRecommender(
        recommender=recommenders.MatrixFactorization_SVDpp_Cython,
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
    RecommenderBaseline.IALS: ExperimentRecommender(
        recommender=recommenders.IALSRecommender,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=5,
    ),
    RecommenderBaseline.MF_BPR: ExperimentRecommender(
        recommender=recommenders.MatrixFactorization_BPR_Cython,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=10,
    ),
    RecommenderBaseline.P3_ALPHA: ExperimentRecommender(
        recommender=recommenders.P3alphaRecommender,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=10,
    ),
    RecommenderBaseline.RP3_BETA: ExperimentRecommender(
        recommender=recommenders.RP3betaRecommender,
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
        priority=2,
    ),
    RecommenderBaseline.LIGHT_FM: ExperimentRecommender(
        recommender=recommenders.LightFMCFRecommender,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=5,
    ),
    RecommenderBaseline.MULT_VAE: ExperimentRecommender(
        recommender=recommenders.MultVAERecommender,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=2,
    ),
    RecommenderBaseline.EASE_R: ExperimentRecommender(
        recommender=recommenders.EASE_R_Recommender,
        search_hyper_parameters=SearchHyperParametersBaseRecommender,
        priority=1,
    ),
    # IMPRESSIONS_FOLDING
    RecommenderFolded.FOLDED: ExperimentRecommender(
        recommender=FoldedMatrixFactorizationRecommender,
        search_hyper_parameters=SearchHyperParametersFoldedMatrixFactorizationRecommender,
        priority=40,
    ),
    # IMPRESSIONS APPROACHES: HEURISTIC
    RecommenderImpressions.LAST_IMPRESSIONS: ExperimentRecommender(
        recommender=LastImpressionsRecommender,
        search_hyper_parameters=SearchHyperParametersLastImpressionsRecommender,
        priority=50,
    ),
    RecommenderImpressions.FREQUENCY_RECENCY: ExperimentRecommender(
        recommender=FrequencyRecencyRecommender,
        search_hyper_parameters=SearchHyperParametersFrequencyRecencyRecommender,
        priority=50,
    ),
    RecommenderImpressions.RECENCY: ExperimentRecommender(
        recommender=RecencyRecommender,
        search_hyper_parameters=SearchHyperParametersRecencyRecommender,
        priority=50,
    ),
    # IMPRESSIONS APPROACHES: RE RANKING
    RecommenderImpressions.CYCLING: ExperimentRecommender(
        recommender=CyclingRecommender,
        search_hyper_parameters=SearchHyperParametersCyclingRecommender,
        priority=60,
    ),
    RecommenderImpressions.IMPRESSIONS_DISCOUNTING: ExperimentRecommender(
        recommender=ImpressionsDiscountingRecommender,
        search_hyper_parameters=SearchHyperParametersImpressionsDiscountingRecommender,
        priority=70,
    ),
    # IMPRESSIONS APPROACHES: USER PROFILES
    RecommenderImpressions.USER_WEIGHTED_USER_PROFILE: ExperimentRecommender(
        recommender=UserWeightedUserProfileRecommender,
        search_hyper_parameters=SearchHyperParametersWeightedUserProfileRecommender,
        priority=65,
    ),
    RecommenderImpressions.ITEM_WEIGHTED_USER_PROFILE: ExperimentRecommender(
        recommender=ItemWeightedUserProfileRecommender,
        search_hyper_parameters=SearchHyperParametersWeightedUserProfileRecommender,
        priority=70,
    ),
}

MAPPER_ABLATION_AVAILABLE_RECOMMENDERS = {
    RecommenderImpressions.IMPRESSIONS_DISCOUNTING: ExperimentRecommender(
        recommender=ImpressionsDiscountingRecommender,
        search_hyper_parameters=SearchHyperParametersImpressionsDiscountingRecommender,
        priority=80,
    ),
}

MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS = {
    EHyperParameterTuningParameters.LEAVE_LAST_OUT_BAYESIAN_50_16: HyperParameterTuningParameters(
        num_cases=50,
        num_random_starts=16,
    ),
    EHyperParameterTuningParameters.LEAVE_LAST_OUT_BAYESIAN_5_2: HyperParameterTuningParameters(
        num_cases=5,
        num_random_starts=2,
    ),
}


MAPPER_FILE_NAME_POSTFIX = {
    TrainedRecommenderType.TRAIN: "best_model",
    TrainedRecommenderType.TRAIN_VALIDATION: "best_model_last",
}


####################################################################################################
####################################################################################################
#             Trained-Recommenders Methods.
####################################################################################################
####################################################################################################
_RecommenderInstance = TypeVar("_RecommenderInstance", bound=BaseRecommender)

_RecommenderExtendedParams = ParamSpec("_RecommenderExtendedParams")
_RecommenderExtendedInstance = TypeVar(
    "_RecommenderExtendedInstance", bound=AbstractExtendedBaseRecommender
)


def load_recommender_trained_baseline(
    *,
    recommender_baseline_class: Type[_RecommenderInstance],
    folder_path: str,
    file_name_postfix: str,
    urm_train: sp.csr_matrix,
    similarity: Optional[str] = None,
) -> Optional[_RecommenderInstance]:
    recommender_impressions = load_recsys_framework_recommender(
        recommender_class=recommender_baseline_class,
        folder_path=folder_path,
        file_name_postfix=file_name_postfix,
        urm_train=urm_train,
        similarity=similarity,
    )

    return recommender_impressions


def load_recommender_trained_folded(
    *,
    recommender_baseline_instance: _RecommenderInstance,
    folder_path: str,
    file_name_postfix: str,
    urm_train: sp.csr_matrix,
) -> Optional[FoldedMatrixFactorizationRecommender]:
    recommender_impressions = load_extended_recommender(
        recommender_class=FoldedMatrixFactorizationRecommender,
        folder_path=folder_path,
        file_name_postfix=file_name_postfix,
        urm_train=urm_train,
        trained_recommender=recommender_baseline_instance,
    )

    return recommender_impressions


def load_recommender_trained_impressions(
    *,
    recommender_class_impressions: Type[AbstractExtendedBaseRecommender],
    folder_path: str,
    file_name_postfix: str,
    urm_train: sp.csr_matrix,
    uim_train: sp.csr_matrix,
    uim_frequency: sp.csr_matrix,
    uim_position: sp.csr_matrix,
    uim_last_seen: sp.csr_matrix,
    uim_timestamp: sp.csr_matrix,
    recommender_baseline: Union[
        BaseRecommender, FoldedMatrixFactorizationRecommender, None
    ],
) -> Optional[AbstractExtendedBaseRecommender]:
    if recommender_class_impressions == ItemWeightedUserProfileRecommender:
        recommender_impressions = load_extended_recommender(
            recommender_class=ItemWeightedUserProfileRecommender,
            folder_path=folder_path,
            file_name_postfix=file_name_postfix,
            urm_train=urm_train,
            uim_train=uim_train,
            trained_recommender=recommender_baseline,
        )

    elif recommender_class_impressions == UserWeightedUserProfileRecommender:
        recommender_impressions = load_extended_recommender(
            recommender_class=UserWeightedUserProfileRecommender,
            folder_path=folder_path,
            file_name_postfix=file_name_postfix,
            urm_train=urm_train,
            uim_train=uim_train,
            trained_recommender=recommender_baseline,
        )

    elif recommender_class_impressions == ImpressionsDiscountingRecommender:
        recommender_impressions = load_extended_recommender(
            recommender_class=ImpressionsDiscountingRecommender,
            folder_path=folder_path,
            file_name_postfix=file_name_postfix,
            urm_train=urm_train,
            uim_frequency=uim_frequency,
            uim_position=uim_position,
            uim_last_seen=uim_last_seen,
            trained_recommender=recommender_baseline,
        )

    elif recommender_class_impressions == CyclingRecommender:
        recommender_impressions = load_extended_recommender(
            recommender_class=CyclingRecommender,
            folder_path=folder_path,
            file_name_postfix=file_name_postfix,
            urm_train=urm_train,
            uim_frequency=uim_frequency,
            trained_recommender=recommender_baseline,
        )

    elif recommender_class_impressions == FrequencyRecencyRecommender:
        recommender_impressions = load_extended_recommender(
            recommender_class=FrequencyRecencyRecommender,
            folder_path=folder_path,
            file_name_postfix=file_name_postfix,
            urm_train=urm_train,
            uim_frequency=uim_frequency,
            uim_timestamp=uim_timestamp,
        )

    elif recommender_class_impressions == RecencyRecommender:
        recommender_impressions = load_extended_recommender(
            recommender_class=RecencyRecommender,
            folder_path=folder_path,
            file_name_postfix=file_name_postfix,
            urm_train=urm_train,
            uim_timestamp=uim_timestamp,
        )

    elif recommender_class_impressions == LastImpressionsRecommender:
        recommender_impressions = load_extended_recommender(
            recommender_class=LastImpressionsRecommender,
            folder_path=folder_path,
            file_name_postfix=file_name_postfix,
            urm_train=urm_train,
            uim_timestamp=uim_timestamp,
            uim_position=uim_position,
        )

    else:
        raise NotImplementedError("Non-supported impressions recommender.")

    return recommender_impressions


####################################################################################################
####################################################################################################
#             Common Methods.
####################################################################################################
####################################################################################################
def create_necessary_folders(
    benchmarks: list[Benchmarks], evaluation_strategies: list[EvaluationStrategy]
):
    """
    Public method to create the results' folder structure needed by the framework.
    """
    for folder in FOLDERS:
        for benchmark, evaluation_strategy in itertools.product(
            benchmarks, evaluation_strategies
        ):
            formatted = folder.format(
                benchmark=benchmark.value,
                evaluation_strategy=evaluation_strategy.value,
            )
            os.makedirs(
                name=formatted,
                exist_ok=True,
            )


def get_reader_from_benchmark(
    benchmark_config: object,
    benchmark: Benchmarks,
) -> DataReader:
    """
    Returns a `DataReader` instance class that lets the user to load a dataset from disk.
    """
    if Benchmarks.ContentWiseImpressions == benchmark:
        benchmark_config = cast(
            ContentWiseImpressionsConfig,
            benchmark_config,
        )

        benchmark_reader = ContentWiseImpressionsReader(
            config=benchmark_config,
        )
    elif Benchmarks.MINDSmall == benchmark:
        benchmark_config = cast(
            MINDSmallConfig,
            benchmark_config,
        )
        benchmark_reader = MINDReader(
            config=benchmark_config,
        )
    elif Benchmarks.FINNNoSlates == benchmark:
        benchmark_config = cast(
            FinnNoSlatesConfig,
            benchmark_config,
        )
        benchmark_reader = FINNNoSlateReader(
            config=benchmark_config,
        )
    else:
        raise ValueError("error fernando-debugger")

    return benchmark_reader


@attrs.define(frozen=True, kw_only=True, slots=False)
class Evaluators:
    validation: ExtendedEvaluatorHoldout = attrs.field()
    validation_early_stopping: ExtendedEvaluatorHoldout = attrs.field()
    test: ExtendedEvaluatorHoldout = attrs.field()


def get_evaluators(
    data_splits: InteractionsDataSplits,
    experiment_hyper_parameter_tuning_parameters: HyperParameterTuningParameters,
) -> Evaluators:
    """
    Encapsulates the configures of `Evaluators` object so the different methods that do hyper-parameter
    tuning do not have to do it.
    """
    if (
        experiment_hyper_parameter_tuning_parameters.evaluation_percentage_ignore_users
        is None
    ):
        users_to_exclude_validation = None
    else:
        users_to_exclude_validation = exclude_from_evaluation(
            urm_test=data_splits.sp_urm_test,
            frac_to_exclude=experiment_hyper_parameter_tuning_parameters.evaluation_percentage_ignore_users,
            type_to_exclude="users",
            seed=experiment_hyper_parameter_tuning_parameters.reproducibility_seed,
        )

    if (
        experiment_hyper_parameter_tuning_parameters.evaluation_percentage_ignore_items
        is None
    ):
        items_to_exclude_validation = None
    else:
        items_to_exclude_validation = exclude_from_evaluation(
            urm_test=data_splits.sp_urm_test,
            frac_to_exclude=experiment_hyper_parameter_tuning_parameters.evaluation_percentage_ignore_items,
            type_to_exclude="items",
            seed=experiment_hyper_parameter_tuning_parameters.reproducibility_seed,
        )

    evaluator_validation = ExtendedEvaluatorHoldout(
        urm_test=data_splits.sp_urm_validation.copy(),
        urm_train=data_splits.sp_urm_train.copy(),
        cutoff_list=experiment_hyper_parameter_tuning_parameters.evaluation_cutoffs,
        exclude_seen=experiment_hyper_parameter_tuning_parameters.evaluation_exclude_seen,
        min_ratings_per_user=experiment_hyper_parameter_tuning_parameters.evaluation_min_ratings_per_user,
        verbose=True,
        ignore_users=users_to_exclude_validation,
        ignore_items=items_to_exclude_validation,
    )
    evaluator_validation_early_stopping = ExtendedEvaluatorHoldout(
        urm_test=data_splits.sp_urm_validation.copy(),
        urm_train=data_splits.sp_urm_train.copy(),
        # The example uses the hyper-param benchmark_config instead of the evaluation cutoff.
        cutoff_list=[experiment_hyper_parameter_tuning_parameters.cutoff_to_optimize],
        exclude_seen=experiment_hyper_parameter_tuning_parameters.evaluation_exclude_seen,
        min_ratings_per_user=experiment_hyper_parameter_tuning_parameters.evaluation_min_ratings_per_user,
        verbose=True,
        ignore_users=users_to_exclude_validation,
        ignore_items=items_to_exclude_validation,
    )
    evaluator_test = ExtendedEvaluatorHoldout(
        urm_test=data_splits.sp_urm_test.copy(),
        urm_train=data_splits.sp_urm_train_validation.copy(),
        cutoff_list=experiment_hyper_parameter_tuning_parameters.evaluation_cutoffs,
        exclude_seen=experiment_hyper_parameter_tuning_parameters.evaluation_exclude_seen,
        min_ratings_per_user=experiment_hyper_parameter_tuning_parameters.evaluation_min_ratings_per_user,
        verbose=True,
        ignore_users=None,  # Always consider all users in the test set.
        ignore_items=None,  # Always consider all items in the test set.
    )

    return Evaluators(
        validation=evaluator_validation,
        validation_early_stopping=evaluator_validation_early_stopping,
        test=evaluator_test,
    )


def ensure_datasets_exist(
    experiment_cases_interface: ExperimentCasesInterface,
) -> None:
    """
    Public method that will try to load a dataset. If the splits are not created then it will create the
    splits accordingly. The dataset processing parameters are given by the `config` on the benchmark mappers.
    """
    for benchmark in experiment_cases_interface.benchmarks:
        experiment_benchmark = MAPPER_AVAILABLE_BENCHMARKS[benchmark]

        benchmark_reader = get_reader_from_benchmark(
            benchmark_config=experiment_benchmark.config,
            benchmark=benchmark,
        )

        loaded_dataset = benchmark_reader.dataset

        print(f"{experiment_benchmark.config=}")
        print(f"{loaded_dataset.interactions=}")
        print(f"{loaded_dataset.impressions=}")
        print(f"{loaded_dataset.sparse_matrices_available_features()=}")


def plot_popularity_of_datasets(
    experiments_interface: ExperimentCasesInterface,
) -> None:
    """
    Public method that will plot the popularity of data splits. Currently untested.
    """
    from Utils.plot_popularity import plot_popularity_bias

    for experiment_case in experiments_interface.experiment_cases:
        experiment_benchmark = MAPPER_AVAILABLE_BENCHMARKS[experiment_case.benchmark]
        experiment_hyper_parameter_tuning_parameters = (
            MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
                experiment_case.hyper_parameter_tuning_parameters
            ]
        )

        dataset_reader = get_reader_from_benchmark(
            benchmark_config=experiment_benchmark.config,
            benchmark=experiment_benchmark.benchmark,
        )
        loaded_dataset = dataset_reader.dataset

        output_folder = DIR_DATASET_POPULARITY.format(
            benchmark=experiment_benchmark.benchmark.value,
            evaluation_strategy=experiment_hyper_parameter_tuning_parameters.evaluation_strategy.value,
        )

        # Interactions plotting.
        urm_splits = loaded_dataset.get_urm_splits(
            evaluation_strategy=experiment_hyper_parameter_tuning_parameters.evaluation_strategy,
        )
        plot_popularity_bias(
            URM_object_list=[
                urm_splits.sp_urm_train,
                urm_splits.sp_urm_validation,
                urm_splits.sp_urm_test,
            ],
            URM_name_list=["Train", "Validation", "Test"],
            output_img_path=os.path.join(output_folder, "interactions"),
            sort_on_all=False,
        )

        plot_popularity_bias(
            URM_object_list=[
                urm_splits.sp_urm_train,
                urm_splits.sp_urm_validation,
                urm_splits.sp_urm_test,
            ],
            URM_name_list=["Train", "Validation", "Test"],
            output_img_path=os.path.join(output_folder, "interactions_sorted"),
            sort_on_all=True,
        )

        # Impressions plotting
        uim_splits = loaded_dataset.get_uim_splits(
            evaluation_strategy=experiment_hyper_parameter_tuning_parameters.evaluation_strategy,
        )
        plot_popularity_bias(
            URM_object_list=[
                uim_splits.sp_uim_train,
                uim_splits.sp_uim_validation,
                uim_splits.sp_uim_test,
            ],
            URM_name_list=["Train", "Validation", "Test"],
            output_img_path=os.path.join(output_folder, "impressions"),
            sort_on_all=False,
        )

        plot_popularity_bias(
            URM_object_list=[
                uim_splits.sp_uim_train,
                uim_splits.sp_uim_validation,
                uim_splits.sp_uim_test,
            ],
            URM_name_list=["Train", "Validation", "Test"],
            output_img_path=os.path.join(output_folder, "impressions_sorted"),
            sort_on_all=True,
        )


def get_feature_key_by_benchmark(
    benchmark: Benchmarks,
    evaluation_strategy: EvaluationStrategy,
    impressions_feature: ImpressionsFeatures,
    impressions_feature_column: Union[
        ImpressionsFeatureColumnsFrequency,
        ImpressionsFeatureColumnsLastSeen,
        ImpressionsFeatureColumnsPosition,
        ImpressionsFeatureColumnsTimestamp,
    ],
    impressions_feature_split: ImpressionsFeaturesSplit,
) -> str:
    """
    Public method that loads an impressions feature by the dataset, evaluation strategy, the feature name,
    and the feature column if it's the case.
    """
    valid_enum = (
        benchmark in MAPPER_AVAILABLE_IMPRESSION_FEATURES
        and impressions_feature in MAPPER_AVAILABLE_IMPRESSION_FEATURES[benchmark]
        and impressions_feature_column
        in MAPPER_AVAILABLE_IMPRESSION_FEATURES[benchmark][impressions_feature]
    )

    if not valid_enum:
        raise ValueError(
            f"Received an invalid column enum {impressions_feature_column} for the feature enum "
            f"{impressions_feature}. Valid column enum for this feature are: {MAPPER_AVAILABLE_IMPRESSION_FEATURES}"
        )

    feature_key = (
        f"{evaluation_strategy.value}"
        f"-{impressions_feature.value}"
        f"-{impressions_feature_column.value}"
        f"-{impressions_feature_split.value}"
    )

    return feature_key


def get_urm_train_by_trained_recommender_type(
    data_splits: InteractionsDataSplits,
    trained_recommender_type: TrainedRecommenderType,
) -> sp.csr_matrix:
    return (
        data_splits.sp_urm_train.copy()
        if TrainedRecommenderType.TRAIN == trained_recommender_type
        else data_splits.sp_urm_train_validation.copy()
    )


def get_similarities_by_recommender_class(
    recommender_class: Type[BaseRecommender],
    knn_similarities: Sequence[T_SIMILARITY_TYPE],
) -> Sequence[Optional[T_SIMILARITY_TYPE]]:
    if recommender_class in [
        recommenders.ItemKNNCFRecommender,
        recommenders.UserKNNCFRecommender,
    ]:
        return knn_similarities

    return [None]
