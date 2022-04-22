import os
from enum import Enum
from typing import Type, Literal, Optional, cast, Union

import attrs
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.BaseRecommender import BaseRecommender
from recsys_framework_extensions.data.io import attach_to_extended_json_decoder
from recsys_framework_extensions.data.mixins import InteractionsDataSplits
from recsys_framework_extensions.recommenders.base import SearchHyperParametersBaseRecommender

from FINNNoReader import FINNNoSlateReader, FinnNoSlatesConfig
from MINDReader import MINDReader, MINDLargeConfig, MINDSmallConfig
from ContentWiseImpressionsReader import ContentWiseImpressionsReader, ContentWiseImpressionsConfig
from recsys_framework_extensions.evaluation import EvaluationStrategy, exclude_from_evaluation
from recsys_framework_extensions.data.reader import DataReader


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


@attach_to_extended_json_decoder
class Benchmarks(Enum):
    MINDSmall = "MINDSmall"
    MINDLarge = "MINDLarge"
    FINNNoSlates = "FINNNoSlates"
    ContentWiseImpressions = "ContentWiseImpressions"


@attach_to_extended_json_decoder
class RecommenderBaseline(Enum):
    RANDOM = "RANDOM"
    TOP_POPULAR = "TOP_POPULAR"
    USER_KNN = "USER_KNN"
    ITEM_KNN = "ITEM_KNN"
    PURE_SVD = "PURE_SVD"
    NMF = "NMF"
    RP3_BETA = "RP3_BETA"
    MF_BPR = "MF_BPR"
    SLIM_ELASTIC_NET = "SLIM_ELASTIC_NET"
    SLIM_BPR = "SLIM_BPR"
    LIGHT_FM = "LIGHT_FM"
    MULT_VAE = "MULT_VAE"
    IALS = "IALS"
    EASE_R = "EASE_R"
    FOLDED = "FOLDED"


@attach_to_extended_json_decoder
class RecommenderImpressions(Enum):
    LAST_IMPRESSIONS = "LAST_IMPRESSIONS"
    FREQUENCY_RECENCY = "FREQUENCY_RECENCY"
    RECENCY = "RECENCY"
    CYCLING = "CYCLING"
    USER_WEIGHTED_USER_PROFILE = "USER_WEIGHTED_USER_PROFILE"
    ITEM_WEIGHTED_USER_PROFILE = "ITEM_WEIGHTED_USER_PROFILE"


DATA_READERS = {
    Benchmarks.MINDSmall: MINDReader,
    Benchmarks.MINDLarge: MINDReader,
    Benchmarks.FINNNoSlates: FINNNoSlateReader,
    Benchmarks.ContentWiseImpressions: ContentWiseImpressionsReader,
}

T_METRIC = Literal["NDCG", "MAP"]
T_SIMILARITY_TYPE = Literal["cosine", "dice", "jaccard", "asymmetric", "tversky"]
T_EVALUATE_ON_TEST = Literal["best", "last"]
T_SAVE_MODEL = Literal["all", "best", "last"]


@attrs.define(frozen=True, kw_only=True)
class HyperParameterTuningParameters:
    evaluation_strategy: EvaluationStrategy = attrs.field(default=EvaluationStrategy.LEAVE_LAST_K_OUT)
    reproducibility_seed: int = attrs.field(default=1234567890)
    max_total_time: int = attrs.field(default=60 * 60 * 24 * 14)
    metric_to_optimize: T_METRIC = attrs.field(default="NDCG")
    cutoff_to_optimize: int = attrs.field(default=10)
    num_cases: int = attrs.field(default=5, validator=[attrs.validators.instance_of(int)])  # 50 <--> 50 / 3
    num_random_starts: int = attrs.field(default=int(2), validator=[attrs.validators.instance_of(int)])
    knn_similarity_types: list[T_SIMILARITY_TYPE] = attrs.field(default=[
        "cosine",
        "dice",
        "jaccard",
        "asymmetric",
        "tversky",
    ])
    resume_from_saved: bool = attrs.field(default=True)
    evaluate_on_test: T_EVALUATE_ON_TEST = attrs.field(default="last")
    evaluation_cutoffs: list[int] = attrs.field(default=[5, 10, 20, 30, 40, 50, 100])
    evaluation_min_ratings_per_user: list[int] = attrs.field(default=1)
    evaluation_exclude_seen: bool = attrs.field(default=True)
    evaluation_percentage_ignore_users: Optional[float] = attrs.field(
        default=None,
        validator=attrs.validators.optional([
            attrs.validators.instance_of(float),
            attrs.validators.ge(0.0),
            attrs.validators.le(1.0)
        ]),
    )
    evaluation_percentage_ignore_items: Optional[float] = attrs.field(
        default=None,
        validator=attrs.validators.optional([
            attrs.validators.instance_of(float),
            attrs.validators.ge(0.0),
            attrs.validators.le(1.0)
        ]),
    )
    save_metadata: bool = attrs.field(default=True, validator=[attrs.validators.instance_of(bool)])
    save_model: T_SAVE_MODEL = attrs.field(default="best")
    terminate_on_memory_error: bool = attrs.field(default=True, validator=[attrs.validators.instance_of(bool)])


@attrs.define(frozen=True, kw_only=True)
class ExperimentBenchmark:
    benchmark: Benchmarks = attrs.field()
    config: object = attrs.field()
    priority: int = attrs.field()
    reader_class: Type[DataReader] = attrs.field(init=False)

    def __attrs_post_init__(self):
        object.__setattr__(self, "reader_class", DATA_READERS[self.benchmark])


@attrs.define(frozen=True, kw_only=True)
class ExperimentRecommender:
    recommender: Type[BaseRecommender] = attrs.field()
    search_hyper_parameters: Type[SearchHyperParametersBaseRecommender] = attrs.field()
    priority: int = attrs.field()


@attrs.define(frozen=True, kw_only=True)
class Experiment:
    hyper_parameter_tuning_parameters: HyperParameterTuningParameters = attrs.field()
    benchmark: ExperimentBenchmark = attrs.field()
    recommenders: list[ExperimentRecommender] = attrs.field()


@attrs.define(frozen=True, kw_only=True)
class Case:
    evaluation_strategy: EvaluationStrategy = attrs.field()
    recommender: Type[BaseRecommender] = attrs.field()
    benchmark: Benchmarks = attrs.field()
    config: object = attrs.field()
    priority: int = attrs.field()
    hyper_parameter_tuning_parameters: HyperParameterTuningParameters = attrs.field()
    reader_class: Type[DataReader] = attrs.field(init=False)

    def __attrs_post_init__(self):
        object.__setattr__(self, "reader_class", DATA_READERS[self.benchmark])


@attrs.define(frozen=True, kw_only=True)
class ExperimentCasesInterface:
    experiments: list[Experiment] = attrs.field()

    @property
    def benchmarks(self) -> list[Benchmarks]:
        return [
            exp.benchmark.benchmark
            for exp in self.experiments
        ]

    @property
    def evaluation_strategies(self) -> list[EvaluationStrategy]:
        return [
            exp.hyper_parameter_tuning_parameters.evaluation_strategy
            for exp in self.experiments
        ]

    # @property
    # def cases(self) -> List[]:
    # def __init__(
    #     self,
    #     experiments: list[Experiment],
    #     # benchmarks: list[Benchmarks],
    #     # priorities_benchmarks: list[int],
    #     # recommenders: list[list[Type[BaseRecommender]]],
    #     # priorities_recommenders: list[list[int]],
    #     # configs: list[object],
    #     # evaluations: list[EvaluationStrategy],
    #     # hyper_parameter_tuning_parameters: list[HyperParameterTuningParameters],
    #
    # ):
    #     assert len(benchmarks) == len(configs)
    #     assert len(benchmarks) == len(evaluations)
    #     assert len(benchmarks) == len(recommenders)
    #     assert len(benchmarks) == len(priorities_benchmarks)
    #     assert len(benchmarks) == len(priorities_recommenders)
    #     assert len(benchmarks) == len(hyper_parameter_tuning_parameters)
    #     assert all(
    #         len(priority_recs) == len(recs)
    #         for priority_recs, recs in zip(priorities_recommenders, recommenders)
    #     )
    #
    #     self.benchmarks = benchmarks
    #     self.configs = configs
    #     self.evaluation_strategies = evaluations
    #     self.recommenders = recommenders
    #     self.priorities_benchmarks = priorities_benchmarks
    #     self.priorities_recommenders = priorities_recommenders
    #     self.hyper_parameter_tuning_parameters = hyper_parameter_tuning_parameters
    #
    # @property  # type: ignore
    # def experiment_cases(self) -> list[ExperimentCase]:
    #     cases = []
    #     for idx_experiment in range(len(self.benchmarks)):
    #
    #         benchmark = self.benchmarks[idx_experiment]
    #         priority_benchmark = self.priorities_benchmarks[idx_experiment]
    #
    #         list_recommenders = self.recommenders[idx_experiment]
    #         list_priorities_recommenders = self.priorities_recommenders[idx_experiment]
    #
    #         config = self.configs[idx_experiment]
    #         evaluation_strategy = self.evaluation_strategies[idx_experiment]
    #         hyper_parameter_tuning_parameters = self.hyper_parameter_tuning_parameters[idx_experiment]
    #
    #         for recommender, priority_recommender in zip(list_recommenders, list_priorities_recommenders):
    #             cases.append(
    #                 ExperimentCase(
    #                     benchmark=benchmark,
    #                     priority=priority_benchmark * priority_recommender,
    #                     config=config,
    #                     evaluation_strategy=evaluation_strategy,
    #                     recommender=recommender,
    #                     hyper_parameter_tuning_parameters=hyper_parameter_tuning_parameters,
    #                 )
    #             )
    #
    #     return cases


RESULTS_EXPERIMENTS_DIR = os.path.join(
    ".",
    "result_experiments",
    ""
)

EVALUATIONS_DIR = os.path.join(
    RESULTS_EXPERIMENTS_DIR,
    "{benchmark}",
    "{evaluation_strategy}",
)

# Each module calls common.FOLDERS.add(<folder_name>) on this variable so they make aware the folder-creator function
# that their folders need to be created.
FOLDERS: set[str] = {
    RESULTS_EXPERIMENTS_DIR,
}


####################################################################################################
####################################################################################################
#             Common Methods.
####################################################################################################
####################################################################################################


# Should be called from main.py
def create_necessary_folders(
    benchmarks: list[Benchmarks],
    evaluation_strategies: list[EvaluationStrategy]
):
    for benchmark, evaluation_strategy in zip(benchmarks, evaluation_strategies):
        for folder in FOLDERS:
            os.makedirs(
                name=folder.format(
                    benchmark=benchmark.value,
                    evaluation_strategy=evaluation_strategy.value,
                ),
                exist_ok=True,
            )


def get_reader_from_benchmark(
    benchmark_config: object,
    benchmark: Benchmarks,
) -> DataReader:
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
    elif Benchmarks.MINDLarge == benchmark:
        benchmark_config = cast(
            MINDLargeConfig,
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
    validation: EvaluatorHoldout = attrs.field()
    validation_early_stopping: EvaluatorHoldout = attrs.field()
    test: EvaluatorHoldout = attrs.field()


def get_evaluators(
    data_splits: InteractionsDataSplits,
    experiment: Experiment,
) -> Evaluators:
    if experiment.hyper_parameter_tuning_parameters.evaluation_percentage_ignore_users is None:
        users_to_exclude_validation = None
    else:
        users_to_exclude_validation = exclude_from_evaluation(
            urm_test=data_splits.sp_urm_test,
            frac_to_exclude=experiment.hyper_parameter_tuning_parameters.evaluation_percentage_ignore_users,
            type_to_exclude="users",
            seed=experiment.hyper_parameter_tuning_parameters.reproducibility_seed,
        )

    if experiment.hyper_parameter_tuning_parameters.evaluation_percentage_ignore_items is None:
        items_to_exclude_validation = None
    else:
        items_to_exclude_validation = exclude_from_evaluation(
            urm_test=data_splits.sp_urm_test,
            frac_to_exclude=experiment.hyper_parameter_tuning_parameters.evaluation_percentage_ignore_items,
            type_to_exclude="items",
            seed=experiment.hyper_parameter_tuning_parameters.reproducibility_seed,
        )

    evaluator_validation = EvaluatorHoldout(
        data_splits.sp_urm_validation,
        cutoff_list=experiment.hyper_parameter_tuning_parameters.evaluation_cutoffs,
        exclude_seen=experiment.hyper_parameter_tuning_parameters.evaluation_exclude_seen,
        min_ratings_per_user=experiment.hyper_parameter_tuning_parameters.evaluation_min_ratings_per_user,
        verbose=True,
        ignore_users=users_to_exclude_validation,
        ignore_items=items_to_exclude_validation,
    )
    evaluator_validation_early_stopping = EvaluatorHoldout(
        data_splits.sp_urm_validation,
        # The example uses the hyper-param benchmark_config instead of the evaluation cutoff.
        cutoff_list=[experiment.hyper_parameter_tuning_parameters.cutoff_to_optimize],
        exclude_seen=experiment.hyper_parameter_tuning_parameters.evaluation_exclude_seen,
        min_ratings_per_user=experiment.hyper_parameter_tuning_parameters.evaluation_min_ratings_per_user,
        verbose=True,
        ignore_users=users_to_exclude_validation,
        ignore_items=items_to_exclude_validation,
    )
    evaluator_test = EvaluatorHoldout(
        data_splits.sp_urm_test,
        cutoff_list=experiment.hyper_parameter_tuning_parameters.evaluation_cutoffs,
        exclude_seen=experiment.hyper_parameter_tuning_parameters.evaluation_exclude_seen,
        min_ratings_per_user=experiment.hyper_parameter_tuning_parameters.evaluation_min_ratings_per_user,
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
    dataset_interface: ExperimentCasesInterface,
) -> None:
    for experiment in dataset_interface.experiments:
        benchmark_reader = get_reader_from_benchmark(
            benchmark_config=experiment.benchmark.config,
            benchmark=experiment.benchmark.benchmark,
        )

        loaded_dataset = benchmark_reader.dataset

        print(f"{loaded_dataset.interactions=}")
        print(f"{loaded_dataset.is_interactions_implicit=}")


def plot_popularity_of_datasets(
    experiments_interface: ExperimentCasesInterface,
) -> None:
    from Utils.plot_popularity import plot_popularity_bias

    for experiment in experiments_interface.experiments:
        dataset_reader = get_reader_from_benchmark(
            benchmark_config=experiment.benchmark.config,
            benchmark=experiment.benchmark.benchmark,
        )
        loaded_dataset = dataset_reader.dataset

        output_folder = EVALUATIONS_DIR.format(
            benchmark=experiment.benchmark.benchmark.value,
            evaluation_strategy=experiment.hyper_parameter_tuning_parameters.evaluation_strategy.value,
        )

        # Interactions plotting.
        urm_splits = loaded_dataset.get_urm_splits(
            evaluation_strategy=experiment.hyper_parameter_tuning_parameters.evaluation_strategy,
        )
        plot_popularity_bias(
            URM_object_list=[
                urm_splits.sp_urm_train,
                urm_splits.sp_urm_validation,
                urm_splits.sp_urm_test
            ],
            URM_name_list=["Train", "Validation", "Test"],
            output_img_path=os.path.join(
                output_folder, "interactions"
            ),
            sort_on_all=False,
        )

        plot_popularity_bias(
            URM_object_list=[
                urm_splits.sp_urm_train,
                urm_splits.sp_urm_validation,
                urm_splits.sp_urm_test
            ],
            URM_name_list=["Train", "Validation", "Test"],
            output_img_path=os.path.join(
                output_folder, "interactions_sorted"
            ),
            sort_on_all=True,
        )

        # Impressions plotting
        uim_splits = loaded_dataset.get_uim_splits(
            evaluation_strategy=experiment.hyper_parameter_tuning_parameters.evaluation_strategy,
        )
        plot_popularity_bias(
            URM_object_list=[
                uim_splits.sp_uim_train,
                uim_splits.sp_uim_validation,
                uim_splits.sp_uim_test
            ],
            URM_name_list=["Train", "Validation", "Test"],
            output_img_path=os.path.join(
                output_folder, "impressions"
            ),
            sort_on_all=False,
        )

        plot_popularity_bias(
            URM_object_list=[
                uim_splits.sp_uim_train,
                uim_splits.sp_uim_validation,
                uim_splits.sp_uim_test
            ],
            URM_name_list=["Train", "Validation", "Test"],
            output_img_path=os.path.join(
                output_folder, "impressions_sorted"
            ),
            sort_on_all=True,
        )


_VALID_COLUMN_ENUMS_BY_FEATURE = {
    ImpressionsFeatures.USER_ITEM_LAST_SEEN: ImpressionsFeatureColumnsLastSeen,
    ImpressionsFeatures.USER_ITEM_POSITION: ImpressionsFeatureColumnsPosition,
    ImpressionsFeatures.USER_ITEM_FREQUENCY: ImpressionsFeatureColumnsFrequency,
    ImpressionsFeatures.USER_ITEM_TIMESTAMP: ImpressionsFeatureColumnsTimestamp,
}


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
    valid_enum = _VALID_COLUMN_ENUMS_BY_FEATURE[impressions_feature]
    if impressions_feature_column not in valid_enum:
        raise ValueError(
            f"Received an invalid column enum {impressions_feature_column} for the feature enum "
            f"{impressions_feature}. Valid column enum for this feature are: "
            f"{_VALID_COLUMN_ENUMS_BY_FEATURE[impressions_feature]}"
        )

    if Benchmarks.ContentWiseImpressions == benchmark:
        if ImpressionsFeatureColumnsLastSeen.EUCLIDEAN == impressions_feature_column:
            raise ValueError(
                f"The benchmark {benchmark} does not compute the impressions feature {impressions_feature} with the "
                f"column {impressions_feature_column}."
            )

        feature_key = (
            f"{evaluation_strategy.value}"
            f"-{impressions_feature.value}"
            f"-{impressions_feature_column.value}"
            f"-{impressions_feature_split.value}"
        )

    else:
        raise NotImplementedError(
            f"Other benchmarks does not have this implemented, yet."
        )

    return feature_key

