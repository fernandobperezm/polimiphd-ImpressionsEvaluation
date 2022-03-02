import os
from enum import Enum
from typing import Type

import attr

from FINNNoReader import FINNNoSlateReader
from MINDReader import MINDReader
from mixins import BaseDataReader, EvaluationStrategy


class Benchmarks(Enum):
    MINDSmall = "MINDSmall"
    MINDLarge = "MINDLarge"
    FINNNoSlates = "FINNNoSlates"
    ContentWiseImpressions = "ContentWiseImpressions"
    XINGRecSysChallenge2016 = "XINGRecSysChallenge2016"


DATA_READERS = {
    Benchmarks.MINDSmall: MINDReader,
    Benchmarks.MINDLarge: MINDReader,
    Benchmarks.FINNNoSlates: FINNNoSlateReader,
}


@attr.s(frozen=True, kw_only=True)
class Dataset:
    benchmark: Benchmarks = attr.ib()
    priority: int = attr.ib()
    reader_class: Type[BaseDataReader] = attr.ib()
    config: object = attr.ib()
    evaluation_strategy: EvaluationStrategy = attr.ib()


class DatasetInterface:
    NAME_URM_TRAIN = "URM_leave_k_out_train"
    NAME_URM_VALIDATION = "URM_leave_k_out_validation"
    NAME_URM_TEST = "URM_leave_k_out_test"

    def __init__(
        self,
        priorities: list[int],
        benchmarks: list[Benchmarks],
        configs: list[object],
        evaluations: list[EvaluationStrategy],
    ):
        assert len(priorities) == len(benchmarks)
        assert len(priorities) == len(configs)
        assert len(priorities) == len(evaluations)

        self.priorities = priorities
        self.benchmarks = benchmarks
        self.configs = configs
        self.evaluation_strategies = evaluations

    @property  # type: ignore
    def datasets(self) -> list[Dataset]:
        return [
            Dataset(
                benchmark=benchmark,
                priority=priority,
                reader_class=DATA_READERS[benchmark],
                config=config,
                evaluation_strategy=evaluation_strategy
            )
            for priority, benchmark, config, evaluation_strategy in zip(
                self.priorities,
                self.benchmarks,
                self.configs,
                self.evaluation_strategies
            )
        ]


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
