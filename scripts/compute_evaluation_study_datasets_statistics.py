#!/usr/bin/env python3
from __future__ import annotations

import logging

from dotenv import load_dotenv
from recsys_framework_extensions.evaluation import EvaluationStrategy
from tap import Tap

from impressions_evaluation import configure_logger
from impressions_evaluation.experiments.print_statistics import (
    print_datasets_statistics,
    print_datasets_statistics_thesis,
)

load_dotenv()

from impressions_evaluation.experiments.commons import (
    create_necessary_folders,
    Benchmarks,
    ensure_datasets_exist,
    EHyperParameterTuningParameters,
    compute_and_plot_popularity_of_datasets,
)


class ConsoleArguments(Tap):
    create_datasets: bool = False
    """If the flag is included, then the script ensures that datasets exists, i.e., it downloads the datasets if possible and then processes the data to create the splits."""

    print_datasets_statistics: bool = False
    """Export to CSV statistics on the different sparse matrices existing for each dataset."""

    plot_datasets_popularity: bool = False
    """Creates plots depicting the popularity of each dataset split."""

    send_email: bool = False
    """Send a notification email via GMAIL when experiments start and finish."""


def main():
    input_flags = ConsoleArguments().parse_args()

    configure_logger()
    logger = logging.getLogger(__name__)

    logger.info(
        "Running script: %(script_name)s with arguments: %(args)s",
        {"script_name": __file__, "args": input_flags.as_dict()},
    )

    to_use_benchmarks = [
        Benchmarks.ContentWiseImpressions,
        Benchmarks.MINDSmall,
        Benchmarks.FINNNoSlates,
    ]

    to_use_hyper_parameter_tuning_parameters = [
        EHyperParameterTuningParameters.LEAVE_LAST_OUT_BAYESIAN_50_16,
    ]

    create_necessary_folders(
        benchmarks=to_use_benchmarks,
        evaluation_strategies=list(EvaluationStrategy),
    )

    if input_flags.create_datasets:
        ensure_datasets_exist(
            to_use_benchmarks=to_use_benchmarks,
        )

    if input_flags.plot_datasets_popularity:
        compute_and_plot_popularity_of_datasets(
            to_use_benchmarks=to_use_benchmarks,
            to_use_hyper_parameters=to_use_hyper_parameter_tuning_parameters,
        )

    if input_flags.print_datasets_statistics:
        print_datasets_statistics(
            to_use_benchmarks=to_use_benchmarks,
            to_use_hyper_parameters=to_use_hyper_parameter_tuning_parameters,
        )
        print_datasets_statistics_thesis(
            to_use_benchmarks=to_use_benchmarks,
            to_use_hyper_parameters=to_use_hyper_parameter_tuning_parameters,
        )

    logger.info(f"Finished running script: {__file__}")


if __name__ == "__main__":
    main()
