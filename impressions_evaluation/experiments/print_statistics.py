import itertools
import os
from typing import Union, Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import logging

import impressions_evaluation.experiments.commons as commons

logger = logging.getLogger(__name__)


####################################################################################################
####################################################################################################
#                                REPRODUCIBILITY VARIABLES                            #
####################################################################################################
####################################################################################################
DIR_RESULTS_DATASETS_STATISTICS = os.path.join(
    commons.DIR_RESULTS_EXPORT,
    "datasets_statistics",
    "{benchmark}",
    "{evaluation_strategy}",
    "",
)

commons.FOLDERS.add(DIR_RESULTS_DATASETS_STATISTICS)


####################################################################################################
####################################################################################################
#             Results exporting          #
####################################################################################################
####################################################################################################
def _extended_describe(
    df: Union[pd.DataFrame, pd.Series],
    stats: list[str],
) -> Union[pd.DataFrame, pd.Series]:
    if isinstance(df, pd.DataFrame):
        d = df.describe(
            percentiles=[0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 0.95, 0.99],
        )
        return d.append(
            df.reindex(d.columns, axis=1).agg(stats),
        )
    else:
        d = df.describe(
            percentiles=[0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 0.95, 0.99],
        )
        return d.append(
            df.reindex().agg(stats),
        )


def _compute_statistics_csr_matrix(
    sp_matrix: sp.csr_matrix,
    sp_matrix_name: str,
    benchmark: commons.Benchmarks,
) -> dict[str, Any]:
    from Data_manager.Dataset import gini_index

    results_dict = {
        "dataset": benchmark.value,
        "matrix_name": sp_matrix_name,
        "nnz": sp_matrix.nnz,
        "num_rows": sp_matrix.shape[0],
        "num_col": sp_matrix.shape[1],
    }

    items = [
        ("data", sp_matrix.data),
        ("user_profile_length", np.ediff1d(sp_matrix.indptr)),
        ("item_profile_length", np.ediff1d(sp_matrix.tocsc().indptr)),
    ]

    for name, data in items:
        data_stats_describe = _extended_describe(
            df=pd.Series(data=data),
            stats=["var", 'skew', 'mad', 'kurt', 'sum']
        ).to_dict()
        results_dict.update({
            f"{name}_{k}": v
            for k,v in data_stats_describe.items()
        })
        results_dict[f"{name}_gini_index"] = gini_index(data)

    return results_dict


def _print_dataset_experiments_statistics(
    experiment_benchmark: commons.ExperimentBenchmark,
    experiment_hyper_parameters: commons.HyperParameterTuningParameters,
    folder_path: str,
) -> None:
    benchmark_reader = commons.get_reader_from_benchmark(
        benchmark_config=experiment_benchmark.config,
        benchmark=experiment_benchmark.benchmark,
    )

    dataset = benchmark_reader.dataset

    interactions_data_splits = dataset.get_urm_splits(
        evaluation_strategy=experiment_hyper_parameters.evaluation_strategy
    )
    impressions_data_splits = dataset.get_uim_splits(
        evaluation_strategy=experiment_hyper_parameters.evaluation_strategy,
    )

    impressions_feature_frequency_train = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_benchmark.benchmark,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_FREQUENCY,
            impressions_feature_column=commons.ImpressionsFeatureColumnsFrequency.FREQUENCY,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
        )
    )
    impressions_feature_frequency_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_benchmark.benchmark,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_FREQUENCY,
            impressions_feature_column=commons.ImpressionsFeatureColumnsFrequency.FREQUENCY,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    impressions_feature_position_train = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_benchmark.benchmark,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_POSITION,
            impressions_feature_column=commons.ImpressionsFeatureColumnsPosition.POSITION,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
        )
    )
    impressions_feature_position_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_benchmark.benchmark,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_POSITION,
            impressions_feature_column=commons.ImpressionsFeatureColumnsPosition.POSITION,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    impressions_feature_timestamp_train = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_benchmark.benchmark,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_TIMESTAMP,
            impressions_feature_column=commons.ImpressionsFeatureColumnsTimestamp.TIMESTAMP,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
        )
    )
    impressions_feature_timestamp_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_benchmark.benchmark,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_TIMESTAMP,
            impressions_feature_column=commons.ImpressionsFeatureColumnsTimestamp.TIMESTAMP,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    if commons.Benchmarks.FINNNoSlates == experiment_benchmark.benchmark:
        impressions_feature_last_seen_train = dataset.sparse_matrix_impression_feature(
            feature=commons.get_feature_key_by_benchmark(
                benchmark=experiment_benchmark.benchmark,
                evaluation_strategy=experiment_hyper_parameters.evaluation_strategy,
                impressions_feature=commons.ImpressionsFeatures.USER_ITEM_LAST_SEEN,
                impressions_feature_column=commons.ImpressionsFeatureColumnsLastSeen.EUCLIDEAN,
                impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
            )
        )
        impressions_feature_last_seen_train_validation = dataset.sparse_matrix_impression_feature(
            feature=commons.get_feature_key_by_benchmark(
                benchmark=experiment_benchmark.benchmark,
                evaluation_strategy=experiment_hyper_parameters.evaluation_strategy,
                impressions_feature=commons.ImpressionsFeatures.USER_ITEM_LAST_SEEN,
                impressions_feature_column=commons.ImpressionsFeatureColumnsLastSeen.EUCLIDEAN,
                impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
            )
        )

    else:
        impressions_feature_last_seen_train = dataset.sparse_matrix_impression_feature(
            feature=commons.get_feature_key_by_benchmark(
                benchmark=experiment_benchmark.benchmark,
                evaluation_strategy=experiment_hyper_parameters.evaluation_strategy,
                impressions_feature=commons.ImpressionsFeatures.USER_ITEM_LAST_SEEN,
                impressions_feature_column=commons.ImpressionsFeatureColumnsLastSeen.TOTAL_DAYS,
                impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
            )
        )
        impressions_feature_last_seen_train_validation = dataset.sparse_matrix_impression_feature(
            feature=commons.get_feature_key_by_benchmark(
                benchmark=experiment_benchmark.benchmark,
                evaluation_strategy=experiment_hyper_parameters.evaluation_strategy,
                impressions_feature=commons.ImpressionsFeatures.USER_ITEM_LAST_SEEN,
                impressions_feature_column=commons.ImpressionsFeatureColumnsLastSeen.TOTAL_DAYS,
                impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
            )
        )

    cases = [
        dict(sp_matrix=interactions_data_splits.sp_urm_train, sp_matrix_name="urm_train",
             benchmark=experiment_benchmark.benchmark),
        dict(sp_matrix=interactions_data_splits.sp_urm_validation, sp_matrix_name="urm_validation",
             benchmark=experiment_benchmark.benchmark),
        dict(sp_matrix=interactions_data_splits.sp_urm_train_validation, sp_matrix_name="urm_train_validation",
             benchmark=experiment_benchmark.benchmark),
        dict(sp_matrix=interactions_data_splits.sp_urm_test, sp_matrix_name="urm_test",
             benchmark=experiment_benchmark.benchmark),

        dict(sp_matrix=impressions_data_splits.sp_uim_train, sp_matrix_name="uim_train",
             benchmark=experiment_benchmark.benchmark),
        dict(sp_matrix=impressions_data_splits.sp_uim_validation, sp_matrix_name="uim_validation",
             benchmark=experiment_benchmark.benchmark),
        dict(sp_matrix=impressions_data_splits.sp_uim_train_validation, sp_matrix_name="uim_train_validation",
             benchmark=experiment_benchmark.benchmark),
        dict(sp_matrix=impressions_data_splits.sp_uim_test, sp_matrix_name="uim_test",
             benchmark=experiment_benchmark.benchmark),

        dict(sp_matrix=impressions_feature_frequency_train, sp_matrix_name="uim_frequency_train",
             benchmark=experiment_benchmark.benchmark),
        dict(sp_matrix=impressions_feature_frequency_train_validation, sp_matrix_name="uim_frequency_train_validation",
             benchmark=experiment_benchmark.benchmark),

        dict(sp_matrix=impressions_feature_position_train, sp_matrix_name="uim_position_train",
             benchmark=experiment_benchmark.benchmark),
        dict(sp_matrix=impressions_feature_position_train_validation, sp_matrix_name="uim_position_train_validation",
             benchmark=experiment_benchmark.benchmark),

        dict(sp_matrix=impressions_feature_timestamp_train, sp_matrix_name="uim_timestamp_train",
             benchmark=experiment_benchmark.benchmark),
        dict(sp_matrix=impressions_feature_timestamp_train_validation, sp_matrix_name="uim_timestamp_train_validation",
             benchmark=experiment_benchmark.benchmark),

        dict(sp_matrix=impressions_feature_last_seen_train, sp_matrix_name="uim_last_seen_train",
             benchmark=experiment_benchmark.benchmark),
        dict(sp_matrix=impressions_feature_last_seen_train_validation, sp_matrix_name="uim_last_seen_train_validation",
             benchmark=experiment_benchmark.benchmark),
    ]
    results = []
    for case in cases:
        results.append(
            _compute_statistics_csr_matrix(**case)
        )

    df_results = pd.DataFrame.from_records(
        data=results,
    )

    with pd.option_context("max_colwidth", 1000):
        df_results.to_csv(
            path_or_buf=os.path.join(folder_path, "dataset_statistics.csv"),
            index=True,
            header=True,
            encoding="utf-8",
            na_rep="-",
        )
        df_results.to_latex(
            buf=os.path.join(folder_path, "dataset_statistics.tex"),
            index=True,
            header=True,
            escape=False,
            float_format="{:.4f}".format,
            encoding="utf-8",
            na_rep="-",
            longtable=True,
        )


def print_datasets_statistics(
    experiment_cases_interface: commons.ExperimentCasesInterface,
) -> None:
    printed_experiments: set[tuple[commons.Benchmarks, commons.EHyperParameterTuningParameters]] = set()

    baseline_benchmarks = experiment_cases_interface.to_use_benchmarks
    baseline_hyper_parameters = experiment_cases_interface.to_use_hyper_parameter_tuning_parameters

    for benchmark, hyper_parameters in itertools.product(baseline_benchmarks, baseline_hyper_parameters):
        if (benchmark, hyper_parameters) in printed_experiments:
            continue
        else:
            printed_experiments.add((benchmark, hyper_parameters))

        experiment_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
            benchmark
        ]
        experiment_hyper_parameters = commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
            hyper_parameters
        ]

        folder_path_export = DIR_RESULTS_DATASETS_STATISTICS.format(
            benchmark=experiment_benchmark.benchmark.value,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
        )

        os.makedirs(folder_path_export, exist_ok=True)

        _print_dataset_experiments_statistics(
            experiment_benchmark=experiment_benchmark,
            experiment_hyper_parameters=experiment_hyper_parameters,
            folder_path=folder_path_export,
        )

        logger.info(
            f"Successfully finished exporting accuracy and beyond-accuracy results to LaTeX"
        )
