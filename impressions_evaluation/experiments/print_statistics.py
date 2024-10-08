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
    "",
)

DIR_RESULTS_DATASETS_STATISTICS_BENCHMARK = os.path.join(
    DIR_RESULTS_DATASETS_STATISTICS,
    "{benchmark}",
    "{evaluation_strategy}",
    "",
)

commons.FOLDERS.add(DIR_RESULTS_DATASETS_STATISTICS)
commons.FOLDERS.add(DIR_RESULTS_DATASETS_STATISTICS_BENCHMARK)


####################################################################################################
####################################################################################################
#             Results exporting          #
####################################################################################################
####################################################################################################
def _extended_describe(
    df: Union[pd.DataFrame, pd.Series],
    stats: list[str],
) -> Union[pd.DataFrame, pd.Series]:
    percentiles = [
        0.01,
        0.05,
        0.1,
        0.2,
        0.25,
        0.3,
        0.4,
        0.5,
        0.6,
        0.75,
        0.8,
        0.9,
        0.95,
        0.99,
    ]

    if isinstance(df, pd.Series):
        df_describe = df.describe(
            percentiles=percentiles,
        )
        df_stats = df.agg(stats)
        return pd.concat(
            objs=[df_describe, df_stats],
            axis=0,
            ignore_index=False,
        )
    elif isinstance(df, pd.DataFrame):
        d = df.describe(
            percentiles=[
                0.01,
                0.05,
                0.1,
                0.2,
                0.25,
                0.3,
                0.4,
                0.5,
                0.6,
                0.75,
                0.8,
                0.9,
                0.95,
                0.99,
            ],
        )
        import pdb

        pdb.set_trace()

        df_stats = df.reindex(d.columns, axis=1).agg(stats)
        return pd.concat(
            objs=[d, df_stats],
            axis=0,
            ignore_index=False,
        )
    else:
        return pd.DataFrame(data=[])


def _compute_statistics_csr_matrix(
    sp_matrix: sp.csr_matrix,
    sp_matrix_name: str,
    benchmark: commons.Benchmarks,
) -> dict[str, Any]:
    from Data_manager.Dataset import gini_index

    sp_matrix_coo = sp_matrix.tocoo()

    results_dict = {
        "dataset": benchmark.value,
        "matrix_name": sp_matrix_name,
        "nnz": sp_matrix.nnz,
        "num_rows": sp_matrix.shape[0],
        "num_col": sp_matrix.shape[1],
        "num_unique_users": np.unique(
            sp_matrix_coo.row,
            return_index=False,
            return_counts=False,
            return_inverse=False,
        ).size,
        "num_unique_items": np.unique(
            sp_matrix_coo.col,
            return_index=False,
            return_counts=False,
            return_inverse=False,
        ).size,
    }

    results_dict = {
        **results_dict,
        "density": results_dict["nnz"]
        / (results_dict["num_rows"] * results_dict["num_col"]),
        "users_per_row": results_dict["num_unique_users"] / results_dict["num_rows"],
        "items_per_row": results_dict["num_unique_items"] / results_dict["num_col"],
    }

    items = [
        ("data", sp_matrix.data),
        ("users", sp_matrix_coo.row),
        ("items", sp_matrix_coo.col),
        ("user_profile_length", np.ediff1d(sp_matrix.indptr)),
        ("item_profile_length", np.ediff1d(sp_matrix.tocsc().indptr)),
    ]

    for name, data in items:
        data_stats_describe = _extended_describe(
            df=pd.Series(data=data),
            stats=["var", "mean", "std", "median", "skew", "kurt", "sum", "count"],
        ).to_dict()
        results_dict.update({f"{name}_{k}": v for k, v in data_stats_describe.items()})
        results_dict[f"{name}_gini_index"] = gini_index(data)

    return results_dict


def _print_dataset_experiments_statistics(
    experiment_benchmark: commons.ExperimentBenchmark,
    experiment_hyper_parameters: commons.HyperParameterTuningParameters,
    folder_path: str,
) -> pd.DataFrame:
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
        dict(
            sp_matrix=interactions_data_splits.sp_urm_train,
            sp_matrix_name="urm_train",
            benchmark=experiment_benchmark.benchmark,
        ),
        dict(
            sp_matrix=interactions_data_splits.sp_urm_validation,
            sp_matrix_name="urm_validation",
            benchmark=experiment_benchmark.benchmark,
        ),
        dict(
            sp_matrix=interactions_data_splits.sp_urm_train_validation,
            sp_matrix_name="urm_train_validation",
            benchmark=experiment_benchmark.benchmark,
        ),
        dict(
            sp_matrix=interactions_data_splits.sp_urm_test,
            sp_matrix_name="urm_test",
            benchmark=experiment_benchmark.benchmark,
        ),
        dict(
            sp_matrix=(
                interactions_data_splits.sp_urm_train_validation
                + interactions_data_splits.sp_urm_test
            ),
            sp_matrix_name="urm_all",
            benchmark=experiment_benchmark.benchmark,
        ),
        dict(
            sp_matrix=impressions_data_splits.sp_uim_train,
            sp_matrix_name="uim_train",
            benchmark=experiment_benchmark.benchmark,
        ),
        dict(
            sp_matrix=impressions_data_splits.sp_uim_validation,
            sp_matrix_name="uim_validation",
            benchmark=experiment_benchmark.benchmark,
        ),
        dict(
            sp_matrix=impressions_data_splits.sp_uim_train_validation,
            sp_matrix_name="uim_train_validation",
            benchmark=experiment_benchmark.benchmark,
        ),
        dict(
            sp_matrix=impressions_data_splits.sp_uim_test,
            sp_matrix_name="uim_test",
            benchmark=experiment_benchmark.benchmark,
        ),
        dict(
            sp_matrix=(
                impressions_data_splits.sp_uim_train_validation
                + impressions_data_splits.sp_uim_test
            ),
            sp_matrix_name="uim_all",
            benchmark=experiment_benchmark.benchmark,
        ),
        dict(
            sp_matrix=impressions_feature_frequency_train,
            sp_matrix_name="uim_frequency_train",
            benchmark=experiment_benchmark.benchmark,
        ),
        dict(
            sp_matrix=impressions_feature_frequency_train_validation,
            sp_matrix_name="uim_frequency_train_validation",
            benchmark=experiment_benchmark.benchmark,
        ),
        dict(
            sp_matrix=impressions_feature_position_train,
            sp_matrix_name="uim_position_train",
            benchmark=experiment_benchmark.benchmark,
        ),
        dict(
            sp_matrix=impressions_feature_position_train_validation,
            sp_matrix_name="uim_position_train_validation",
            benchmark=experiment_benchmark.benchmark,
        ),
        dict(
            sp_matrix=impressions_feature_timestamp_train,
            sp_matrix_name="uim_timestamp_train",
            benchmark=experiment_benchmark.benchmark,
        ),
        dict(
            sp_matrix=impressions_feature_timestamp_train_validation,
            sp_matrix_name="uim_timestamp_train_validation",
            benchmark=experiment_benchmark.benchmark,
        ),
        dict(
            sp_matrix=impressions_feature_last_seen_train,
            sp_matrix_name="uim_last_seen_train",
            benchmark=experiment_benchmark.benchmark,
        ),
        dict(
            sp_matrix=impressions_feature_last_seen_train_validation,
            sp_matrix_name="uim_last_seen_train_validation",
            benchmark=experiment_benchmark.benchmark,
        ),
    ]
    results = []
    for case in cases:
        results.append(_compute_statistics_csr_matrix(**case))

    df_results = pd.DataFrame.from_records(
        data=results,
    )

    with pd.option_context("max_colwidth", 1000):
        df_results.to_parquet(
            path=os.path.join(folder_path, "dataset_statistics.parquet"),
            engine="pyarrow",
            compression=None,
            index=True,
        )
        df_results.to_csv(
            path_or_buf=os.path.join(folder_path, "dataset_statistics.csv"),
            index=True,
            header=True,
            encoding="utf-8",
            na_rep="-",
            sep=";",
            compression=None,
            decimal=",",
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

    return df_results


def print_datasets_statistics(
    to_use_benchmarks: list[commons.Benchmarks],
    to_use_hyper_parameters: list[commons.EHyperParameterTuningParameters],
) -> None:
    printed_experiments: set[
        tuple[commons.Benchmarks, commons.EHyperParameterTuningParameters]
    ] = set()

    results = []

    benchmark: commons.Benchmarks
    hyper_parameters: commons.EHyperParameterTuningParameters

    for benchmark, hyper_parameters in itertools.product(
        to_use_benchmarks,
        to_use_hyper_parameters,
        repeat=1,
    ):
        if (benchmark, hyper_parameters) in printed_experiments:
            continue
        else:
            printed_experiments.add((benchmark, hyper_parameters))

        experiment_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[benchmark]
        experiment_hyper_parameters = (
            commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[hyper_parameters]
        )

        folder_path_export = DIR_RESULTS_DATASETS_STATISTICS_BENCHMARK.format(
            benchmark=experiment_benchmark.benchmark.value,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
        )

        os.makedirs(folder_path_export, exist_ok=True)

        df_results = _print_dataset_experiments_statistics(
            experiment_benchmark=experiment_benchmark,
            experiment_hyper_parameters=experiment_hyper_parameters,
            folder_path=folder_path_export,
        )
        results.append(df_results)

    df_results = pd.concat(
        objs=results,
        axis=0,
        ignore_index=True,
    )

    folder_path_export = DIR_RESULTS_DATASETS_STATISTICS
    os.makedirs(folder_path_export, exist_ok=True)

    with pd.option_context("max_colwidth", 1000):
        df_results.to_csv(
            path_or_buf=os.path.join(folder_path_export, "datasets_statistics.csv"),
            index=True,
            header=True,
            encoding="utf-8",
            na_rep="-",
            sep=";",
            compression=None,
            decimal=",",
        )
        df_results.to_latex(
            buf=os.path.join(folder_path_export, "datasets_statistics.tex"),
            index=True,
            header=True,
            escape=False,
            float_format="{:.4f}".format,
            encoding="utf-8",
            na_rep="-",
            longtable=True,
        )

    logger.info(f"Successfully finished exporting statistics of datasets.")


def print_datasets_statistics_thesis(
    to_use_benchmarks: list[commons.Benchmarks],
    to_use_hyper_parameters: list[commons.EHyperParameterTuningParameters],
) -> None:
    printed_experiments: set[
        tuple[commons.Benchmarks, commons.EHyperParameterTuningParameters]
    ] = set()

    results = []

    benchmark: commons.Benchmarks
    hyper_parameters: commons.EHyperParameterTuningParameters

    for benchmark, hyper_parameters in itertools.product(
        to_use_benchmarks,
        to_use_hyper_parameters,
        repeat=1,
    ):
        if (benchmark, hyper_parameters) in printed_experiments:
            continue
        else:
            printed_experiments.add((benchmark, hyper_parameters))

        experiment_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[benchmark]
        experiment_hyper_parameters = (
            commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[hyper_parameters]
        )

        folder_path_export = DIR_RESULTS_DATASETS_STATISTICS_BENCHMARK.format(
            benchmark=experiment_benchmark.benchmark.value,
            evaluation_strategy=experiment_hyper_parameters.evaluation_strategy.value,
        )

        filename = "dataset_statistics.parquet"

        df_results = pd.read_parquet(
            path=os.path.join(folder_path_export, filename),
            engine="pyarrow",
        )
        results.append(df_results)

    df_results = pd.concat(
        objs=results,
        axis=0,
        ignore_index=True,
    )

    columns_to_export = [
        "matrix_name",
        "dataset",
        "nnz",
        "num_rows",
        "num_col",
        "num_unique_users",
        "num_unique_items",
        "density",
        "user_profile_length_mean",
        "user_profile_length_std",
        "user_profile_length_median",
        "user_profile_length_skew",
        "user_profile_length_kurt",
        "user_profile_length_sum",
        "user_profile_length_gini_index",
        "item_profile_length_mean",
        "item_profile_length_std",
        "item_profile_length_median",
        "item_profile_length_skew",
        "item_profile_length_kurt",
        "item_profile_length_sum",
        "item_profile_length_gini_index",
    ]
    matrices_to_export = ["urm_all", "uim_all"]

    df_results = df_results[df_results["matrix_name"].isin(matrices_to_export)][
        columns_to_export
    ]

    df_results_transposed = df_results.set_index(["matrix_name", "dataset"]).transpose()

    folder_path_export = DIR_RESULTS_DATASETS_STATISTICS
    os.makedirs(folder_path_export, exist_ok=True)

    with pd.option_context("max_colwidth", 1000):
        df_results.to_csv(
            path_or_buf=os.path.join(
                folder_path_export, "thesis_datasets_statistics.csv"
            ),
            index=True,
            header=True,
            encoding="utf-8",
            na_rep="-",
            sep=";",
            compression=None,
            decimal=",",
        )
        df_results.to_latex(
            buf=os.path.join(folder_path_export, "thesis_datasets_statistics.tex"),
            index=True,
            header=True,
            escape=False,
            float_format="{:.4f}".format,
            encoding="utf-8",
            na_rep="-",
            longtable=True,
        )

        df_results_transposed.to_csv(
            path_or_buf=os.path.join(
                folder_path_export, "thesis_datasets_statistics_transposed.csv"
            ),
            index=True,
            header=True,
            encoding="utf-8",
            na_rep="-",
            sep=";",
            compression=None,
            decimal=",",
        )
        df_results_transposed.to_latex(
            buf=os.path.join(
                folder_path_export, "thesis_datasets_statistics_transposed.tex"
            ),
            index=True,
            header=True,
            escape=False,
            float_format="{:.4f}".format,
            encoding="utf-8",
            na_rep="-",
            longtable=True,
        )

    logger.info(
        f"Successfully finished exporting statistics of datasets to folder '%(folder)s'",
        {"folder": folder_path_export},
    )
