#!/usr/bin/env python3
import logging
import os
import pandas as pd

from impressions_evaluation.readers.ContentWiseImpressions.statistics import (
    content_wise_impressions_statistics_full_dataset,
    contentwise_impressions_compute_statistics_thesis,
)
from impressions_evaluation.readers.FINNNoSlates.statistics import (
    finn_no_slates_statistics_full_dataset,
)
from impressions_evaluation.readers.MIND.statistics import (
    mind_small_statistics_full_dataset,
    mind_large_statistics_full_dataset,
)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    dir_export_base = os.path.join(
        os.getcwd(),
        "result_experiments",
        "script_dataset_statistics",
        "",
    )
    dir_export_parquet = os.path.join(dir_export_base, "parquet", "")
    dir_export_latex = os.path.join(dir_export_base, "latex", "")
    dir_export_plots = os.path.join(dir_export_base, "plots", "")
    filename_export = "script_dataset_statistics"

    os.makedirs(dir_export_parquet, exist_ok=True)
    os.makedirs(dir_export_latex, exist_ok=True)
    os.makedirs(dir_export_plots, exist_ok=True)

    results = mind_large_statistics_full_dataset()
    print(results)

    # results_cw_impressions = contentwise_impressions_compute_statistics_thesis(
    #     dir_results=dir_export_plots,
    # )

    # print(results_cw_impressions)

    # statistics_cw_full = content_wise_impressions_statistics_full_dataset()
    # statistics_finn_full = finn_no_slates_statistics_full_dataset()
    # statistics_mind_full = mind_large_statistics_full_dataset()
    # statistics_mind_small_full = mind_small_statistics_full_dataset()
    #
    # df_statistics = pd.DataFrame.from_records(
    #     data=[
    #         statistics_cw_full,
    #         statistics_finn_full,
    #         statistics_mind_full,
    #         statistics_mind_small_full,
    #     ],
    # )
    #
    # df_statistics.style.to_latex(
    #     buf=os.path.join(dir_export_latex, f"{filename_export}.tex"),
    #     position="t",
    #     position_float="centering",
    #     column_format="l|SS|SS|SS",
    #     siunitx=True,
    #     encoding="utf-8",
    #     caption="Dataset Statistics",
    # )
    # df_statistics.to_parquet(
    #     path=os.path.join(dir_export_parquet, f"{filename_export}.parquet"),
    #     engine="pyarrow",
    # )
