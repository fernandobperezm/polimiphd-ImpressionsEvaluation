""" ContentWiseImpressionsReader.py
This module reads, processes, splits, creates impressiosn features, and saves into disk the ContentWiseImpressions
dataset.

"""
import impressions_evaluation.readers.ContentWiseImpressions.reader as cw_impressions_reader

from recsys_framework_extensions.logging import get_logger
from tqdm import tqdm

import pdb
pdb.set_trace()


tqdm.pandas()


logger = get_logger(
    logger_name=__file__,
)


def content_wise_impressions_statistics_full_dataset():
    pdb.set_trace()

    config = cw_impressions_reader.ContentWiseImpressionsConfig()

    raw_data = cw_impressions_reader.PandasContentWiseImpressionsRawData(
        config=config,
    )

    df_raw_data = raw_data.data

    pdb.set_trace()

    print(df_raw_data)


def content_wise_impressions_statistics_non_duplicates_dataset():
    ...
