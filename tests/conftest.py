import datetime

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pytest import fixture
from faker import Faker

# ORIGINAL SEED IS 1234567890.
# Do not change seed unless you want to generate different fake data points. This is not
# recommended as all tests cases are done expecting the data created with this random seed.
seed = 1234567890

Faker.seed(seed)
fake = Faker()

rng = np.random.default_rng(seed=seed)

NUM_USERS = 10
NUM_ITEMS = 7
NUM_INTERACTIONS = 20
LENGTH_IMPRESSIONS = 3

ALL_USER_IDS = np.arange(start=0, stop=NUM_USERS, step=1, dtype=np.int32)
ALL_ITEM_IDS = np.arange(start=0, stop=NUM_ITEMS, step=1, dtype=np.int32)

MIN_TIMESTAMP = datetime.datetime(year=2022, month=1, day=1, hour=0, minute=0, second=1)
MAX_TIMESTAMP = datetime.datetime(year=2022, month=1, day=2, hour=23, minute=59, second=59)

USER_IDS = rng.choice(
    a=ALL_USER_IDS, size=NUM_INTERACTIONS, replace=True, shuffle=True,
)
ITEM_IDS = rng.choice(
    a=ALL_ITEM_IDS, size=NUM_INTERACTIONS, replace=True, shuffle=True,
)

IMPRESSIONS = []
for item_id in ITEM_IDS:
    items_except_interacted = list(
        set(ALL_ITEM_IDS).difference([item_id])
    )

    impressions = np.empty(shape=(LENGTH_IMPRESSIONS,), dtype=np.int32)
    impressions[0] = item_id
    impressions[1:] = rng.choice(
        a=items_except_interacted, size=LENGTH_IMPRESSIONS - 1, replace=False, shuffle=False
    )

    rng.shuffle(impressions)
    IMPRESSIONS.append(impressions)

POSITIONS = np.array(
    [0, 1, 2] * NUM_INTERACTIONS
).reshape(
    (NUM_INTERACTIONS, LENGTH_IMPRESSIONS)
)
TIMESTAMPS = np.array(
    [
        fake.date_time_between(start_date=MIN_TIMESTAMP, end_date=MAX_TIMESTAMP)
        for _ in range(NUM_INTERACTIONS)
    ],
    dtype=object
)


@fixture
def df() -> pd.DataFrame:
    dataframe = pd.DataFrame(
        data={
            "user_id": USER_IDS,
            "item_id": ITEM_IDS,
            "timestamp": TIMESTAMPS,
            "impressions": IMPRESSIONS,
            "positions": list(POSITIONS),
        }
    ).sort_values(
        by="timestamp",
        ascending=True,
        inplace=False,
        ignore_index=True,
    )
    return dataframe


@fixture
def df_debug(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(
        subset="user_id",
        inplace=False,
        keep="last",
        ignore_index=False,
    ).sort_values(
        by="user_id",
        ascending=True,
        inplace=False,
        ignore_index=False,
    )
    return df


@fixture
def urm(df: pd.DataFrame) -> sp.csr_matrix:
    df_data = df.drop_duplicates(
        subset=["user_id", "item_id"],
        keep="first",
        inplace=False,
    )

    arr_user_id = df_data["user_id"].to_numpy()
    arr_item_id = df_data["item_id"].to_numpy()
    arr_data = np.ones_like(arr_item_id, dtype=np.int32)

    return sp.csr_matrix(
        (
            arr_data,
            (arr_user_id, arr_item_id),
        ),
        shape=(NUM_USERS, NUM_ITEMS),
    )


@fixture
def uim_timestamp(df: pd.DataFrame) -> sp.csr_matrix:
    df = df.set_index(
        ["timestamp", "user_id", "item_id", ]
    ).apply(
        pd.Series.explode,
    ).reset_index(
        drop=False,
    ).drop_duplicates(
        subset=["user_id", "impressions"],
        keep="last",
        inplace=False,
    )

    arr_user_id = df["user_id"].to_numpy()
    arr_item_id = df["impressions"].to_numpy()
    arr_data = df["timestamp"].to_numpy().astype(np.int64) // 10 ** 9

    return sp.csr_matrix(
        (
            arr_data,
            (arr_user_id, arr_item_id),
        ),
        shape=(NUM_USERS, NUM_ITEMS),
        dtype=np.int64,
    )


@fixture
def uim_position(df: pd.DataFrame) -> sp.csr_matrix:
    df = df.set_index(
        ["timestamp", "user_id", "item_id"]
    ).apply(
        pd.Series.explode,
    ).reset_index(
        drop=False,
    ).drop_duplicates(
        subset=["user_id", "positions"],
        keep="last",
        inplace=False,
    )

    arr_user_id = df["user_id"].to_numpy()
    arr_item_id = df["impressions"].to_numpy()
    arr_data = df["positions"].to_numpy()

    return sp.csr_matrix(
        (
            arr_data,
            (arr_user_id, arr_item_id),
        ),
        shape=(NUM_USERS, NUM_ITEMS),
        dtype=np.int32,
    )


@fixture
def uim_frequency(df: pd.DataFrame) -> sp.csr_matrix:
    df = df.set_index(
        ["timestamp", "user_id", "item_id"]
    ).apply(
        pd.Series.explode,
    ).reset_index(
        drop=False,
    )

    arr_user_id = df["user_id"].to_numpy()
    arr_item_id = df["impressions"].to_numpy()
    arr_data = np.ones_like(arr_user_id, dtype=np.int32)

    return sp.csr_matrix(
        (
            arr_data,
            (arr_user_id, arr_item_id),
        ),
        shape=(NUM_USERS, NUM_ITEMS),
        dtype=np.int32,
    )


@fixture
def uim_last_seen(df: pd.DataFrame) -> sp.csr_matrix:
    """Last seen hours."""
    from recsys_framework_extensions.data.features import extract_last_seen_user_item

    df_keep, _ = extract_last_seen_user_item(
        df=df,
        users_column="user_id",
        items_column="impressions",
        timestamp_column="timestamp"
    )

    arr_user_id = df_keep["user_id"].to_numpy()
    arr_item_id = df_keep["impressions"].to_numpy()
    arr_data = df_keep["feature_last_seen_total_hours"].to_numpy()

    return sp.csr_matrix(
        (
            arr_data,
            (arr_user_id, arr_item_id),
        ),
        shape=(NUM_USERS, NUM_ITEMS),
        dtype=np.int32,
    )


@fixture
def uim(df: pd.DataFrame) -> sp.csr_matrix:
    df = df.set_index(
        ["timestamp", "user_id", "item_id"]
    ).apply(
        pd.Series.explode,
    ).reset_index(
        drop=False,
    ).drop_duplicates(
        subset=["user_id", "impressions"],
        keep="last",
        inplace=False,
    )

    arr_user_id = df["user_id"].to_numpy()
    arr_item_id = df["impressions"].to_numpy()
    arr_data = np.ones_like(arr_item_id, dtype=np.int32)

    return sp.csr_matrix(
        (
            arr_data,
            (arr_user_id, arr_item_id),
        ),
        shape=(NUM_USERS, NUM_ITEMS),
        dtype=np.int32,
    )
