import datetime

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pytest import fixture
from faker import Faker

from impression_recommenders.heuristics.latest_impressions import LastImpressionsRecommender

Faker.seed(1234567890)
fake = Faker()

rng = np.random.default_rng(
    seed=1234567890,
)

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

    impressions = np.empty(shape=(LENGTH_IMPRESSIONS, ), dtype=np.int32)
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
        ["timestamp", "user_id", "item_id",]
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
def uim(df: pd.DataFrame) -> sp.csr_matrix:
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
    arr_data = np.ones_like(arr_item_id, dtype=np.int32)

    return sp.csr_matrix(
        (
            arr_data,
            (arr_user_id, arr_item_id),
        ),
        shape=(NUM_USERS, NUM_ITEMS),
        dtype=np.int32,
    )


class TestLastImpressionsRecommender:

    EXPECTED_RECOMMENDATIONS = np.array([
        [2, 3, 1],
        [0, 3, 4],
        [0, 3, 6],
        [],  # User without impressions
        [1, 6, 2],
        [2, 4, 5],
        [2, 0, 5],
        [5, 3, 2],
        [],  # User without impressions
        [1, 6, 4],
    ], dtype=object)

    EXPECTED_ITEM_SCORES = np.array([
        [np.NINF, -2., 0., -1., np.NINF, np.NINF, np.NINF],
        [0., np.NINF, np.NINF, -1., -2., np.NINF, np.NINF],
        [0., np.NINF, np.NINF, -1., np.NINF, np.NINF, -2.],
        [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],  # User without impressions
        [np.NINF, 0., -2., np.NINF, np.NINF, np.NINF, -1.],
        [np.NINF, np.NINF, 0., np.NINF, -1., -2., np.NINF],
        [-1., np.NINF, 0., np.NINF, np.NINF, -2., np.NINF],
        [np.NINF, np.NINF, -2., -1., np.NINF, 0., np.NINF],
        [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],  # User without impressions
        [np.NINF, 0., np.NINF, np.NINF, -2., np.NINF, -1.]
    ], dtype=np.float32)

    def test_impl(
        self, urm: sp.csr_matrix, uim_timestamp: sp.csr_matrix, uim_position: sp.csr_matrix
    ):
        # arrange
        test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_items = None
        test_cutoff = 3

        rec = LastImpressionsRecommender(
            urm_train=urm,
            uim_timestamp=uim_timestamp,
            uim_position=uim_position,
        )

        # act
        rec.fit()
        recommendations, scores = rec.recommend(
            user_id_array=test_users,
            cutoff=test_cutoff,
            remove_seen_flag=False,
            remove_top_pop_flag=False,
            remove_custom_items_flag=False,
            return_scores=True,
        )

        # assert
        assert np.array_equal(self.EXPECTED_RECOMMENDATIONS, recommendations)
        assert np.array_equal(self.EXPECTED_ITEM_SCORES, scores)
