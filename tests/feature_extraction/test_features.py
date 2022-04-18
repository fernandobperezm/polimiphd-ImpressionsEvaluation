import numpy as np
import pandas as pd
from recsys_framework_extensions.data.features import (
    extract_frequency_user_item,
    extract_position_user_item,
    extract_timestamp_user_item,
    extract_last_seen_user_item, extract_last_seen_user_item_2,
)


class TestExtractPositionUserImpression:
    def test_keep_last(
        self, df: pd.DataFrame,
    ):
        # Arrange
        df = df[df["user_id"] == 0]
        test_to_keep = "last"
        df_expected = pd.DataFrame(
            data={
                "user_id": [0, 0, 0, 0, 0],
                "impressions": [0, 5, 2, 3, 1],
                "feature-user_id-impressions-position": [1, 1, 0, 1, 2],
            },
            index=[1, 4, 6, 7, 8],
        ).astype({
            "user_id": np.int64,
            "impressions": "object",
            "feature-user_id-impressions-position": np.int32,
        })

        # Act
        df_result, _ = extract_position_user_item(
            df=df,
            users_column="user_id",
            items_column="impressions",
            positions_column=None,
            to_keep=test_to_keep,
        )

        # Assert
        pd.testing.assert_frame_equal(
            left=df_expected,
            right=df_result,
        )

    def test_keep_first(
        self, df: pd.DataFrame,
    ):
        # Arrange
        df = df[df["user_id"] == 0]
        test_to_keep = "first"
        df_expected = pd.DataFrame(
            data={
                "user_id": [0, 0, 0, 0, 0],
                "impressions": [5, 0, 3, 1, 2],
                "feature-user_id-impressions-position": [0, 1, 2, 0, 2],
            },
            index=[0, 1, 2, 3, 5],
        ).astype({
            "user_id": np.int64,
            "impressions": "object",
            "feature-user_id-impressions-position": np.int32,
        })

        # Act
        df_result, _ = extract_position_user_item(
            df=df,
            users_column="user_id",
            items_column="impressions",
            positions_column=None,
            to_keep=test_to_keep,
        )

        # Assert
        pd.testing.assert_frame_equal(
            left=df_expected,
            right=df_result,
        )

    def test_keep_false(
        self, df: pd.DataFrame,
    ):
        # Arrange
        df = df[df["user_id"] == 0]
        test_to_keep = False
        df_expected = pd.DataFrame(
            data={
                "user_id": [0],
                "impressions": [0],
                "feature-user_id-impressions-position": [1],
            },
            index=[1],
        ).astype({
            "user_id": np.int64,
            "impressions": "object",
            "feature-user_id-impressions-position": np.int32,
        })

        # Act
        df_result, _ = extract_position_user_item(
            df=df,
            users_column="user_id",
            items_column="impressions",
            positions_column=None,
            to_keep=test_to_keep,
        )

        # Assert
        pd.testing.assert_frame_equal(
            left=df_expected,
            right=df_result,
        )


class TestExtractTimestampUserImpression:
    def test_keep_last(
        self, df: pd.DataFrame,
    ):
        # Arrange
        df = df[df["user_id"] == 0]
        test_to_keep = "last"
        df_expected = pd.DataFrame(
            data={
                "user_id": [0, 0, 0, 0, 0],
                "impressions": [0, 5, 2, 3, 1],
            },
            index=[1, 4, 6, 7, 8],
        ).astype({
            "user_id": np.int32,
            "impressions": "object",
        })
        df_expected["timestamp"] = pd.to_datetime(
            [
                "2022-01-01 06:06:49",
                "2022-01-01 11:41:57",
                "2022-01-02 07:26:15",
                "2022-01-02 07:26:15",
                "2022-01-02 07:26:15",
            ],
            yearfirst=True,
            infer_datetime_format=True,
        )
        df_expected["feature-user_id-impressions-timestamp"] = df_expected["timestamp"].astype(np.int64) / 10**9

        # Act
        df_result, _ = extract_timestamp_user_item(
            df=df,
            users_column="user_id",
            items_column="impressions",
            timestamp_column="timestamp",
            to_keep=test_to_keep,
        )

        # Assert
        pd.testing.assert_frame_equal(
            left=df_expected,
            right=df_result,
        )

    def test_keep_first(
        self, df: pd.DataFrame,
    ):
        # Arrange
        df = df[df["user_id"] == 0]
        test_to_keep = "first"
        df_expected = pd.DataFrame(
            data={
                "user_id": [0, 0, 0, 0, 0],
                "impressions": [5, 0, 3, 1, 2],
            },
            index=[0, 1, 2, 3, 5],
        ).astype({
            "user_id": np.int32,
            "impressions": "object",
        })
        df_expected["timestamp"] = pd.to_datetime(
            [
                "2022-01-01 06:06:49",
                "2022-01-01 06:06:49",
                "2022-01-01 06:06:49",
                "2022-01-01 11:41:57",
                "2022-01-01 11:41:57",
            ],
            yearfirst=True,
            infer_datetime_format=True,
        )
        df_expected["feature-user_id-impressions-timestamp"] = df_expected["timestamp"].astype(np.int64) / 10 ** 9

        # Act
        df_result, _ = extract_timestamp_user_item(
            df=df,
            users_column="user_id",
            items_column="impressions",
            timestamp_column="timestamp",
            to_keep=test_to_keep,
        )

        # Assert
        pd.testing.assert_frame_equal(
            left=df_expected,
            right=df_result,
        )

    def test_keep_false(
        self, df: pd.DataFrame,
    ):
        # Arrange
        df = df[df["user_id"] == 0]
        test_to_keep = False
        df_expected = pd.DataFrame(
            data={
                "user_id": [0],
                "impressions": [0],
            },
            index=[1],
        ).astype({
            "user_id": np.int32,
            "impressions": "object",
        })
        df_expected["timestamp"] = pd.to_datetime(
            [
                "2022-01-01 06:06:49",
            ],
            yearfirst=True,
            infer_datetime_format=True,
        )
        df_expected["feature-user_id-impressions-timestamp"] = df_expected["timestamp"].astype(np.int64) / 10 ** 9

        # Act
        df_result, _ = extract_timestamp_user_item(
            df=df,
            users_column="user_id",
            items_column="impressions",
            timestamp_column="timestamp",
            to_keep=test_to_keep,
        )

        # Assert
        pd.testing.assert_frame_equal(
            left=df_expected,
            right=df_result,
        )


class TestExtractFrequencyUserImpression:
    def test_works(
        self, df: pd.DataFrame,
    ):
        # Arrange
        df = df[df["user_id"] == 0]
        print(df)
        df_expected = pd.DataFrame(
            data={
                "user_id": [0, 0, 0, 0, 0],
                "impressions": [0, 1, 2, 3, 5],
                "feature-user_id-impressions-frequency": [1, 2, 2, 2, 2],
            },
            index=[0, 1, 2, 3, 4],
        ).astype({
            "user_id": np.int64,
            "impressions": np.int64,
        })

        # Act
        df_result, _ = extract_frequency_user_item(
            df=df,
            users_column="user_id",
            items_column="impressions",
        )

        # Assert
        pd.testing.assert_frame_equal(
            left=df_expected,
            right=df_result,
        )


class TestExtractLastSeenUserImpression:
    def test_works(
        self, df: pd.DataFrame,
    ):
        # Arrange
        df = df[df["user_id"] == 0]

        df_expected = pd.DataFrame(
            data={
                "user_id": [0, 0, 0, 0],
                "impressions": [1, 2, 3, 5],
                "timestamp": [
                    "2022-01-02 07:26:15",
                    "2022-01-02 07:26:15",
                    "2022-01-02 07:26:15",
                    "2022-01-01 11:41:57",
                ],
                "feature_last_seen_total_seconds": [
                    71058.,
                    71058.,
                    91166.,
                    20108.,
                ],
                "feature_last_seen_total_minutes": [
                    1184.3,
                    1184.3,
                    1519.4333333333334,
                    335.1333333333333,
                ],
                "feature_last_seen_total_hours": [
                    19.738333333333333,
                    19.738333333333333,
                    25.323888888888888,
                    5.585555555555556,
                ],
                "feature_last_seen_total_days": [
                    0.8224305555555556,
                    0.8224305555555556,
                    1.055162037037037,
                    0.23273148148148148,
                ],
                "feature_last_seen_total_weeks": [
                    0.11749007936507937,
                    0.11749007936507937,
                    0.15073743386243385,
                    0.0332473544973545,
                ],
            },
            index=[2, 4, 6, 8],
        ).astype({
            "user_id": np.int32,
            "impressions": object,
            "feature_last_seen_total_seconds": np.float64,
            "feature_last_seen_total_minutes": np.float64,
            "feature_last_seen_total_hours": np.float64,
            "feature_last_seen_total_days": np.float64,
            "feature_last_seen_total_weeks": np.float64,
        })
        df_expected["timestamp"] = pd.to_datetime(
            [
                "2022-01-02 07:26:15",
                "2022-01-02 07:26:15",
                "2022-01-02 07:26:15",
                "2022-01-01 11:41:57",
            ],
            yearfirst=True,
            infer_datetime_format=True,
        )

        # Act
        df_result, _ = extract_last_seen_user_item(
            df=df,
            users_column="user_id",
            items_column="impressions",
            timestamp_column="timestamp",
        )

        # Assert
        pd.testing.assert_frame_equal(
            left=df_expected,
            right=df_result,
        )

    def test_works_2(
        self, df: pd.DataFrame,
    ):
        # Arrange
        df = df[df["user_id"] == 0]

        df_expected = pd.DataFrame(
            data={
                "user_id": [0, 0, 0, 0],
                "impressions": [5, 3, 1, 2], # [1, 2, 3, 5],
                "timestamp": [
                    "2022-01-01 11:41:57",
                    "2022-01-02 07:26:15",
                    "2022-01-02 07:26:15",
                    "2022-01-02 07:26:15",
                ],
                "feature_last_seen_total_seconds": [
                    20108.,
                    91166.,
                    71058.,
                    71058.,
                ],
                "feature_last_seen_total_minutes": [
                    335.1333333333333,
                    1519.4333333333334,
                    1184.3,
                    1184.3,
                ],
                "feature_last_seen_total_hours": [
                    5.585555555555556,
                    25.323888888888888,
                    19.738333333333333,
                    19.738333333333333,
                ],
                "feature_last_seen_total_days": [
                    0.23273148148148148,
                    1.055162037037037,
                    0.8224305555555556,
                    0.8224305555555556,
                ],
                "feature_last_seen_total_weeks": [
                    0.0332473544973545,
                    0.15073743386243385,
                    0.11749007936507937,
                    0.11749007936507937,
                ],
            },
            index=[4, 7, 8, 6],
        ).astype({
            "user_id": np.int32,
            "impressions": object,
            "feature_last_seen_total_seconds": np.float64,
            "feature_last_seen_total_minutes": np.float64,
            "feature_last_seen_total_hours": np.float64,
            "feature_last_seen_total_days": np.float64,
            "feature_last_seen_total_weeks": np.float64,
        })
        df_expected["timestamp"] = pd.to_datetime(
            [
                "2022-01-01 11:41:57",
                "2022-01-02 07:26:15",
                "2022-01-02 07:26:15",
                "2022-01-02 07:26:15",
            ],
            yearfirst=True,
            infer_datetime_format=True,
        )

        # Act
        df_result, _ = extract_last_seen_user_item_2(
            df=df,
            users_column="user_id",
            items_column="impressions",
            timestamp_column="timestamp",
        )

        # Assert
        pd.testing.assert_frame_equal(
            left=df_expected,
            right=df_result,
        )

    def test_methods_equivalent(
        self, df: pd.DataFrame
    ):
        # Arrange

        # Act
        df_result, _ = extract_last_seen_user_item(
            df=df,
            users_column="user_id",
            items_column="impressions",
            timestamp_column="timestamp",
        )

        df_result_2, _ = extract_last_seen_user_item_2(
            df=df,
            users_column="user_id",
            items_column="impressions",
            timestamp_column="timestamp",
        )

        df_result = df_result.sort_values(
            by=["user_id", "impressions", "timestamp"],
            ignore_index=True,
            axis="index",
            inplace=False,
            ascending=True,
        )
        df_result_2 = df_result_2.sort_values(
            by=["user_id", "impressions", "timestamp"],
            ignore_index=True,
            axis="index",
            inplace=False,
            ascending=True,
        )

        # Assert
        pd.testing.assert_frame_equal(df_result, df_result_2)
