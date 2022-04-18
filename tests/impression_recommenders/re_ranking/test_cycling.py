from mock import patch
import numpy as np
import scipy.sparse as sp

from impression_recommenders.re_ranking.cycling import CyclingRecommender
from tests.conftest import seed
from Recommenders.BaseRecommender import BaseRecommender


class TestCyclingRecommender:
    def test_all_users_no_items(
        self, urm: sp.csr_matrix, uim_frequency: sp.csr_matrix,
    ):

        test_trained_recommender_compute_item_score = np.array(
            [
                [1, 6, 3, 2, 3, 5, 4],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 2, 3, 4, 5, 6, 7],
                [7, 6, 5, 4, 3, 2, 1],
                [9, 7, 5, 4, 8, 7, 3],

                [1, 6, 3, 2, 3, 5, 4],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 2, 3, 4, 5, 6, 7],
                [7, 6, 5, 4, 3, 2, 1],
                [9, 7, 5, 4, 8, 7, 3],
            ],
            dtype=np.int32,
        )

        mock_base_recommender = BaseRecommender(URM_train=urm)
        with patch.object(
            mock_base_recommender,
            '_compute_item_score',
            return_value=test_trained_recommender_compute_item_score
        ) as _:
            # arrange
            test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            test_items = None
            test_cutoff = 3
            test_weight = 1  # weight as 1 to have presentation score = frequency of impressions.

            expected_item_scores = np.array([
                [3, 7, 5, 4, 1, 6, 2],
                [6, 1, 1, 6, 4, 1, 4],
                [5, 1, 2, 6, 3, 4, 7],
                [7, 6, 5, 4, 3, 2, 1],
                [5, 3, 6, 2, 1, 3, 7],
                [2, 4, 7, 5, 3, 6, 1],
                [5, 1, 5, 1, 1, 5, 1],
                [1, 3, 5, 7, 4, 6, 2],
                [7, 6, 5, 4, 3, 2, 1],
                [4, 6, 2, 1, 7, 3, 5],
            ], dtype=np.float64)

            rec = CyclingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                trained_recommender=mock_base_recommender,
                seed=seed,
            )

            # act
            rec.fit(weight=test_weight)
            recommendations, scores = rec.recommend(
                user_id_array=test_users,
                items_to_compute=test_items,
                cutoff=test_cutoff,
                remove_seen_flag=False,
                remove_top_pop_flag=False,
                remove_custom_items_flag=False,
                return_scores=True,
            )

            # assert
            # For this particular recommender, we cannot test recommendations, as there might be several ties (same
            # timestamp for two impressions) and the .recommend handles ties in a non-deterministic way.
            for row in range(expected_item_scores.shape[0]):
                for col in range(expected_item_scores.shape[1]):
                    assert expected_item_scores[row, col] == scores[row, col]

            assert np.array_equal(expected_item_scores, scores)

    def test_all_users_some_items(
        self, urm: sp.csr_matrix, uim_frequency: sp.csr_matrix,
    ):
        test_trained_recommender_compute_item_score = np.array(
            [
                [1, 6, 3, 2, 3, 5, 4],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 2, 3, 4, 5, 6, 7],
                [7, 6, 5, 4, 3, 2, 1],
                [9, 7, 5, 4, 8, 7, 3],

                [1, 6, 3, 2, 3, 5, 4],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 2, 3, 4, 5, 6, 7],
                [7, 6, 5, 4, 3, 2, 1],
                [9, 7, 5, 4, 8, 7, 3],
            ],
            dtype=np.float32,
        )

        mock_base_recommender = BaseRecommender(URM_train=urm)
        with patch.object(
            mock_base_recommender,
            '_compute_item_score',
            return_value=test_trained_recommender_compute_item_score
        ) as _:
            # arrange
            test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            test_items = [1, 2, 5]
            test_cutoff = 3
            test_weight = 1  # weight as 1 to have presentation score = frequency of impressions.

            expected_item_scores = np.array([
                [np.NINF, 7, 5, np.NINF, np.NINF, 6, np.NINF],
                [np.NINF, 1, 1, np.NINF, np.NINF, 1, np.NINF],
                [np.NINF, 1, 2, np.NINF, np.NINF, 4, np.NINF],
                [np.NINF, 6, 5, np.NINF, np.NINF, 2, np.NINF],
                [np.NINF, 3, 6, np.NINF, np.NINF, 3, np.NINF],
                [np.NINF, 4, 7, np.NINF, np.NINF, 6, np.NINF],
                [np.NINF, 1, 5, np.NINF, np.NINF, 5, np.NINF],
                [np.NINF, 3, 5, np.NINF, np.NINF, 6, np.NINF],
                [np.NINF, 6, 5, np.NINF, np.NINF, 2, np.NINF],
                [np.NINF, 6, 2, np.NINF, np.NINF, 3, np.NINF],
            ], dtype=np.float64)

            rec = CyclingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                trained_recommender=mock_base_recommender,
                seed=seed,
            )

            # act
            rec.fit(weight=test_weight)
            recommendations, scores = rec.recommend(
                user_id_array=test_users,
                items_to_compute=test_items,
                cutoff=test_cutoff,
                remove_seen_flag=False,
                remove_top_pop_flag=False,
                remove_custom_items_flag=False,
                return_scores=True,
            )

            # assert
            for row in range(expected_item_scores.shape[0]):
                for col in range(expected_item_scores.shape[1]):
                    assert expected_item_scores[row, col] == scores[row, col]

            assert np.allclose(expected_item_scores, scores)

    def test_all_users_all_items(
        self, urm: sp.csr_matrix, uim_frequency: sp.csr_matrix,
    ):
        test_trained_recommender_compute_item_score = np.array(
            [
                [1, 6, 3, 2, 3, 5, 4],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 2, 3, 4, 5, 6, 7],
                [7, 6, 5, 4, 3, 2, 1],
                [9, 7, 5, 4, 8, 7, 3],

                [1, 6, 3, 2, 3, 5, 4],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 2, 3, 4, 5, 6, 7],
                [7, 6, 5, 4, 3, 2, 1],
                [9, 7, 5, 4, 8, 7, 3],
            ],
            dtype=np.int32,
        )

        mock_base_recommender = BaseRecommender(URM_train=urm)
        with patch.object(
            mock_base_recommender,
            '_compute_item_score',
            return_value=test_trained_recommender_compute_item_score
        ) as _:
            # arrange
            test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            test_items = [0, 1, 2, 3, 4, 5, 6]
            test_cutoff = 3
            test_weight = 1  # weight is 1 so presentation_score = frequency.

            expected_item_scores = np.array([
                [3, 7, 5, 4, 1, 6, 2],
                [6, 1, 1, 6, 4, 1, 4],
                [5, 1, 2, 6, 3, 4, 7],
                [7, 6, 5, 4, 3, 2, 1],
                [5, 3, 6, 2, 1, 3, 7],
                [2, 4, 7, 5, 3, 6, 1],
                [5, 1, 5, 1, 1, 5, 1],
                [1, 3, 5, 7, 4, 6, 2],
                [7, 6, 5, 4, 3, 2, 1],
                [4, 6, 2, 1, 7, 3, 5],
            ], dtype=np.float64)

            rec = CyclingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                trained_recommender=mock_base_recommender,
                seed=seed,
            )

            # act
            rec.fit(weight=test_weight)
            recommendations, scores = rec.recommend(
                user_id_array=test_users,
                items_to_compute=test_items,
                cutoff=test_cutoff,
                remove_seen_flag=False,
                remove_top_pop_flag=False,
                remove_custom_items_flag=False,
                return_scores=True,
            )

            # assert
            assert np.array_equal(expected_item_scores, scores)

    def test_some_users_no_items(
        self, urm: sp.csr_matrix, uim_frequency: sp.csr_matrix,
    ):
        test_trained_recommender_compute_item_score = np.array(
            [
                [1, 6, 3, 2, 3, 5, 4],
                [1, 1, 1, 1, 1, 1, 1],
                [7, 6, 5, 4, 3, 2, 1],

                [1, 1, 1, 1, 1, 1, 1],
                [1, 2, 3, 4, 5, 6, 7],
                [7, 6, 5, 4, 3, 2, 1],
                [9, 7, 5, 4, 8, 7, 3],
            ],
            dtype=np.int32,
        )

        mock_base_recommender = BaseRecommender(URM_train=urm)
        with patch.object(
            mock_base_recommender,
            '_compute_item_score',
            return_value=test_trained_recommender_compute_item_score
        ) as _:
            # arrange
            test_users = [0, 1, 3, 6, 7, 8, 9]
            test_items = None
            test_cutoff = 3
            test_weight = 1  # weight as 1 to have presentation score = frequency of impressions.

            expected_item_scores = np.array([
                [3, 7, 5, 4, 1, 6, 2],
                [6, 1, 1, 6, 4, 1, 4],
                [7, 6, 5, 4, 3, 2, 1],
                [5, 1, 5, 1, 1, 5, 1],
                [1, 3, 5, 7, 4, 6, 2],
                [7, 6, 5, 4, 3, 2, 1],
                [4, 6, 2, 1, 7, 3, 5],
            ], dtype=np.float64)

            rec = CyclingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                trained_recommender=mock_base_recommender,
                seed=seed,
            )

            # act
            rec.fit(weight=test_weight)
            recommendations, scores = rec.recommend(
                user_id_array=test_users,
                items_to_compute=test_items,
                cutoff=test_cutoff,
                remove_seen_flag=False,
                remove_top_pop_flag=False,
                remove_custom_items_flag=False,
                return_scores=True,
            )

            # assert
            assert np.array_equal(expected_item_scores, scores)

    def test_some_users_some_items(
        self, urm: sp.csr_matrix, uim_frequency: sp.csr_matrix,
    ):

        test_trained_recommender_compute_item_score = np.array(
            [
                [1, 6, 3, 2, 3, 5, 4],
                [1, 1, 1, 1, 1, 1, 1],
                [7, 6, 5, 4, 3, 2, 1],

                [1, 1, 1, 1, 1, 1, 1],
                [1, 2, 3, 4, 5, 6, 7],
                [7, 6, 5, 4, 3, 2, 1],
                [9, 7, 5, 4, 8, 7, 3],
            ],
            dtype=np.int32,
        )

        mock_base_recommender = BaseRecommender(URM_train=urm)
        with patch.object(
            mock_base_recommender,
            '_compute_item_score',
            return_value=test_trained_recommender_compute_item_score
        ) as _:
            # arrange
            test_users = [0, 1, 3, 6, 7, 8, 9]
            test_items = [1, 2, 5]
            test_cutoff = 3
            test_weight = 1  # weight as 1 to have presentation score = frequency of impressions.

            expected_item_scores = np.array([
                [np.NINF, 7, 5, np.NINF, np.NINF, 6, np.NINF],
                [np.NINF, 1, 1, np.NINF, np.NINF, 1, np.NINF],
                [np.NINF, 6, 5, np.NINF, np.NINF, 2, np.NINF],
                [np.NINF, 1, 5, np.NINF, np.NINF, 5, np.NINF],
                [np.NINF, 3, 5, np.NINF, np.NINF, 6, np.NINF],
                [np.NINF, 6, 5, np.NINF, np.NINF, 2, np.NINF],
                [np.NINF, 6, 2, np.NINF, np.NINF, 3, np.NINF],
            ], dtype=np.float64)

            rec = CyclingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                trained_recommender=mock_base_recommender,
                seed=seed,
            )

            # act
            rec.fit(weight=test_weight)
            recommendations, scores = rec.recommend(
                user_id_array=test_users,
                items_to_compute=test_items,
                cutoff=test_cutoff,
                remove_seen_flag=False,
                remove_top_pop_flag=False,
                remove_custom_items_flag=False,
                return_scores=True,
            )

            # assert
            assert np.allclose(expected_item_scores, scores)

    def test_some_users_all_items(
        self, urm: sp.csr_matrix, uim_frequency: sp.csr_matrix,
    ):
        test_trained_recommender_compute_item_score = np.array(
            [
                [1, 6, 3, 2, 3, 5, 4],
                [1, 1, 1, 1, 1, 1, 1],
                [7, 6, 5, 4, 3, 2, 1],

                [1, 1, 1, 1, 1, 1, 1],
                [1, 2, 3, 4, 5, 6, 7],
                [7, 6, 5, 4, 3, 2, 1],
                [9, 7, 5, 4, 8, 7, 3],
            ],
            dtype=np.int32,
        )

        mock_base_recommender = BaseRecommender(URM_train=urm)
        with patch.object(
            mock_base_recommender,
            '_compute_item_score',
            return_value=test_trained_recommender_compute_item_score
        ) as _:
            # arrange
            test_users = [0, 1, 3, 6, 7, 8, 9]
            test_items = [0, 1, 2, 3, 4, 5, 6]
            test_cutoff = 3
            test_weight = 1  # weight as 1 to have presentation score = frequency of impressions.

            expected_item_scores = np.array([
                [3, 7, 5, 4, 1, 6, 2],
                [6, 1, 1, 6, 4, 1, 4],
                [7, 6, 5, 4, 3, 2, 1],
                [5, 1, 5, 1, 1, 5, 1],
                [1, 3, 5, 7, 4, 6, 2],
                [7, 6, 5, 4, 3, 2, 1],
                [4, 6, 2, 1, 7, 3, 5],
            ], dtype=np.float64)

            rec = CyclingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                trained_recommender=mock_base_recommender,
                seed=seed,
            )

            # act
            rec.fit(weight=test_weight)
            recommendations, scores = rec.recommend(
                user_id_array=test_users,
                items_to_compute=test_items,
                cutoff=test_cutoff,
                remove_seen_flag=False,
                remove_top_pop_flag=False,
                remove_custom_items_flag=False,
                return_scores=True,
            )

            # assert
            assert np.array_equal(expected_item_scores, scores)
