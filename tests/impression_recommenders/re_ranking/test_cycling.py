from typing import Literal

from mock import patch

import numpy as np
import scipy.sparse as sp

from impression_recommenders.re_ranking.cycling import CyclingRecommender
from tests.conftest import seed
from Recommenders.BaseRecommender import BaseRecommender


class TestCyclingRecommender:
    def test_fit_is_correct(
        self, urm: sp.csr_matrix, uim_frequency: sp.csr_matrix,
    ):
        # arrange
        test_recommender = BaseRecommender(URM_train=urm)
        test_weight = 2
        test_sign: Literal[1] = 1

        expected_presentation_scores = np.array([
                [1, 1, 1, 1, 0, 1, 0],
                [1, 0, 0, 1, 1, 0, 1],
                [1, 0, 0, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 1, 1],
                [1, 1, 2, 1, 1, 2, 0],
                [1, 0, 1, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 1],
            ],
            dtype=np.float32
        )

        rec = CyclingRecommender(
            urm_train=urm,
            uim_frequency=uim_frequency,
            trained_recommender=test_recommender,
            seed=seed,
        )

        # act
        rec.fit(weight=test_weight, sign=test_sign)

        # assert
        # For this particular recommender, we cannot test recommendations, as there might be several ties (same
        # timestamp for two impressions) and the .recommend handles ties in a non-deterministic way.
        assert np.allclose(
            rec._matrix_presentation_scores.toarray(),
            expected_presentation_scores
        )

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
            test_sign: Literal[-1] = -1

            expected_item_scores = np.array([
                [5., 4., 2., 1., 6., 3., 7.],
                [1., 5., 6., 2., 3., 7., 4.],
                [1., 4., 5., 2., 6., 7., 3.],
                [7., 6., 5., 4., 3., 2., 1.],
                [6., 4., 2., 3., 7., 5., 1.],
                [4., 6., 1., 3., 5., 2., 7.],
                [1., 4., 2., 5., 6., 3., 7.],
                [6., 4., 2., 1., 5., 3., 7.],
                [7., 6., 5., 4., 3., 2., 1.],
                [7., 2., 5., 4., 3., 6., 1.],
            ], dtype=np.float32)

            rec = CyclingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                trained_recommender=mock_base_recommender,
                seed=seed,
            )

            # act
            rec.fit(weight=test_weight, sign=test_sign)
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
            assert np.allclose(expected_item_scores, scores)

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
            test_sign: Literal[-1] = -1

            expected_item_scores = np.array([
                [np.NINF, 4., 2., np.NINF, np.NINF, 3., np.NINF],
                [np.NINF, 5., 6., np.NINF, np.NINF, 7., np.NINF],
                [np.NINF, 4., 5., np.NINF, np.NINF, 7., np.NINF],
                [np.NINF, 6., 5., np.NINF, np.NINF, 2., np.NINF],
                [np.NINF, 4., 2., np.NINF, np.NINF, 5., np.NINF],
                [np.NINF, 6., 1., np.NINF, np.NINF, 2., np.NINF],
                [np.NINF, 4., 2., np.NINF, np.NINF, 3., np.NINF],
                [np.NINF, 4., 2., np.NINF, np.NINF, 3., np.NINF],
                [np.NINF, 6., 5., np.NINF, np.NINF, 2., np.NINF],
                [np.NINF, 2., 5., np.NINF, np.NINF, 6., np.NINF],
            ], dtype=np.float64)

            rec = CyclingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                trained_recommender=mock_base_recommender,
                seed=seed,
            )

            # act
            rec.fit(weight=test_weight, sign=test_sign)
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
            test_sign: Literal[-1] = -1

            expected_item_scores = np.array([
                [5., 4., 2., 1., 6., 3., 7.],
                [1., 5., 6., 2., 3., 7., 4.],
                [1., 4., 5., 2., 6., 7., 3.],
                [7., 6., 5., 4., 3., 2., 1.],
                [6., 4., 2., 3., 7., 5., 1.],
                [4., 6., 1., 3., 5., 2., 7.],
                [1., 4., 2., 5., 6., 3., 7.],
                [6., 4., 2., 1., 5., 3., 7.],
                [7., 6., 5., 4., 3., 2., 1.],
                [7., 2., 5., 4., 3., 6., 1.],
            ], dtype=np.float32)

            rec = CyclingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                trained_recommender=mock_base_recommender,
                seed=seed,
            )

            # act
            rec.fit(weight=test_weight, sign=test_sign)
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
            test_sign: Literal[-1] = -1

            expected_item_scores = np.array([
                [5., 4., 2., 1., 6., 3., 7.],
                [1., 5., 6., 2., 3., 7., 4.],
                [7., 6., 5., 4., 3., 2., 1.],
                [1., 4., 2., 5., 6., 3., 7.],
                [6., 4., 2., 1., 5., 3., 7.],
                [7., 6., 5., 4., 3., 2., 1.],
                [7., 2., 5., 4., 3., 6., 1.],
            ], dtype=np.float32)

            rec = CyclingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                trained_recommender=mock_base_recommender,
                seed=seed,
            )

            # act
            rec.fit(weight=test_weight, sign=test_sign)
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
            test_sign: Literal[-1] = -1

            expected_item_scores = np.array([
                [np.NINF, 4., 2., np.NINF, np.NINF, 3., np.NINF],
                [np.NINF, 5., 6., np.NINF, np.NINF, 7., np.NINF],
                [np.NINF, 6., 5., np.NINF, np.NINF, 2., np.NINF],
                [np.NINF, 4., 2., np.NINF, np.NINF, 3., np.NINF],
                [np.NINF, 4., 2., np.NINF, np.NINF, 3., np.NINF],
                [np.NINF, 6., 5., np.NINF, np.NINF, 2., np.NINF],
                [np.NINF, 2., 5., np.NINF, np.NINF, 6., np.NINF],
            ], dtype=np.float64)

            rec = CyclingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                trained_recommender=mock_base_recommender,
                seed=seed,
            )

            # act
            rec.fit(weight=test_weight, sign=test_sign)
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
            test_sign: Literal[-1] = -1

            expected_item_scores = np.array([
                [5., 4., 2., 1., 6., 3., 7.],
                [1., 5., 6., 2., 3., 7., 4.],
                [7., 6., 5., 4., 3., 2., 1.],
                [1., 4., 2., 5., 6., 3., 7.],
                [6., 4., 2., 1., 5., 3., 7.],
                [7., 6., 5., 4., 3., 2., 1.],
                [7., 2., 5., 4., 3., 6., 1.],
            ], dtype=np.float32)

            rec = CyclingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                trained_recommender=mock_base_recommender,
                seed=seed,
            )

            # act
            rec.fit(weight=test_weight, sign=test_sign)
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
