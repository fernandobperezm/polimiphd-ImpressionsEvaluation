import numpy as np
import scipy.sparse as sp

from impression_recommenders.heuristics.latest_impressions import LastImpressionsRecommender


class TestLastImpressionsRecommender:
    def test_all_users_no_items(
        self, urm: sp.csr_matrix, uim_timestamp: sp.csr_matrix, uim_position: sp.csr_matrix,
    ):
        # arrange
        test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_items = None
        test_cutoff = 3

        expected_recommendations = np.array([
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

        expected_item_scores = np.array([
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

        rec = LastImpressionsRecommender(
            urm_train=urm,
            uim_timestamp=uim_timestamp,
            uim_position=uim_position,
        )

        # act
        rec.fit()
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
        assert np.array_equal(expected_recommendations, recommendations)

    def test_all_users_some_items(
        self, urm: sp.csr_matrix, uim_timestamp: sp.csr_matrix, uim_position: sp.csr_matrix,
    ):
        # arrange
        test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_items = [1, 2, 5]
        test_cutoff = 2

        expected_recommendations = np.array([
            [2, 1],
            [],  # User without impressions on items 1, 2, 5
            [],  # User without impressions on items 1, 2, 5
            [],  # User without impressions
            [1, 2],
            [2, 5],
            [2, 5],
            [5, 2],
            [],  # User without impressions
            [1],
        ], dtype=object)

        expected_item_scores = np.array([
            [np.NINF, -2., 0., np.NINF, np.NINF, np.NINF, np.NINF],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],  # User without impressions on items 1, # 2, 5
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],  # User without impressions on items 1, 2, 5
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],  # User without impressions
            [np.NINF, 0., -2., np.NINF, np.NINF, np.NINF, np.NINF],
            [np.NINF, np.NINF, 0., np.NINF, np.NINF, -2., np.NINF],
            [np.NINF, np.NINF, 0., np.NINF, np.NINF, -2., np.NINF],
            [np.NINF, np.NINF, -2., np.NINF, np.NINF, 0., np.NINF],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],  # User without impressions
            [np.NINF, 0., np.NINF, np.NINF, np.NINF, np.NINF, np.NINF]
        ], dtype=np.float32)

        rec = LastImpressionsRecommender(
            urm_train=urm,
            uim_timestamp=uim_timestamp,
            uim_position=uim_position,
        )

        # act
        rec.fit()
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
        assert np.array_equal(expected_recommendations, recommendations)

    def test_all_users_all_items(
        self, urm: sp.csr_matrix, uim_timestamp: sp.csr_matrix, uim_position: sp.csr_matrix,
    ):
        # arrange
        test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_items = [0, 1, 2, 3, 4, 5, 6]
        test_cutoff = 3

        expected_recommendations = np.array([
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

        expected_item_scores = np.array([
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

        rec = LastImpressionsRecommender(
            urm_train=urm,
            uim_timestamp=uim_timestamp,
            uim_position=uim_position,
        )

        # act
        rec.fit()
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
        assert np.array_equal(expected_recommendations, recommendations)

    def test_some_users_no_items(
        self, urm: sp.csr_matrix, uim_timestamp: sp.csr_matrix, uim_position: sp.csr_matrix,
    ):
        # arrange
        test_users = [0, 1, 3, 6, 7, 8, 9]
        test_items = None
        test_cutoff = 3

        expected_recommendations = np.array([
            [2, 3, 1],
            [0, 3, 4],
            [],  # User without impressions
            [2, 0, 5],
            [5, 3, 2],
            [],  # User without impressions
            [1, 6, 4],
        ], dtype=object)

        expected_item_scores = np.array([
            [np.NINF, -2., 0., -1, np.NINF, np.NINF, np.NINF],
            [0., np.NINF, np.NINF, -1., -2., np.NINF, np.NINF],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],  # User without impressions
            [-1., np.NINF, 0., np.NINF, np.NINF, -2., np.NINF],
            [np.NINF, np.NINF, -2., -1., np.NINF, 0., np.NINF],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],  # User without impressions
            [np.NINF, 0., np.NINF, np.NINF, -2., np.NINF, -1.],
        ], dtype=np.float32)

        rec = LastImpressionsRecommender(
            urm_train=urm,
            uim_timestamp=uim_timestamp,
            uim_position=uim_position,
        )

        # act
        rec.fit()
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
        assert np.array_equal(expected_recommendations, recommendations)

    def test_some_users_some_items(
        self, urm: sp.csr_matrix, uim_timestamp: sp.csr_matrix, uim_position: sp.csr_matrix,
    ):
        # arrange
        test_users = [0, 1, 3, 6, 7, 8, 9]
        test_items = [1, 2, 5]
        test_cutoff = 2

        expected_recommendations = np.array([
            [2, 1],
            [],  # User without impressions on items 1, 2, 5
            [],  # User without impressions
            [2, 5],
            [5, 2],
            [],  # User without impressions
            [1],
        ], dtype=object)

        expected_item_scores = np.array([
            [np.NINF, -2., 0., np.NINF, np.NINF, np.NINF, np.NINF],
            # User without impressions on items 1, # 2, 5
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],  # User without impressions
            [np.NINF, np.NINF, 0., np.NINF, np.NINF, -2., np.NINF],
            [np.NINF, np.NINF, -2., np.NINF, np.NINF, 0., np.NINF],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],  # User without impressions
            [np.NINF, 0., np.NINF, np.NINF, np.NINF, np.NINF, np.NINF]
        ], dtype=np.float32)

        rec = LastImpressionsRecommender(
            urm_train=urm,
            uim_timestamp=uim_timestamp,
            uim_position=uim_position,
        )

        # act
        rec.fit()
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
        assert np.array_equal(expected_recommendations, recommendations)

    def test_some_users_all_items(
        self, urm: sp.csr_matrix, uim_timestamp: sp.csr_matrix, uim_position: sp.csr_matrix,
    ):
        # arrange
        test_users = [0, 1, 3, 6, 7, 8, 9]
        test_items = [0, 1, 2, 3, 4, 5, 6]
        test_cutoff = 3

        expected_recommendations = np.array([
            [2, 3, 1],
            [0, 3, 4],
            [],  # User without impressions
            [2, 0, 5],
            [5, 3, 2],
            [],  # User without impressions
            [1, 6, 4],
        ], dtype=object)

        expected_item_scores = np.array([
            [np.NINF, -2., 0., -1, np.NINF, np.NINF, np.NINF],
            [0., np.NINF, np.NINF, -1., -2., np.NINF, np.NINF],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],  # User without impressions
            [-1., np.NINF, 0., np.NINF, np.NINF, -2., np.NINF],
            [np.NINF, np.NINF, -2., -1., np.NINF, 0., np.NINF],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],  # User without impressions
            [np.NINF, 0., np.NINF, np.NINF, -2., np.NINF, -1.],
        ], dtype=np.float32)

        rec = LastImpressionsRecommender(
            urm_train=urm,
            uim_timestamp=uim_timestamp,
            uim_position=uim_position,
        )

        # act
        rec.fit()
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
        assert np.array_equal(expected_recommendations, recommendations)
