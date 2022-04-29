import numpy as np
import scipy.sparse as sp

from impression_recommenders.heuristics.frequency_and_recency import RecencyRecommender, FrequencyRecencyRecommender, \
    T_SIGN


class TestRecencyRecommender:
    def test_all_users_no_items(
        self, urm: sp.csr_matrix, uim_timestamp: sp.csr_matrix,
    ):
        # arrange
        test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_items = None
        test_cutoff = 3
        test_sign_recency: T_SIGN = 1

        expected_item_scores = np.array([
            [1641017209, 1641108375, 1641108375, 1641108375, np.NINF, 1641037317,  np.NINF],
            [1641111467, np.NINF, np.NINF, 1641111467, 1641111467, np.NINF, 1641031516],
            [1641070743, np.NINF, np.NINF, 1641070743, np.NINF, np.NINF, 1641070743],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF,  np.NINF],
            [1641092129, 1641148079, 1641148079, 1641092129, np.NINF, 1641037173, 1641148079],
            [1641152250, 1641072907, 1641164761, 1641152250, 1641164761, 1641164761,  np.NINF],
            [1641075505, np.NINF, 1641075505, np.NINF, np.NINF, 1641075505,  np.NINF],
            [np.NINF, 1641110800, 1641161688, 1641161688, 1641110800, 1641161688,  np.NINF],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF,  np.NINF],
            [np.NINF, 1641005198, np.NINF, np.NINF, 1641005198, np.NINF, 1641005198]
        ], dtype=np.float64)

        rec = RecencyRecommender(
            urm_train=urm,
            uim_timestamp=uim_timestamp,
        )

        # act
        rec.fit(sign_recency=test_sign_recency)
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
        self, urm: sp.csr_matrix, uim_timestamp: sp.csr_matrix,
    ):
        # arrange
        test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_items = [1, 2, 5]
        test_cutoff = 3
        test_sign_recency: T_SIGN = 1
        expected_item_scores = np.array([
            [np.NINF, 1641108375, 1641108375, np.NINF, np.NINF, 1641037317, np.NINF],

            # User without impressions on items 1, 2, 5
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],
            # User without impressions on items 1, 2, 5
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],
            # User without impressions
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],

            [np.NINF, 1641148079, 1641148079, np.NINF, np.NINF, 1641037173, np.NINF],
            [np.NINF, 1641072907, 1641164761, np.NINF, np.NINF, 1641164761, np.NINF],
            [np.NINF, np.NINF, 1641075505, np.NINF, np.NINF, 1641075505, np.NINF],
            [np.NINF, 1641110800, 1641161688, np.NINF, np.NINF, 1641161688, np.NINF],

            # User without impressions
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],

            [np.NINF, 1641005198, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF]
        ], dtype=np.float64)

        rec = RecencyRecommender(
            urm_train=urm,
            uim_timestamp=uim_timestamp,
        )

        # act
        rec.fit(sign_recency=test_sign_recency)
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
        self, urm: sp.csr_matrix, uim_timestamp: sp.csr_matrix,
    ):
        # arrange
        test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_items = [0, 1, 2, 3, 4, 5, 6]
        test_cutoff = 3
        test_sign_recency: T_SIGN = 1
        expected_item_scores = np.array([
            [1641017209, 1641108375, 1641108375, 1641108375, np.NINF, 1641037317, np.NINF],
            [1641111467, np.NINF, np.NINF, 1641111467, 1641111467, np.NINF, 1641031516],
            [1641070743, np.NINF, np.NINF, 1641070743, np.NINF, np.NINF, 1641070743],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],
            [1641092129, 1641148079, 1641148079, 1641092129, np.NINF, 1641037173, 1641148079],
            [1641152250, 1641072907, 1641164761, 1641152250, 1641164761, 1641164761, np.NINF],
            [1641075505, np.NINF, 1641075505, np.NINF, np.NINF, 1641075505, np.NINF],
            [np.NINF, 1641110800, 1641161688, 1641161688, 1641110800, 1641161688, np.NINF],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],
            [np.NINF, 1641005198, np.NINF, np.NINF, 1641005198, np.NINF, 1641005198]
        ], dtype=np.float64)

        rec = RecencyRecommender(
            urm_train=urm,
            uim_timestamp=uim_timestamp,
        )

        # act
        rec.fit(sign_recency=test_sign_recency)
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
        self, urm: sp.csr_matrix, uim_timestamp: sp.csr_matrix,
    ):
        # arrange
        test_users = [0, 1, 3, 6, 7, 8, 9]
        test_items = None
        test_cutoff = 3
        test_sign_recency: T_SIGN = 1
        expected_item_scores = np.array([
            [1641017209, 1641108375, 1641108375, 1641108375, np.NINF, 1641037317, np.NINF],
            [1641111467, np.NINF, np.NINF, 1641111467, 1641111467, np.NINF, 1641031516],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],
            [1641075505, np.NINF, 1641075505, np.NINF, np.NINF, 1641075505, np.NINF],
            [np.NINF, 1641110800, 1641161688, 1641161688, 1641110800, 1641161688, np.NINF],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],
            [np.NINF, 1641005198, np.NINF, np.NINF, 1641005198, np.NINF, 1641005198]
        ], dtype=np.float64)

        rec = RecencyRecommender(
            urm_train=urm,
            uim_timestamp=uim_timestamp,
        )

        # act
        rec.fit(sign_recency=test_sign_recency)
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

    def test_some_users_some_items(
        self, urm: sp.csr_matrix, uim_timestamp: sp.csr_matrix,
    ):
        # arrange
        test_users = [0, 1, 3, 6, 7, 8, 9]
        test_items = [1, 2, 5]
        test_cutoff = 3
        test_sign_recency: T_SIGN = 1
        expected_item_scores = np.array([
            [np.NINF, 1641108375, 1641108375, np.NINF, np.NINF, 1641037317, np.NINF],

            # User without impressions on items 1, 2, 5
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],
            # User without impressions
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],

            [np.NINF, np.NINF, 1641075505, np.NINF, np.NINF, 1641075505, np.NINF],
            [np.NINF, 1641110800, 1641161688, np.NINF, np.NINF, 1641161688, np.NINF],

            # User without impressions
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],

            [np.NINF, 1641005198, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF]
        ], dtype=np.float64)

        rec = RecencyRecommender(
            urm_train=urm,
            uim_timestamp=uim_timestamp,
        )

        # act
        rec.fit(sign_recency=test_sign_recency)
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
        self, urm: sp.csr_matrix, uim_timestamp: sp.csr_matrix,
    ):
        # arrange
        test_users = [0, 1, 3, 6, 7, 8, 9]
        test_items = [0, 1, 2, 3, 4, 5, 6]
        test_cutoff = 3
        test_sign_recency: T_SIGN = 1
        expected_item_scores = np.array([
            [1641017209, 1641108375, 1641108375, 1641108375, np.NINF, 1641037317, np.NINF],
            [1641111467, np.NINF, np.NINF, 1641111467, 1641111467, np.NINF, 1641031516],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],
            [1641075505, np.NINF, 1641075505, np.NINF, np.NINF, 1641075505, np.NINF],
            [np.NINF, 1641110800, 1641161688, 1641161688, 1641110800, 1641161688, np.NINF],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],
            [np.NINF, 1641005198, np.NINF, np.NINF, 1641005198, np.NINF, 1641005198]
        ], dtype=np.float64)

        rec = RecencyRecommender(
            urm_train=urm,
            uim_timestamp=uim_timestamp,
        )

        # act
        rec.fit(sign_recency=test_sign_recency)
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


class TestFrequencyRecencyRecommender:
    def test_all_users_no_items(
        self, urm: sp.csr_matrix, uim_timestamp: sp.csr_matrix, uim_frequency: sp.csr_matrix,
    ):
        # arrange
        test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_items = None
        test_cutoff = 3
        test_sign_recency: T_SIGN = 1
        test_sign_frequency: T_SIGN = 1
        expected_item_scores = np.array([
            [3., 5., 6., 7., np.NINF, 4., np.NINF],
            [6., np.NINF, np.NINF, 7., 5., np.NINF, 4.],
            [5., np.NINF, np.NINF, 6., np.NINF, np.NINF, 7.],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],
            [3., 5., 6., 4., np.NINF, 2., 7.],
            [3., 2., 7., 5., 4., 6., np.NINF],
            [5., np.NINF, 6., np.NINF, np.NINF, 7., np.NINF],
            [np.NINF, 3., 5., 7., 4., 6., np.NINF],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],
            [np.NINF, 5., np.NINF, np.NINF, 6., np.NINF, 7.]
        ], dtype=np.float32)

        rec = FrequencyRecencyRecommender(
            urm_train=urm,
            uim_timestamp=uim_timestamp,
            uim_frequency=uim_frequency,
        )

        # act
        rec.fit(sign_recency=test_sign_recency, sign_frequency=test_sign_frequency)
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
        self, urm: sp.csr_matrix, uim_timestamp: sp.csr_matrix, uim_frequency: sp.csr_matrix,
    ):
        # arrange
        test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_items = [1, 2, 5]
        test_cutoff = 3
        test_sign_recency: T_SIGN = 1
        test_sign_frequency: T_SIGN = 1
        expected_item_scores = np.array([
            [np.NINF, 5., 6., np.NINF, np.NINF, 4., np.NINF],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],
            [np.NINF, 5., 6., np.NINF, np.NINF, 2., np.NINF],
            [np.NINF, 2., 7., np.NINF, np.NINF, 6., np.NINF],
            [np.NINF, np.NINF, 6., np.NINF, np.NINF, 7., np.NINF],
            [np.NINF, 3., 5., np.NINF, np.NINF, 6., np.NINF],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],
            [np.NINF, 5., np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],
        ], dtype=np.float32)

        rec = FrequencyRecencyRecommender(
            urm_train=urm,
            uim_timestamp=uim_timestamp,
            uim_frequency=uim_frequency,
        )

        # act
        rec.fit(sign_recency=test_sign_recency, sign_frequency=test_sign_frequency)
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
        self, urm: sp.csr_matrix, uim_timestamp: sp.csr_matrix, uim_frequency: sp.csr_matrix,
    ):
        # arrange
        test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_items = [0, 1, 2, 3, 4, 5, 6]
        test_cutoff = 3
        test_sign_recency: T_SIGN = 1
        test_sign_frequency: T_SIGN = 1
        expected_item_scores = np.array([
            [3., 5., 6., 7., np.NINF, 4., np.NINF],
            [6., np.NINF, np.NINF, 7., 5., np.NINF, 4.],
            [5., np.NINF, np.NINF, 6., np.NINF, np.NINF, 7.],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],
            [3., 5., 6., 4., np.NINF, 2., 7.],
            [3., 2., 7., 5., 4., 6., np.NINF],
            [5., np.NINF, 6., np.NINF, np.NINF, 7., np.NINF],
            [np.NINF, 3., 5., 7., 4., 6., np.NINF],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],
            [np.NINF, 5., np.NINF, np.NINF, 6., np.NINF, 7.]
        ], dtype=np.float32)

        rec = FrequencyRecencyRecommender(
            urm_train=urm,
            uim_timestamp=uim_timestamp,
            uim_frequency=uim_frequency,
        )

        # act
        rec.fit(sign_recency=test_sign_recency, sign_frequency=test_sign_frequency)
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
        self, urm: sp.csr_matrix, uim_timestamp: sp.csr_matrix, uim_frequency: sp.csr_matrix,
    ):
        # arrange
        test_users = [0, 1, 3, 6, 7, 8, 9]
        test_items = None
        test_cutoff = 3
        test_sign_recency: T_SIGN = 1
        test_sign_frequency: T_SIGN = 1
        expected_item_scores = np.array([
            [3., 5., 6., 7., np.NINF, 4., np.NINF],
            [6., np.NINF, np.NINF, 7., 5., np.NINF, 4.],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],
            [5., np.NINF, 6., np.NINF, np.NINF, 7., np.NINF],
            [np.NINF, 3., 5., 7., 4., 6., np.NINF],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],
            [np.NINF, 5., np.NINF, np.NINF, 6., np.NINF, 7.]
        ], dtype=np.float32)

        rec = FrequencyRecencyRecommender(
            urm_train=urm,
            uim_timestamp=uim_timestamp,
            uim_frequency=uim_frequency,
        )

        # act
        rec.fit(sign_recency=test_sign_recency, sign_frequency=test_sign_frequency)
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

    def test_some_users_some_items(
        self, urm: sp.csr_matrix, uim_timestamp: sp.csr_matrix, uim_frequency: sp.csr_matrix,
    ):
        # arrange
        test_users = [0, 1, 3, 6, 7, 8, 9]
        test_items = [1, 2, 5]
        test_cutoff = 3
        test_sign_recency: T_SIGN = 1
        test_sign_frequency: T_SIGN = 1
        expected_item_scores = np.array([
            [np.NINF, 5., 6., np.NINF, np.NINF, 4., np.NINF],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],
            [np.NINF, np.NINF, 6., np.NINF, np.NINF, 7., np.NINF],
            [np.NINF, 3., 5., np.NINF, np.NINF, 6., np.NINF],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],
            [np.NINF, 5., np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],
        ], dtype=np.float32)

        rec = FrequencyRecencyRecommender(
            urm_train=urm,
            uim_timestamp=uim_timestamp,
            uim_frequency=uim_frequency,
        )

        # act
        rec.fit(sign_recency=test_sign_recency, sign_frequency=test_sign_frequency)
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
        self, urm: sp.csr_matrix, uim_timestamp: sp.csr_matrix, uim_frequency: sp.csr_matrix,
    ):
        # arrange
        test_users = [0, 1, 3, 6, 7, 8, 9]
        test_items = [0, 1, 2, 3, 4, 5, 6]
        test_cutoff = 3
        test_sign_recency: T_SIGN = 1
        test_sign_frequency: T_SIGN = 1
        expected_item_scores = np.array([
            [3., 5., 6., 7., np.NINF, 4., np.NINF],
            [6., np.NINF, np.NINF, 7., 5., np.NINF, 4.],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],
            [5., np.NINF, 6., np.NINF, np.NINF, 7., np.NINF],
            [np.NINF, 3., 5., 7., 4., 6., np.NINF],
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF],
            [np.NINF, 5., np.NINF, np.NINF, 6., np.NINF, 7.]
        ], dtype=np.float64)

        rec = FrequencyRecencyRecommender(
            urm_train=urm,
            uim_timestamp=uim_timestamp,
            uim_frequency=uim_frequency,
        )

        # act
        rec.fit(sign_recency=test_sign_recency, sign_frequency=test_sign_frequency)
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
