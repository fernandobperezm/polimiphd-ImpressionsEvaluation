from mock import patch
import numpy as np
import scipy.sparse as sp

from impression_recommenders.re_ranking.dithering import DitheringRecommender
from tests.conftest import seed
from Recommenders.BaseRecommender import BaseRecommender


class TestDitheringRecommender:
    def test_all_users_no_items(
        self, urm: sp.csr_matrix,
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
            test_epsilon = np.e  # epsilon is e (euler's constant) to have variance as 1.

            expected_item_scores = np.array([
                [-2.0393271210549409e+00, 1.9641606530383462e+00,
                 3.6600608835509396e-01, 1.5023951412871326e+00,
                 1.6939727918703777e+00, 2.0653165831437441e+00,
                 1.0909460514040021e+00],
                [5.6577465866342447e-01, -2.4610815472552749e-01,
                 3.4414406748927910e-03, -1.4176262311667958e+00,
                 -1.0636458449794775e-01, -2.9512158276358369e-01,
                 1.6421783667474224e-01],
                [-7.8052643948284617e-01, 4.8013023876037786e-02,
                 3.4040519904603528e+00, 1.6798949623461141e+00,
                 1.7294125119013291e+00, 3.3421912976091108e+00,
                 1.3802139517369862e+00],
                [1.9504541911788671e+00, 3.0549645951302367e+00,
                 2.6544019328612309e+00, 2.8061155109144398e-01,
                 1.0768122028333977e+00, 3.0730733820046396e-01,
                 -3.1123324576516120e-01],
                [1.7766284944939010e+00, 4.7094017949213629e-01,
                 5.1734407828047724e-02, 1.1818082923129325e-01,
                 1.0039179231881570e+00, 3.5015985681899648e+00,
                 -9.4045791142081459e-01],
                [1.7524909490095990e+00, 1.7347325035357810e+00,
                 1.3880215890478940e+00, -2.4421666215992799e-01,
                 1.8607111844714479e+00, 2.3319813508912968e+00,
                 1.8380693582327527e+00],
                [5.5019718557080778e-03, -4.7863672346544378e-01,
                 -1.5036173930472192e+00, 4.7879567722905142e-01,
                 2.3912271635999613e+00, -5.0430499783587303e-02,
                 7.1992214868495941e-01],
                [-9.8885327057612604e-03, 1.3271559623616014e+00,
                 7.5733498662355969e-01, 1.3294075937357359e+00,
                 1.3749274737174262e+00, 1.5838031872145168e+00,
                 1.6207373140777870e-01],
                [3.8210357008921227e+00, 8.6188835050542623e-01,
                 -3.6042929325774731e-01, 1.1771252336150819e+00,
                 9.9233177037740639e-01, -1.0422138233159395e-01,
                 1.4713126380383208e+00],
                [5.9845907675131405e-01, 3.3610515492967656e+00,
                 8.1842852016970791e-01, 1.2277051710001428e+00,
                 1.9112160002183629e+00, 1.9633214208054413e+00,
                 8.9635581814181065e-01],
            ], dtype=np.float64)

            rec = DitheringRecommender(
                urm_train=urm,
                trained_recommender=mock_base_recommender,
                seed=seed,
            )

            # act
            rec.fit(epsilon=test_epsilon)
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
        test_trained_recommender_compute_item_score = np.array(
            [
                [np.NINF, 6, 3, np.NINF, np.NINF, 5, np.NINF],
                [np.NINF, 1, 1, np.NINF, np.NINF, 1, np.NINF],
                [np.NINF, 2, 3, np.NINF, np.NINF, 6, np.NINF],
                [np.NINF, 6, 5, np.NINF, np.NINF, 2, np.NINF],
                [np.NINF, 7, 5, np.NINF, np.NINF, 7, np.NINF],

                [np.NINF, 6, 3, np.NINF, np.NINF, 5, np.NINF],
                [np.NINF, 1, 1, np.NINF, np.NINF, 1, np.NINF],
                [np.NINF, 2, 3, np.NINF, np.NINF, 6, np.NINF],
                [np.NINF, 6, 5, np.NINF, np.NINF, 2, np.NINF],
                [np.NINF, 7, 5, np.NINF, np.NINF, 7, np.NINF],
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
            test_epsilon = np.e  # epsilon is e (euler's constant) to have variance as 1.

            expected_item_scores = np.array([
                [np.NINF, 1.9641606530383462, 0.8768317121210845,
                 np.NINF, np.NINF, 2.065316583143744,
                 np.NINF],
                [np.NINF, 1.3633297577085728, 1.612879353108993,
                 np.NINF, np.NINF, 1.3143163296705165,
                 np.NINF],
                [np.NINF, 0.9643037557501928, 4.097199171020298,
                 np.NINF, np.NINF, 3.4963419774363693,
                 np.NINF],
                [np.NINF, 3.2091152749574947, 2.8367234896551854,
                 np.NINF, np.NINF, 1.223598070074619,
                 np.NINF],
                [np.NINF, 0.8764052876003007, 0.5625600315940382,
                 np.NINF, np.NINF, 3.9070636762981295,
                 np.NINF],
                [np.NINF, 1.734732503535781, 1.8988472128138845,
                 np.NINF, np.NINF, 2.3319813508912968,
                 np.NINF],
                [np.NINF, 1.1308011889686564, 0.1058205193868811,
                 np.NINF, np.NINF, 1.559007412650513,
                 np.NINF],
                [np.NINF, 2.2434466942357565, 1.4504821671835049,
                 np.NINF, np.NINF, 1.7379538670417751,
                 np.NINF],
                [np.NINF, 1.0160390303326845, -0.1781077364637926,
                 np.NINF, np.NINF, 0.812069349542561,
                 np.NINF],
                [np.NINF, 3.76651665740493, 1.3292541439356984,
                 np.NINF, np.NINF, 2.368786528913606,
                 np.NINF],
            ], dtype=np.float32)

            rec = DitheringRecommender(
                urm_train=urm,
                trained_recommender=mock_base_recommender,
                seed=seed,
            )

            # act
            rec.fit(epsilon=test_epsilon)
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
        self, urm: sp.csr_matrix,
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
            test_epsilon = np.e  # epsilon is e (euler's constant) to have variance as 1.

            expected_item_scores = np.array([
                [-2.0393271210549409e+00, 1.9641606530383462e+00,
                 3.6600608835509396e-01, 1.5023951412871326e+00,
                 1.6939727918703777e+00, 2.0653165831437441e+00,
                 1.0909460514040021e+00],
                [5.6577465866342447e-01, -2.4610815472552749e-01,
                 3.4414406748927910e-03, -1.4176262311667958e+00,
                 -1.0636458449794775e-01, -2.9512158276358369e-01,
                 1.6421783667474224e-01],
                [-7.8052643948284617e-01, 4.8013023876037786e-02,
                 3.4040519904603528e+00, 1.6798949623461141e+00,
                 1.7294125119013291e+00, 3.3421912976091108e+00,
                 1.3802139517369862e+00],
                [1.9504541911788671e+00, 3.0549645951302367e+00,
                 2.6544019328612309e+00, 2.8061155109144398e-01,
                 1.0768122028333977e+00, 3.0730733820046396e-01,
                 -3.1123324576516120e-01],
                [1.7766284944939010e+00, 4.7094017949213629e-01,
                 5.1734407828047724e-02, 1.1818082923129325e-01,
                 1.0039179231881570e+00, 3.5015985681899648e+00,
                 -9.4045791142081459e-01],
                [1.7524909490095990e+00, 1.7347325035357810e+00,
                 1.3880215890478940e+00, -2.4421666215992799e-01,
                 1.8607111844714479e+00, 2.3319813508912968e+00,
                 1.8380693582327527e+00],
                [5.5019718557080778e-03, -4.7863672346544378e-01,
                 -1.5036173930472192e+00, 4.7879567722905142e-01,
                 2.3912271635999613e+00, -5.0430499783587303e-02,
                 7.1992214868495941e-01],
                [-9.8885327057612604e-03, 1.3271559623616014e+00,
                 7.5733498662355969e-01, 1.3294075937357359e+00,
                 1.3749274737174262e+00, 1.5838031872145168e+00,
                 1.6207373140777870e-01],
                [3.8210357008921227e+00, 8.6188835050542623e-01,
                 -3.6042929325774731e-01, 1.1771252336150819e+00,
                 9.9233177037740639e-01, -1.0422138233159395e-01,
                 1.4713126380383208e+00],
                [5.9845907675131405e-01, 3.3610515492967656e+00,
                 8.1842852016970791e-01, 1.2277051710001428e+00,
                 1.9112160002183629e+00, 1.9633214208054413e+00,
                 8.9635581814181065e-01],
            ], dtype=np.float32)

            rec = DitheringRecommender(
                urm_train=urm,
                trained_recommender=mock_base_recommender,
                seed=seed,
            )

            # act
            rec.fit(epsilon=test_epsilon)
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
        self, urm: sp.csr_matrix,
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
            test_epsilon = np.e  # epsilon is e (euler's constant) to have variance as 1.

            expected_item_scores = np.array([
                [-2.0393271210549409e+00, 1.9641606530383462e+00,
                 3.6600608835509396e-01, 1.5023951412871326e+00,
                 1.6939727918703777e+00, 2.0653165831437441e+00,
                 1.0909460514040021e+00],
                [5.6577465866342447e-01, -2.4610815472552749e-01,
                 3.4414406748927910e-03, -1.4176262311667958e+00,
                 -1.0636458449794775e-01, -2.9512158276358369e-01,
                 1.6421783667474224e-01],
                [1.1653837095724671e+00, 1.1466253125441475e+00,
                 3.9148776142263433e+00, 1.6798949623461141e+00,
                 1.2185868881353386e+00, 2.2435790089410013e+00,
                 -5.6569619731832699e-01],
                [4.5440421235538016e-03, 1.2632051259021815e+00,
                 1.0449640204271304e+00, -1.1056828100284466e+00,
                 -2.1800085834712102e-02, -3.8583984235948132e-01,
                 -3.1123324576516120e-01],
                [-1.6928165456141209e-01, -2.2220700106780900e-01,
                 5.1734407828047724e-02, 8.1132800979123854e-01,
                 8.2159636639420230e-01, 3.9070636762981295e+00,
                 1.0054522376344988e+00],
                [3.6984010980649122e+00, 1.5805818237085227e+00,
                 1.8988472128138845e+00, 4.4893051840001730e-01,
                 1.8607111844714479e+00, 1.2333690622231870e+00,
                 2.2863144579865252e-01],
                [1.9514121209110213e+00, 9.0765763765444674e-01,
                 -4.0500510437910942e-01, 1.1719428577889968e+00,
                 4.1829866328280163e+00, 1.3358638613363032e+00,
                 7.1992214868495941e-01]
            ], dtype=np.float64)

            rec = DitheringRecommender(
                urm_train=urm,
                trained_recommender=mock_base_recommender,
                seed=seed,
            )

            # act
            rec.fit(epsilon=test_epsilon)
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
        self, urm: sp.csr_matrix, uim_timestamp: sp.csr_matrix, uim_frequency: sp.csr_matrix,
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
            test_epsilon = np.e  # epsilon is e (euler's constant) to have variance as 1.

            expected_item_scores = np.array([
                [np.NINF, 1.9641606530383462e+00,
                 3.6600608835509396e-01, np.NINF,
                 np.NINF, 2.0653165831437441e+00,
                 np.NINF],
                [np.NINF, -2.4610815472552749e-01,
                 3.4414406748927910e-03, np.NINF,
                 np.NINF, -2.9512158276358369e-01,
                 np.NINF],
                [np.NINF, 1.1466253125441475e+00,
                 3.9148776142263433e+00, np.NINF,
                 np.NINF, 2.2435790089410013e+00,
                 np.NINF],
                [np.NINF, 1.2632051259021815e+00,
                 1.0449640204271304e+00, np.NINF,
                 np.NINF, -3.8583984235948132e-01,
                 np.NINF],
                [np.NINF, -2.2220700106780900e-01,
                 5.1734407828047724e-02, np.NINF,
                 np.NINF, 3.9070636762981295e+00,
                 np.NINF],
                [np.NINF, 1.5805818237085227e+00,
                 1.8988472128138845e+00, np.NINF,
                 np.NINF, 1.2333690622231870e+00,
                 np.NINF],
                [np.NINF, 9.0765763765444674e-01,
                 -4.0500510437910942e-01, np.NINF,
                 np.NINF, 1.3358638613363032e+00,
                 np.NINF],
            ], dtype=np.float64)

            rec = DitheringRecommender(
                urm_train=urm,
                trained_recommender=mock_base_recommender,
                seed=seed,
            )

            # act
            rec.fit(epsilon=test_epsilon)
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
            test_epsilon = np.e  # epsilon is e (euler's constant) to have variance as 1.

            expected_item_scores = np.array([
                [-2.0393271210549409e+00, 1.9641606530383462e+00,
                 3.6600608835509396e-01, 1.5023951412871326e+00,
                 1.6939727918703777e+00, 2.0653165831437441e+00,
                 1.0909460514040021e+00],
                [5.6577465866342447e-01, -2.4610815472552749e-01,
                 3.4414406748927910e-03, -1.4176262311667958e+00,
                 -1.0636458449794775e-01, -2.9512158276358369e-01,
                 1.6421783667474224e-01],
                [1.1653837095724671e+00, 1.1466253125441475e+00,
                 3.9148776142263433e+00, 1.6798949623461141e+00,
                 1.2185868881353386e+00, 2.2435790089410013e+00,
                 -5.6569619731832699e-01],
                [4.5440421235538016e-03, 1.2632051259021815e+00,
                 1.0449640204271304e+00, -1.1056828100284466e+00,
                 -2.1800085834712102e-02, -3.8583984235948132e-01,
                 -3.1123324576516120e-01],
                [-1.6928165456141209e-01, -2.2220700106780900e-01,
                 5.1734407828047724e-02, 8.1132800979123854e-01,
                 8.2159636639420230e-01, 3.9070636762981295e+00,
                 1.0054522376344988e+00],
                [3.6984010980649122e+00, 1.5805818237085227e+00,
                 1.8988472128138845e+00, 4.4893051840001730e-01,
                 1.8607111844714479e+00, 1.2333690622231870e+00,
                 2.2863144579865252e-01],
                [1.9514121209110213e+00, 9.0765763765444674e-01,
                 -4.0500510437910942e-01, 1.1719428577889968e+00,
                 4.1829866328280163e+00, 1.3358638613363032e+00,
                 7.1992214868495941e-01]
            ], dtype=np.float64)

            rec = DitheringRecommender(
                urm_train=urm,
                trained_recommender=mock_base_recommender,
                seed=seed,
            )

            # act
            rec.fit(epsilon=test_epsilon)
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
