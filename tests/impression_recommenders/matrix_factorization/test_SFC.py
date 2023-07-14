import torch

import numpy as np
import scipy.sparse as sp
from pytest import fixture

from impressions_evaluation.impression_recommenders.matrix_factorization.SFC import (
    SoftFrequencyCappingRecommender,
    SFCModel,
    FREQUENCY_MODE,
    _get_frequency_num_embeddings,
)
from tests.conftest import seed, NUM_USERS, NUM_ITEMS


TEST_EPOCHS = 10
TEST_FREQUENCY_MODE: FREQUENCY_MODE = "global"
test_frequency_num_bins = 26
test_batch_size = 500
test_embedding_size = 10
test_learning_rate = 1e-1
test_l2_reg = 1e-2
test_scheduler_alpha = 1e-2
test_scheduler_beta = 1e-2
test_device = torch.device("cpu:0")


@fixture
def binned_uim_frequency(
    uim_frequency: sp.csr_matrix,
) -> sp.dok_matrix:
    uim_frequency.data = np.where(
        uim_frequency.data < test_frequency_num_bins,
        uim_frequency.data,
        test_frequency_num_bins,
    )
    uim_frequency_dok = uim_frequency.astype(
        dtype=np.int32,
    ).todok()
    return uim_frequency_dok


class TestSoftFrequencyCappingModel:
    def test_forward_function_global(self):
        # arrange
        test_frequency_mode: FREQUENCY_MODE = "global"
        test_frequency_num_embeddings = _get_frequency_num_embeddings(
            frequency_mode=test_frequency_mode,
            num_users=NUM_USERS,
            num_items=NUM_ITEMS,
        )
        model = SFCModel(
            num_users=NUM_USERS,
            num_items=NUM_ITEMS,
            frequency_mode=test_frequency_mode,
            frequency_num_bins=test_frequency_num_bins,
            frequency_num_embeddings=test_frequency_num_embeddings,
            embedding_size=test_embedding_size,
            device=test_device,
        )
        # Make sure those embeddings have values (the model sets them as 0).
        [
            torch.nn.init.normal_(
                emb.weight,
            )
            for emb in model.embedding_frequencies
        ]
        model.eval()

        test_user_id = [1]
        test_item_id = [2]
        test_freq = [4]
        test_idx_emb = [0]
        test_zero = [0]

        arr_bias = model.embedding_bias.weight.detach().cpu().numpy()
        arr_user_factors = model.embedding_user.weight.detach().cpu().numpy()
        arr_item_factors = model.embedding_item.weight.detach().cpu().numpy()
        arr_frequency_factors = np.vstack(
            [emb.weight.detach().cpu().numpy() for emb in model.embedding_frequencies],
        )

        expected_rating = (
            arr_bias[0]
            + np.dot(
                arr_user_factors[test_user_id, :],
                arr_item_factors[test_item_id, :].T,
            )
            + arr_frequency_factors[test_freq]
        )

        # act
        rating = model(
            torch.tensor(test_user_id),
            torch.tensor(test_item_id),
            torch.tensor(test_freq),
            torch.tensor(test_idx_emb),
            torch.tensor(test_zero),
        )

        # assert
        assert np.allclose(
            expected_rating,
            rating.detach().cpu().numpy(),
        )

    def test_forward_function_user(self):
        # arrange
        test_frequency_mode: FREQUENCY_MODE = "user"
        test_frequency_num_embeddings = _get_frequency_num_embeddings(
            frequency_mode=test_frequency_mode,
            num_users=NUM_USERS,
            num_items=NUM_ITEMS,
        )
        model = SFCModel(
            num_users=NUM_USERS,
            num_items=NUM_ITEMS,
            frequency_mode=test_frequency_mode,
            frequency_num_bins=test_frequency_num_bins,
            frequency_num_embeddings=test_frequency_num_embeddings,
            embedding_size=test_embedding_size,
            device=test_device,
        )
        # Make sure those embeddings have values (the model sets them as 0).
        [
            torch.nn.init.normal_(
                emb.weight,
            )
            for emb in model.embedding_frequencies
        ]
        model.eval()

        test_user_id = [1]
        test_item_id = [2]
        test_freq = [4]
        test_idx_emb = [1]
        test_zero = [0]

        arr_bias = model.embedding_bias.weight.detach().cpu().numpy()
        arr_user_factors = model.embedding_user.weight.detach().cpu().numpy()
        arr_item_factors = model.embedding_item.weight.detach().cpu().numpy()
        arr_frequency_factors = np.vstack(
            [
                emb.weight.detach().cpu().numpy().reshape(1, -1)
                for emb in model.embedding_frequencies
            ]
        )
        print(NUM_USERS, NUM_ITEMS)
        print(arr_frequency_factors, arr_frequency_factors.shape)

        expected_rating = (
            arr_bias[0]
            + np.dot(
                arr_user_factors[test_user_id, :],
                arr_item_factors[test_item_id, :].T,
            )
            + arr_frequency_factors[test_idx_emb, test_freq]
        )

        # act
        rating = model(
            torch.tensor(test_user_id),
            torch.tensor(test_item_id),
            torch.tensor(test_freq),
            torch.tensor(test_idx_emb),
            torch.tensor(test_zero),
        )

        # assert
        assert np.allclose(
            expected_rating,
            rating.detach().cpu().numpy(),
        )

    def test_forward_function_item(self):
        # arrange
        test_frequency_mode: FREQUENCY_MODE = "item"
        test_frequency_num_embeddings = _get_frequency_num_embeddings(
            frequency_mode=test_frequency_mode,
            num_users=NUM_USERS,
            num_items=NUM_ITEMS,
        )
        model = SFCModel(
            num_users=NUM_USERS,
            num_items=NUM_ITEMS,
            frequency_mode=test_frequency_mode,
            frequency_num_bins=test_frequency_num_bins,
            frequency_num_embeddings=test_frequency_num_embeddings,
            embedding_size=test_embedding_size,
            device=test_device,
        )
        # Make sure those embeddings have values (the model sets them as 0).
        [
            torch.nn.init.normal_(
                emb.weight,
            )
            for emb in model.embedding_frequencies
        ]
        model.eval()

        test_user_id = [1]
        test_item_id = [2]
        test_freq = [4]
        test_idx_emb = [2]
        test_zero = [0]

        arr_bias = model.embedding_bias.weight.detach().cpu().numpy()
        arr_user_factors = model.embedding_user.weight.detach().cpu().numpy()
        arr_item_factors = model.embedding_item.weight.detach().cpu().numpy()
        arr_frequency_factors = np.vstack(
            [
                emb.weight.detach().cpu().numpy().reshape(1, -1)
                for emb in model.embedding_frequencies
            ]
        )
        print(NUM_USERS, NUM_ITEMS)
        print(arr_frequency_factors, arr_frequency_factors.shape)

        expected_rating = (
            arr_bias[0]
            + np.dot(
                arr_user_factors[test_user_id, :],
                arr_item_factors[test_item_id, :].T,
            )
            + arr_frequency_factors[test_idx_emb, test_freq]
        )

        # act
        rating = model(
            torch.tensor(test_user_id),
            torch.tensor(test_item_id),
            torch.tensor(test_freq),
            torch.tensor(test_idx_emb),
            torch.tensor(test_zero),
        )

        # assert
        assert np.allclose(
            expected_rating,
            rating.detach().cpu().numpy(),
        )


class TestSoftFrequencyCappingRecommender:
    def test_shape_trained_factors_is_correct(
        self,
        urm: sp.csr_matrix,
        uim: sp.csr_matrix,
        uim_frequency: sp.csr_matrix,
        binned_uim_frequency: sp.dok_matrix,
    ):
        # arrange
        test_frequency_mode: FREQUENCY_MODE
        for test_frequency_mode in ["global", "user", "item"]:
            test_frequency_num_embeddings = _get_frequency_num_embeddings(
                frequency_mode=test_frequency_mode,
                num_users=NUM_USERS,
                num_items=NUM_ITEMS,
            )

            rec = SoftFrequencyCappingRecommender(
                urm_train=urm,
                uim_train=uim,
                uim_frequency=uim_frequency,
                use_gpu=False,
                verbose=False,
            )

            torch.manual_seed(seed=seed)

            # act
            rec.fit(
                epochs=TEST_EPOCHS,
                frequency_mode=test_frequency_mode,
                frequency_num_bins=test_frequency_num_bins,
                batch_size=test_batch_size,
                embedding_size=test_embedding_size,
                learning_rate=test_learning_rate,
                l2_reg=test_l2_reg,
                scheduler_alpha=test_scheduler_alpha,
                scheduler_beta=test_scheduler_beta,
            )

            # assert
            assert (1,) == rec.BIAS_factors.shape
            assert (
                NUM_USERS,
                test_embedding_size,
            ) == rec.USER_factors.shape
            assert (
                NUM_ITEMS,
                test_embedding_size,
            ) == rec.ITEM_factors.shape
            assert (
                test_frequency_num_embeddings,
                test_frequency_num_bins,
            ) == rec.FREQUENCY_factors.shape

    def test_factors_are_deterministic_on_cpu(
        self,
        urm: sp.csr_matrix,
        uim: sp.csr_matrix,
        uim_frequency: sp.csr_matrix,
        binned_uim_frequency: sp.dok_matrix,
    ):
        # arrange
        test_frequency_mode: FREQUENCY_MODE
        for test_frequency_mode in ["global", "user", "item"]:
            rec1 = SoftFrequencyCappingRecommender(
                urm_train=urm,
                uim_train=uim,
                uim_frequency=uim_frequency,
                use_gpu=False,
                verbose=False,
            )
            rec2 = SoftFrequencyCappingRecommender(
                urm_train=urm,
                uim_train=uim,
                uim_frequency=uim_frequency,
                use_gpu=False,
                verbose=False,
            )

            # act
            torch.manual_seed(seed=seed)
            rec1.fit(
                epochs=TEST_EPOCHS,
                frequency_mode=test_frequency_mode,
                frequency_num_bins=test_frequency_num_bins,
                batch_size=test_batch_size,
                embedding_size=test_embedding_size,
                learning_rate=test_learning_rate,
                l2_reg=test_l2_reg,
                scheduler_alpha=test_scheduler_alpha,
                scheduler_beta=test_scheduler_beta,
            )

            torch.manual_seed(seed=seed)
            rec2.fit(
                epochs=TEST_EPOCHS,
                frequency_mode=test_frequency_mode,
                frequency_num_bins=test_frequency_num_bins,
                batch_size=test_batch_size,
                embedding_size=test_embedding_size,
                learning_rate=test_learning_rate,
                l2_reg=test_l2_reg,
                scheduler_alpha=test_scheduler_alpha,
                scheduler_beta=test_scheduler_beta,
            )

            # assert
            assert np.allclose(rec1.BIAS_factors, rec2.BIAS_factors)
            assert np.allclose(rec1.USER_factors, rec2.USER_factors)
            assert np.allclose(rec1.ITEM_factors, rec2.ITEM_factors)
            assert np.allclose(rec1.FREQUENCY_factors, rec2.FREQUENCY_factors)

    def test_correct_score_all_items(
        self,
        urm: sp.csr_matrix,
        uim: sp.csr_matrix,
        uim_frequency: sp.csr_matrix,
        binned_uim_frequency: sp.dok_matrix,
    ):
        # arrange
        test_users = np.arange(start=0, stop=10, dtype=np.int32)
        test_items = None
        test_cutoff = 3

        rec = SoftFrequencyCappingRecommender(
            urm_train=urm,
            uim_train=uim,
            uim_frequency=uim_frequency,
            use_gpu=False,
            verbose=False,
        )

        torch.manual_seed(seed=seed)

        # act
        rec.fit(
            epochs=TEST_EPOCHS,
            frequency_mode=TEST_FREQUENCY_MODE,
            frequency_num_bins=test_frequency_num_bins,
            batch_size=test_batch_size,
            embedding_size=test_embedding_size,
            learning_rate=test_learning_rate,
            l2_reg=test_l2_reg,
            scheduler_alpha=test_scheduler_alpha,
            scheduler_beta=test_scheduler_beta,
        )
        recommendations, scores = rec.recommend(
            user_id_array=test_users,
            items_to_compute=test_items,
            cutoff=test_cutoff,
            remove_seen_flag=False,
            remove_top_pop_flag=False,
            remove_custom_items_flag=False,
            return_scores=True,
        )
        arr_bias_factors = rec.BIAS_factors
        arr_user_factors = rec.USER_factors
        arr_item_factors = rec.ITEM_factors
        arr_freq_factors = rec.FREQUENCY_factors

        expected_item_scores = arr_bias_factors[0] + np.dot(
            arr_user_factors,
            arr_item_factors.T,
        )

        for user in range(NUM_USERS):
            for item in range(NUM_ITEMS):
                idx_freq_ui = binned_uim_frequency[user, item]

                expected_item_scores[user, item] += arr_freq_factors[0, idx_freq_ui]

        # assert
        assert np.allclose(expected_item_scores, scores)

    def test_correct_score_some_items(
        self,
        urm: sp.csr_matrix,
        uim: sp.csr_matrix,
        uim_frequency: sp.csr_matrix,
        binned_uim_frequency: sp.dok_matrix,
    ):
        # arrange
        test_users = np.arange(
            start=0,
            stop=10,
            dtype=np.int32,
        )
        test_items = np.array(
            [0, 2, 4],
            dtype=np.int32,
        )
        test_cutoff = 3

        rec = SoftFrequencyCappingRecommender(
            urm_train=urm,
            uim_train=uim,
            uim_frequency=uim_frequency,
            use_gpu=False,
            verbose=False,
        )

        torch.manual_seed(seed=seed)

        # act
        rec.fit(
            epochs=TEST_EPOCHS,
            frequency_mode=TEST_FREQUENCY_MODE,
            frequency_num_bins=test_frequency_num_bins,
            batch_size=test_batch_size,
            embedding_size=test_embedding_size,
            learning_rate=test_learning_rate,
            l2_reg=test_l2_reg,
            scheduler_alpha=test_scheduler_alpha,
            scheduler_beta=test_scheduler_beta,
        )
        recommendations, scores = rec.recommend(
            user_id_array=test_users,
            items_to_compute=test_items,
            cutoff=test_cutoff,
            remove_seen_flag=False,
            remove_top_pop_flag=False,
            remove_custom_items_flag=False,
            return_scores=True,
        )
        arr_bias_factors = rec.BIAS_factors
        arr_user_factors = rec.USER_factors
        arr_item_factors = rec.ITEM_factors
        arr_freq_factors = rec.FREQUENCY_factors

        expected_item_scores = arr_bias_factors[0] + np.dot(
            arr_user_factors,
            arr_item_factors.T,
        )

        for user in range(NUM_USERS):
            for item in range(NUM_ITEMS):
                idx_freq_ui = binned_uim_frequency[user, item]

                expected_item_scores[user, item] += arr_freq_factors[0, idx_freq_ui]

        mask_ninf = np.ones(NUM_ITEMS, bool)
        mask_ninf[test_items] = 0
        expected_item_scores[:, mask_ninf] = np.NINF

        # assert
        assert np.allclose(expected_item_scores, scores)
