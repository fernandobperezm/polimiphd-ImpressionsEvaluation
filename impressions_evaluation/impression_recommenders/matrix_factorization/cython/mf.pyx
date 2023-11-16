"""
Created on 07/09/17

@author: Maurizio Ferrari Dacrema
"""

#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: unpack_method_calls=True
#cython: overflowcheck=False

#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

from Recommenders.Recommender_utils import check_matrix
from impressions_evaluation.impression_recommenders.matrix_factorization.cython.framework cimport MatrixFactorization_Cython_Epoch
from impressions_evaluation.impression_recommenders.matrix_factorization.cython.framework cimport BPR_sample

import cython

import numpy as np
cimport numpy as np
import time
import sys

from libc.math cimport exp
from libc.stdlib cimport rand


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
cdef class MatrixFactorizationCythonBPRModel(MatrixFactorization_Cython_Epoch):
    def __init__(
        self,
        URM_train,
        n_factors = 1,
        batch_size = 1,
        dropout_quota = None,
        WARP_neg_item_attempts=10,
        learning_rate = 1e-3,
        use_bias = False,
        use_embeddings = True,
        user_reg = 0.0,
        item_reg = 0.0,
        bias_reg = 0.0,
        positive_reg = 0.0,
        negative_reg = 0.0,
        verbose = False,
        random_seed = None,
        print_step_seconds = 300,
        init_mean = 0.0,
        init_std_dev = 0.1,
        sgd_mode='sgd',
        gamma=0.995,
        beta_1=0.9,
        beta_2=0.999,
    ):
        super().__init__(
            URM_train=URM_train,
            n_factors=n_factors,
            algorithm_name="MF_BPR",
            batch_size=batch_size,
            dropout_quota=dropout_quota,
            WARP_neg_item_attempts=WARP_neg_item_attempts,
            learning_rate=learning_rate,
            use_bias=use_bias,
            use_embeddings=use_embeddings,
            user_reg=user_reg,
            item_reg=item_reg,
            bias_reg=bias_reg,
            positive_reg=positive_reg,
            negative_reg=negative_reg,
            init_mean=init_mean,
            init_std_dev=init_std_dev,
            sgd_mode=sgd_mode,
            gamma=gamma,
            beta_1=beta_1,
            beta_2=beta_2,
            random_seed=random_seed,
            verbose=verbose,
            print_step_seconds=print_step_seconds,
        )

    def epochIteration_Cython(self):
        self.epoch()

    cdef void epoch(self):
        cdef BPR_sample sample
        cdef long u, i, j
        cdef long factor_index, num_current_batch, num_sample_in_batch, processed_samples_last_print, print_block_size = 500
        cdef double x_uij, sigmoid_user, sigmoid_item, local_gradient_i, local_gradient_j, local_gradient_u

        cdef double H_i, H_j, W_u, cumulative_loss = 0.0

        cdef long start_time_epoch = time.time()
        cdef long last_print_time = start_time_epoch

        # Get number of available interactions
        cdef long num_total_batch = int(self.n_users / self.batch_size) + 1

        # Renew dropout mask
        if self.dropout_flag:
            for factor_index in range(self.n_factors):
                self.factors_dropout_mask[factor_index] = rand() > self.dropout_quota

            if self.n_factors == 1:
                self.factors_dropout_mask[0] = True

        for num_current_batch in range(num_total_batch):

            self._clear_minibatch_data_structures()

            # Iterate over samples in batch
            for num_sample_in_batch in range(self.batch_size):
                # Uniform user sampling with replacement
                sample = self.sample()

                self._add_BPR_sample_in_minibatch(sample)

                u = sample.user
                i = sample.pos_item
                j = sample.neg_item

                x_uij = 0.0

                for factor_index in range(self.n_factors):
                    if self.factors_dropout_mask[factor_index]:
                        x_uij += (
                            self.USER_factors[u, factor_index]
                            * (
                                self.ITEM_factors[i, factor_index] - self.ITEM_factors[j, factor_index]
                            )
                        )

                # Use gradient of log(sigm(-x_uij))
                sigmoid_item = 1 / (1 + exp(x_uij))
                sigmoid_user = sigmoid_item

                cumulative_loss += x_uij ** 2

                for factor_index in range(self.n_factors):
                    if self.factors_dropout_mask[factor_index]:
                        # Copy original value to avoid messing up the updates
                        H_i = self.ITEM_factors[i, factor_index]
                        H_j = self.ITEM_factors[j, factor_index]
                        W_u = self.USER_factors[u, factor_index]

                        # Compute gradients
                        local_gradient_i = sigmoid_item * (W_u) - self.positive_reg * H_i
                        local_gradient_j = sigmoid_item * (-W_u) - self.negative_reg * H_j
                        local_gradient_u = sigmoid_user * (H_i - H_j) - self.user_reg * W_u

                        self.USER_factors_minibatch_accumulator[u, factor_index] += local_gradient_u
                        self.ITEM_factors_minibatch_accumulator[i, factor_index] += local_gradient_i
                        self.ITEM_factors_minibatch_accumulator[j, factor_index] += local_gradient_j

            self._apply_minibatch_updates_to_latent_factors()

            # Exponentiation of beta at the end of each sample
            if self.useAdam:
                self.beta_1_power_t *= self.beta_1
                self.beta_2_power_t *= self.beta_2

            if self.verbose and (
                processed_samples_last_print >= print_block_size or num_current_batch == num_total_batch - 1
            ):

                current_time = time.time()

                # Set block size to the number of items necessary in order to print every 30 seconds
                samples_per_sec = num_current_batch / (time.time() - start_time_epoch)

                print_block_size = int(samples_per_sec * 30)

                if current_time - last_print_time > 30 or num_current_batch == num_total_batch - 1:
                    print(
                    "{}: Processed {} ( {:.2f}% ) in {:.2f} seconds. BPR loss {:.2E}. Sample per second: {:.0f}".format(
                        self.algorithm_name,
                        num_current_batch * self.batch_size,
                        100.0 * num_current_batch / num_total_batch,
                        time.time() - last_print_time,
                        cumulative_loss / (num_current_batch * self.batch_size + 1),
                        float(num_current_batch * self.batch_size + 1) / (time.time() - start_time_epoch)))

                    last_print_time = current_time
                    processed_samples_last_print = 0

                    sys.stdout.flush()
                    sys.stderr.flush()

    cdef BPR_sample sample(self):
        cdef long user = -1
        cdef long pos_item = -1
        cdef long neg_item = -1

        user = self._sample_user()
        pos_item = self._sample_positive(user)
        neg_item = self._sample_negative(user)

        return BPR_sample(user, pos_item, neg_item)

    cdef long _sample_user(self):
        cdef long user
        cdef long start_pos_seen_items
        cdef long end_pos_seen_items
        cdef int n_seen_items = 0

        # Skip users with no interactions or with no negative items
        while n_seen_items == 0 or n_seen_items == self.n_items:
            user = rand() % self.n_users

            start_pos_seen_items = self.URM_train_indptr[user]
            end_pos_seen_items = self.URM_train_indptr[user + 1]

            n_seen_items = end_pos_seen_items - start_pos_seen_items

        return user

    cdef long _sample_positive(self, long user):
        cdef long pos_item
        cdef long index
        cdef long start_pos_seen_items
        cdef long end_pos_seen_items
        cdef int n_seen_items

        start_pos_seen_items = self.URM_train_indptr[user]
        end_pos_seen_items = self.URM_train_indptr[user + 1]

        n_seen_items = end_pos_seen_items - start_pos_seen_items

        index = rand() % n_seen_items

        pos_item = self.URM_train_indices[start_pos_seen_items + index]

        return pos_item

    cdef long _sample_negative(self, long user):
        cdef long neg_item
        cdef long index
        cdef long start_pos_seen_items
        cdef long end_pos_seen_items
        cdef int n_seen_items
        cdef int neg_item_selected

        start_pos_seen_items = self.URM_train_indptr[user]
        end_pos_seen_items = self.URM_train_indptr[user + 1]

        n_seen_items = end_pos_seen_items - start_pos_seen_items

        # It's faster to just try again then to build a mapping of the non-seen items
        # for every user
        neg_item_selected = False
        while not neg_item_selected:
            neg_item = rand() % self.n_items

            index = 0
            # Indices data is sorted, so I don't need to go to the end of the current row
            while index < n_seen_items and self.URM_train_indices[start_pos_seen_items + index] < neg_item:
                index += 1

            # If the positive item in position 'index' is == sample.neg_item, negative not selected
            # If the positive item in position 'index' is > sample.neg_item or index == n_seen_items, negative selected
            if index == n_seen_items or self.URM_train_indices[start_pos_seen_items + index] > neg_item:
                neg_item_selected = True

        return neg_item


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
cdef class SampleImpressionsNegativeMatrixFactorizationCythonBPRModel(MatrixFactorizationCythonBPRModel):
    cdef double[:] URM_impressions_data
    cdef int[:] URM_impressions_indices
    cdef int[:] URM_impressions_indptr

    cdef int impression_data_flag, impression_sampling_mode_inside_flag, impression_sampling_mode_ratio_flag
    cdef double impression_sampling_inside_ratio

    def __init__(
        self,
        URM_train,
        UIM_train,
        n_factors = 1,
        batch_size = 1,
        dropout_quota = None,
        WARP_neg_item_attempts=10,
        learning_rate = 1e-3,
        use_bias = False,
        use_embeddings = True,
        user_reg = 0.0,
        item_reg = 0.0,
        bias_reg = 0.0,
        positive_reg = 0.0,
        negative_reg = 0.0,
        verbose = False,
        random_seed = None,
        print_step_seconds = 300,
        init_mean = 0.0,
        init_std_dev = 0.1,
        sgd_mode='sgd',
        gamma=0.995,
        beta_1=0.9,
        beta_2=0.999,
    ):
        super().__init__(
            URM_train=URM_train,
            n_factors=n_factors,
            batch_size=batch_size,
            dropout_quota=dropout_quota,
            WARP_neg_item_attempts=WARP_neg_item_attempts,
            learning_rate=learning_rate,
            use_bias=use_bias,
            use_embeddings=use_embeddings,
            user_reg=user_reg,
            item_reg=item_reg,
            bias_reg=bias_reg,
            positive_reg=positive_reg,
            negative_reg=negative_reg,
            init_mean=init_mean,
            init_std_dev=init_std_dev,
            sgd_mode=sgd_mode,
            gamma=gamma,
            beta_1=beta_1,
            beta_2=beta_2,
            random_seed=random_seed,
            verbose=verbose,
            print_step_seconds=print_step_seconds,
        )

        # Create copy of URM_train in csr format
        # make sure indices are sorted
        UIM_train = check_matrix(UIM_train, 'csr')
        UIM_train = UIM_train.sorted_indices()

        self.URM_impressions_indices = UIM_train.indices
        self.URM_impressions_data = np.array(UIM_train.data, dtype=np.float64)
        self.URM_impressions_indptr = UIM_train.indptr

    cdef long _sample_negative(self, long user):
        cdef long neg_item
        cdef long index
        cdef long start_pos_seen_items
        cdef long end_pos_seen_items
        cdef long start_pos_impression_items
        cdef long end_pos_impression_items

        cdef int neg_item_selected

        cdef int n_seen_items
        cdef int n_impression_items = 0

        start_pos_seen_items = self.URM_train_indptr[user]
        end_pos_seen_items = self.URM_train_indptr[user + 1]

        start_pos_impression_items = self.URM_impressions_indptr[user]
        end_pos_impression_items = self.URM_impressions_indptr[user + 1]

        n_seen_items = end_pos_seen_items - start_pos_seen_items
        n_impression_items = end_pos_impression_items - start_pos_impression_items

        # It's faster to just try again then to build a mapping of the non-seen items
        # for every user
        neg_item_selected = False
        while not neg_item_selected:
            index = rand() % n_impression_items
            neg_item = self.URM_impressions_indices[start_pos_impression_items + index]

            index = 0
            # Indices data is sorted, so I don't need to go to the end of the current row
            while index < n_seen_items and self.URM_train_indices[start_pos_seen_items + index] < neg_item:
                index += 1

            # If the positive item in position 'index' is == sample.neg_item, negative not selected
            # If the positive item in position 'index' is > neg_item or index == n_seen_items, negative selected
            if index == n_seen_items or self.URM_train_indices[start_pos_seen_items + index] > neg_item:
                neg_item_selected = True

        return neg_item
