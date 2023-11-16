import cython

cdef struct BPR_sample:
    long user
    long pos_item
    long neg_item

cdef struct MSE_sample:
    long user
    long item
    double rating

cdef class MatrixFactorization_Cython_Epoch:
    cdef int n_users, n_items, n_factors, print_step_seconds
    cdef algorithm_name

    cdef double learning_rate, user_reg, item_reg, positive_reg, negative_reg, bias_reg
    cdef double init_mean, init_std_dev, MSE_negative_interactions_quota, MSE_sample_negative_interactions_flag

    cdef int batch_size

    cdef int algorithm_is_svdpp, algorithm_is_asy_svd, algorithm_is_BPR, algorithm_is_WARP
    cdef int WARP_neg_item_attempts

    cdef int[:] URM_train_indices, URM_train_indptr, profile_length
    cdef double[:] URM_train_data

    cdef double[:,:] USER_factors, ITEM_factors
    cdef double[:] USER_bias, ITEM_bias, GLOBAL_bias
    cdef int[:] factors_dropout_mask
    cdef int dropout_flag
    cdef double dropout_quota

    # Mini-batch sample data
    cdef double[:,:] USER_factors_minibatch_accumulator, ITEM_factors_minibatch_accumulator
    cdef double[:] USER_bias_minibatch_accumulator, ITEM_bias_minibatch_accumulator, GLOBAL_bias_minibatch_accumulator

    cdef long[:] mini_batch_sampled_items, mini_batch_sampled_users
    cdef long[:] mini_batch_sampled_items_flag, mini_batch_sampled_users_flag
    cdef long mini_batch_sampled_items_counter, mini_batch_sampled_users_counter

    # Adaptive gradient
    cdef int useAdaGrad, useRmsprop, useAdam, verbose, use_bias, use_embeddings

    cdef double [:,:] sgd_cache_I, sgd_cache_U, sgd_cache_bias_I, sgd_cache_bias_U, sgd_cache_bias_GLOBAL
    cdef double gamma

    cdef double [:,:] sgd_cache_I_momentum_1, sgd_cache_I_momentum_2
    cdef double [:,:] sgd_cache_U_momentum_1, sgd_cache_U_momentum_2
    cdef double [:,:] sgd_cache_bias_I_momentum_1, sgd_cache_bias_I_momentum_2
    cdef double [:,:] sgd_cache_bias_U_momentum_1, sgd_cache_bias_U_momentum_2
    cdef double [:,:] sgd_cache_bias_GLOBAL_momentum_1, sgd_cache_bias_GLOBAL_momentum_2
    cdef double beta_1, beta_2, beta_1_power_t, beta_2_power_t
    cdef double momentum_1, momentum_2

    cdef void _clear_minibatch_data_structures(self)
    cdef void _add_MSE_sample_in_minibatch(self, MSE_sample sample)
    cdef void _add_BPR_sample_in_minibatch(self, BPR_sample sample)
    cdef void _apply_minibatch_updates_to_latent_factors(self)
    cdef double adaptive_gradient(self, double gradient, long user_or_item_id, long factor_id, double[:,:] sgd_cache, double[:,:] sgd_cache_momentum_1, double[:,:] sgd_cache_momentum_2)
    cdef MSE_sample sampleMSE_Cython(self)
    cdef BPR_sample sampleBPR_Cython(self)
    cdef BPR_sample sampleWARP_Cython(self)