import logging

import attrs
import lightgbm
import scipy.sparse as sp
from recsys_framework_extensions.recommenders.base import (
    SearchHyperParametersBaseRecommender,
)
from skopt.space import Categorical, Integer, Real

logger = logging.getLogger(__name__)


@attrs.define(kw_only=True, frozen=True, slots=False)
class SearchHyperParametersSFCRecommender(SearchHyperParametersBaseRecommender):
    epochs: Categorical = attrs.field(
        default=Categorical(  # The paper does not specify.
            [1],
        )
    )
    frequency_num_bins: Categorical = attrs.field(
        default=Categorical(  # The paper uses 26
            [26],
        )
    )
    frequency_mode: Categorical = attrs.field(
        default=Categorical(  # The paper uses global or two content-based variants of item.
            [
                "global",
                "user",
                "item",
            ],
        )
    )
    batch_size: Integer = attrs.field(
        default=Integer(  # The paper does not specify ranges.
            low=2**7,  # 128
            high=2**14,  # 16384
            prior="uniform",
        ),
    )
    embedding_size: Integer = attrs.field(
        default=Integer(  # The paper does not specify ranges.
            low=2**0,
            high=2**10,
            prior="uniform",
        ),
    )
    learning_rate: Real = attrs.field(
        default=Real(  # The paper does not specify ranges.
            low=1e-5,
            high=1e-2,
            prior="log-uniform",
        )
    )
    l2_reg: Real = attrs.field(
        default=Real(  # The paper does not specify ranges.
            low=1e-5,
            high=1e-2,
            prior="log-uniform",
        )
    )
    scheduler_alpha: Real = attrs.field(
        default=Real(  # The paper does not specify ranges.
            low=1e-5,
            high=1e-2,
            prior="log-uniform",
        )
    )
    scheduler_beta: Real = attrs.field(
        default=Real(  # The paper does not specify ranges.
            low=1e-5,
            high=1e-2,
            prior="log-uniform",
        )
    )


class GBDT:
    def __init__(
        self,
        urm_train: sp.csr_matrix,
        uim_train: sp.csr_matrix,
    ):
        ...

    def fit(self, *args, **kwargs):
        ranker = lightgbm.LGBMRanker(
            objective="lambdarank",
            boosting_type="gbdt",
            n_estimators=5,
            importance_type="gain",
            metric="ndcg",
            num_leaves=10,
            learning_rate=0.05,
            max_depth=-1,
            label_gain=[i for i in range(max(y_train.max(), y_test.max()) + 1)],
        )

        # Training the model
        ranker.fit(
            X=X_train,
            y=y_train,
            group=qids_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_group=[qids_train, qids_test],
            eval_at=[4, 8],
        )
        ...
