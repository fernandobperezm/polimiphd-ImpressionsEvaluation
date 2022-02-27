import os
import traceback
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd
import scipy.sparse as sp
from recsys_framework.Data_manager.DataReader import DataReader
from recsys_framework.Data_manager.DataReader_utils import compute_density
from recsys_framework.Data_manager.Dataset import gini_index
from recsys_framework.Recommenders.DataIO import DataIO


class EvaluationStrategy(Enum):
    TIMESTAMP = "TIMESTAMP"
    LEAVE_LAST_K_OUT = "LEAVE_LAST_K_OUT"


class ParquetDataMixin:
    engine = "pyarrow"
    use_nullable_dtypes = True

    def _to_parquet(
        self,
        df: pd.DataFrame,
        file_path: str,
    ) -> None:
        df.to_parquet(
            path=file_path,
            engine=self.engine,
        )

    def load_parquet(
        self,
        file_path: str,
        to_pandas_func: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        if not os.path.exists(file_path):
            self._to_parquet(
                df=to_pandas_func(),
                file_path=file_path,
            )

        return pd.read_parquet(
            path=file_path,
            engine=self.engine,
            use_nullable_dtypes=self.use_nullable_dtypes,
        )


class BaseDataMixin:
    dataset_name: str
    mapper_item_original_id_to_index: dict[int, int]
    mapper_user_original_id_to_index: dict[int, int]

    def verify_data_consistency(self) -> None:
        pass

    def print_statistics(self) -> None:
        pass

    def save_data(self, save_folder_path) -> None:
        global_attributes_dict = {
            "mapper_item_original_id_to_index": self.mapper_item_original_id_to_index,
            "mapper_user_original_id_to_index": self.mapper_user_original_id_to_index,
            "dataset_name": self.dataset_name,
        }

        data_io = DataIO(folder_path=save_folder_path)
        data_io.save_data(
            data_dict_to_save=global_attributes_dict,
            file_name="dataset_global_attributes"
        )

    def load_data(self, save_folder_path) -> None:
        data_io = DataIO(folder_path=save_folder_path)
        global_attributes_dict = data_io.load_data(
            file_name="dataset_global_attributes"
        )

        for attrib_name, attrib_object in global_attributes_dict.items():
            self.__setattr__(attrib_name, attrib_object)

    def _assert_is_initialized(self) -> None:
        pass

    def get_dataset_name(self):
        return self.dataset_name

    def get_mapper_item_original_id_to_index(self):
        return self.mapper_item_original_id_to_index.copy()

    def get_mapper_user_original_id_to_index(self):
        return self.mapper_user_original_id_to_index.copy()

    def get_global_mapper_dict(self):
        return {
            "user_original_ID_to_index": self.mapper_user_original_id_to_index,
            "item_original_ID_to_index": self.mapper_item_original_id_to_index,
        }


class CSRMatrixStatisticsMixin:
    dataset_name: str
    statistics_matrix: sp.csr_matrix
    statistics_matrix_name: str

    def print_statistics_matrix(
        self,
    ) -> None:
        n_interactions = self.statistics_matrix.nnz
        n_users, n_items = self.statistics_matrix.shape

        uim_all = sp.csr_matrix(self.statistics_matrix)
        user_profile_length = np.ediff1d(uim_all.indptr)

        max_interactions_per_user = user_profile_length.max()
        mean_interactions_per_user = user_profile_length.mean()
        std_interactions_per_user = user_profile_length.std()
        min_interactions_per_user = user_profile_length.min()

        uim_all = sp.csc_matrix(uim_all)
        item_profile_length = np.ediff1d(uim_all.indptr)

        max_interactions_per_item = item_profile_length.max()
        mean_interactions_per_item = item_profile_length.mean()
        std_interactions_per_item = item_profile_length.std()
        min_interactions_per_item = item_profile_length.min()

        print(
            f"DataReader: current dataset is: {self.dataset_name} - {self.statistics_matrix_name}\n"
            f"\tNumber of items: {n_items}\n"
            f"\tNumber of users: {n_users}\n"
            f"\tNumber of interactions: {n_interactions}\n"
            f"\tValue range: {np.min(uim_all.data):.2f}-{np.max(uim_all.data):.2f}\n"
            f"\tInteraction density: {compute_density(uim_all):.2E}\n"
            f"\tInteractions per user:\n"
            f"\t\t Min: {min_interactions_per_user:.2E}\n"
            f"\t\t Mean \u00B1 std: {mean_interactions_per_user:.2E} \u00B1 {std_interactions_per_user:.2E} \n"
            f"\t\t Max: {max_interactions_per_user:.2E}\n"
            f"\tInteractions per item:\n"
            f"\t\t Min: {min_interactions_per_item:.2E}\n"
            f"\t\t Mean \u00B1 std: {mean_interactions_per_item:.2E} \u00B1 {std_interactions_per_item:.2E} \n"
            f"\t\t Max: {max_interactions_per_item:.2E}\n"
            f"\tGini Index: {gini_index(user_profile_length):.2f}\n"
        )


class InteractionsMixin(CSRMatrixStatisticsMixin, BaseDataMixin):
    NAME_URM_ALL = "URM_all"

    NAME_URM_TIMESTAMP_TRAIN = "URM_timestamp_train"
    NAME_URM_TIMESTAMP_VALIDATION = "URM_timestamp_validation"
    NAME_URM_TIMESTAMP_TEST = "URM_timestamp_test"

    NAME_URM_LEAVE_LAST_K_OUT_TRAIN = "URM_leave_last_k_out_train"
    NAME_URM_LEAVE_LAST_K_OUT_VALIDATION = "URM_leave_last_k_out_validation"
    NAME_URM_LEAVE_LAST_K_OUT_TEST = "URM_leave_last_k_out_test"

    is_interactions_implicit: bool
    interactions: dict[str, sp.csr_matrix]

    def verify_data_consistency(self) -> None:
        super().verify_data_consistency()

        print_preamble = f"{self.dataset_name} consistency check:"

        if len(self.interactions.values()) == 0:
            raise ValueError(
                f"{print_preamble} No interactions exist"
            )

        urm_all = self.get_URM_all()
        num_users, num_items = urm_all.shape
        num_interactions = urm_all.nnz

        if num_interactions <= 0:
            raise ValueError(
                f"{print_preamble} Number of interactions in URM is 0."
            )

        if self.is_interactions_implicit and np.any(urm_all.data != 1.0):
            raise ValueError(
                f"{print_preamble} The DataReader is stated to be implicit but the main URM is not"
            )

        if urm_all.shape <= (0, 0):
            raise ValueError(
                f"{print_preamble} No users or items in the URM_all matrix. Shape is {urm_all.shape}"
            )

        for URM_name, URM_object in self.interactions.items():
            if urm_all.shape != URM_object.shape:
                raise ValueError(
                    f"{print_preamble} Number of users or items are different between URM_all and {URM_name}. Shapes "
                    f"are {urm_all.shape} and {URM_object.shape}, respectively."
                )

        # Check if item index-id and user index-id are consistent
        if len(set(self.mapper_user_original_id_to_index.values())) != len(self.mapper_user_original_id_to_index):
            raise ValueError(
                f"{print_preamble} user it-to-index mapper values do not have a 1-to-1 correspondence with the key"
            )

        if len(set(self.mapper_item_original_id_to_index.values())) != len(self.mapper_item_original_id_to_index):
            raise ValueError(
                f"{print_preamble} item it-to-index mapper values do not have a 1-to-1 correspondence with the key"
            )

        if num_users != len(self.mapper_user_original_id_to_index):
            raise ValueError(
                f"{print_preamble} user ID-to-index mapper contains a number of keys different then the number of users"
            )

        if num_items != len(self.mapper_item_original_id_to_index):
            raise ValueError(
                f"{print_preamble} ({num_items=}/{len(self.mapper_item_original_id_to_index)=}" 
                f"mapper contains a number of keys different then the number of items"
            )

        if num_users < max(self.mapper_user_original_id_to_index.values()):
            raise ValueError(
                f"{print_preamble} user ID-to-index mapper contains indices greater than number of users."
            )

        if num_items < max(self.mapper_item_original_id_to_index.values()):
            raise ValueError(
                f"{print_preamble} item ID-to-index mapper contains indices greater than number of item."
            )

        # Check if every non-empty user and item has a mapper value
        URM_all = sp.csc_matrix(urm_all)
        nonzero_items_mask = np.ediff1d(URM_all.indptr) > 0
        nonzero_items = np.arange(0, num_items, dtype=np.int)[nonzero_items_mask]

        if not np.isin(
            nonzero_items,
            np.array(list(self.mapper_item_original_id_to_index.values()))
        ).all():
            raise ValueError(
                f"{print_preamble} there exist items with interactions that do not have a mapper entry"
            )

        URM_all = sp.csr_matrix(urm_all)
        nonzero_users_mask = np.ediff1d(URM_all.indptr) > 0
        nonzero_users = np.arange(0, num_users, dtype=np.int)[nonzero_users_mask]
        if not np.isin(
            nonzero_users,
            np.array(list(self.mapper_user_original_id_to_index.values()))
        ).all():
            raise ValueError(
                f"{print_preamble} there exist users with interactions that do not have a mapper entry"
            )

    def print_statistics(self) -> None:
        super().print_statistics()

        for matrix_name, matrix in self.interactions.items():
            self.statistics_matrix = matrix
            self.statistics_matrix_name = matrix_name
            self.print_statistics_matrix()

    def _assert_is_initialized(self):
        super()._assert_is_initialized()

        if self.interactions is None:
            raise ValueError(
                f"DataReader {self.dataset_name}: Unable to load data split. The split has not been generated"
                f" yet, call the load_data function to do so."
            )

    def get_URM_all(self) -> sp.csr_matrix:
        return self.interactions[self.NAME_URM_ALL].copy()

    def get_loaded_URM_names(self):
        return list(self.interactions.keys())

    def get_urm_by_name(self, name: str) -> sp.csr_matrix:
        return self.interactions[name].copy()

    def get_urm_splits(self, evaluation_strategy: EvaluationStrategy):
        if evaluation_strategy == EvaluationStrategy.LEAVE_LAST_K_OUT:
            return self._get_urm_leave_last_k_out_splits()
        elif evaluation_strategy == EvaluationStrategy.TIMESTAMP:
            return self._get_urm_timestamp_splits()
        else:
            raise ValueError(
                f"Requested split ({evaluation_strategy}) does not exist."
            )

    def _get_urm_leave_last_k_out_splits(self) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
        return (
            self.interactions[self.NAME_URM_LEAVE_LAST_K_OUT_TRAIN],
            self.interactions[self.NAME_URM_LEAVE_LAST_K_OUT_VALIDATION],
            self.interactions[self.NAME_URM_LEAVE_LAST_K_OUT_TEST],
        )

    def _get_urm_timestamp_splits(self) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
        return (
            self.interactions[self.NAME_URM_TIMESTAMP_TRAIN],
            self.interactions[self.NAME_URM_TIMESTAMP_VALIDATION],
            self.interactions[self.NAME_URM_TIMESTAMP_TEST],
        )

    def save_data(self, save_folder_path):
        super().save_data(
            save_folder_path=save_folder_path,
        )

        data_io = DataIO(folder_path=save_folder_path)
        data_io.save_data(
            data_dict_to_save={
                "interactions": self.interactions,
                "is_interactions_implicit": self.is_interactions_implicit,
            },
            file_name="dataset_URM"
        )

    def load_data(self, save_folder_path):
        super().load_data(
            save_folder_path=save_folder_path,
        )

        data_io = DataIO(folder_path=save_folder_path)
        impressions_attributes_dict = data_io.load_data(
            file_name="dataset_URM"
        )

        for attrib_name, attrib_object in impressions_attributes_dict.items():
            self.__setattr__(attrib_name, attrib_object)


class ImpressionsMixin(CSRMatrixStatisticsMixin, BaseDataMixin):
    NAME_UIM_ALL = "UIM_all"

    NAME_UIM_TIMESTAMP_TRAIN = "UIM_timestamp_train"
    NAME_UIM_TIMESTAMP_VALIDATION = "UIM_timestamp_validation"
    NAME_UIM_TIMESTAMP_TEST = "UIM_timestamp_test"

    NAME_UIM_LEAVE_LAST_K_OUT_TRAIN = "UIM_leave_last_k_out_train"
    NAME_UIM_LEAVE_LAST_K_OUT_VALIDATION = "UIM_leave_last_k_out_validation"
    NAME_UIM_LEAVE_LAST_K_OUT_TEST = "UIM_leave_last_k_out_test"

    is_impressions_implicit: bool
    impressions: dict[str, sp.csr_matrix]

    def print_statistics(self) -> None:
        super().print_statistics()

        for matrix_name, matrix in self.impressions.items():
            self.statistics_matrix = matrix
            self.statistics_matrix_name = matrix_name
            self.print_statistics_matrix()

    def _assert_is_initialized(self):
        super()._assert_is_initialized()

        if self.impressions is None:
            raise ValueError(
                f"DataReader {self.dataset_name}: Unable to load data split. The split has not been generated"
                f" yet, call the load_data function to do so."
            )

    def get_uim_by_name(self, name: str) -> sp.csr_matrix:
        return self.impressions[name].copy()

    def get_uim_all(self) -> sp.csr_matrix:
        return self.impressions[self.NAME_UIM_ALL].copy()

    def get_uim_splits(self, evaluation_strategy: EvaluationStrategy):
        if evaluation_strategy == EvaluationStrategy.LEAVE_LAST_K_OUT:
            return self._get_uim_leave_last_k_out_splits()
        elif evaluation_strategy == EvaluationStrategy.TIMESTAMP:
            return self._get_uim_timestamp_splits()
        else:
            raise ValueError(
                f"Requested split ({evaluation_strategy}) does not exist."
            )

    def _get_uim_leave_last_k_out_splits(self) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
        return (
            self.impressions[self.NAME_UIM_LEAVE_LAST_K_OUT_TRAIN],
            self.impressions[self.NAME_UIM_LEAVE_LAST_K_OUT_VALIDATION],
            self.impressions[self.NAME_UIM_LEAVE_LAST_K_OUT_TEST],
        )

    def _get_uim_timestamp_splits(self) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
        return (
            self.impressions[self.NAME_UIM_TIMESTAMP_TRAIN],
            self.impressions[self.NAME_UIM_TIMESTAMP_VALIDATION],
            self.impressions[self.NAME_UIM_TIMESTAMP_TEST],
        )

    def get_loaded_UIM_names(self):
        return list(self.impressions.keys())

    def save_data(self, save_folder_path):
        super().save_data(
            save_folder_path=save_folder_path,
        )

        data_io = DataIO(folder_path=save_folder_path)
        data_io.save_data(
            data_dict_to_save={
                "impressions": self.impressions,
                "is_impressions_implicit": self.is_impressions_implicit,
            },
            file_name="dataset_UIM"
        )

    def load_data(self, save_folder_path):
        super().load_data(
            save_folder_path=save_folder_path,
        )

        data_io = DataIO(folder_path=save_folder_path)
        impressions_attributes_dict = data_io.load_data(
            file_name="dataset_UIM"
        )

        for attrib_name, attrib_object in impressions_attributes_dict.items():
            self.__setattr__(attrib_name, attrib_object)


class BaseDataset(ImpressionsMixin, InteractionsMixin):
    def __init__(
        self,
        dataset_name: str,
        impressions: dict[str, sp.csr_matrix],
        interactions: dict[str, sp.csr_matrix],
        mapper_item_original_id_to_index: dict[Any, int],
        mapper_user_original_id_to_index: dict[Any, int],
        is_impressions_implicit: bool,
        is_interactions_implicit: bool,
    ):
        self.dataset_name = dataset_name
        self.impressions = impressions
        self.interactions = interactions
        self.mapper_item_original_id_to_index = mapper_item_original_id_to_index
        self.mapper_user_original_id_to_index = mapper_user_original_id_to_index
        self.is_impressions_implicit = is_impressions_implicit
        self.is_interactions_implicit = is_interactions_implicit

    @staticmethod
    def empty_dataset() -> "BaseDataset":
        return BaseDataset(
            dataset_name="",
            impressions=dict(),
            interactions=dict(),
            mapper_item_original_id_to_index=dict(),
            mapper_user_original_id_to_index=dict(),
            is_impressions_implicit=False,
            is_interactions_implicit=False,
        )


class BinaryImplicitDataset(BaseDataset):
    def __init__(
        self,
        dataset_name: str,
        impressions: dict[str, sp.csr_matrix],
        interactions: dict[str, sp.csr_matrix],
        mapper_item_original_id_to_index: dict[Any, int],
        mapper_user_original_id_to_index: dict[Any, int],
    ):
        super().__init__(
            dataset_name=dataset_name,
            impressions=impressions,
            interactions=interactions,
            mapper_item_original_id_to_index=mapper_item_original_id_to_index,
            mapper_user_original_id_to_index=mapper_user_original_id_to_index,
            is_impressions_implicit=True,
            is_interactions_implicit=True,
        )


class BaseDataReader(DataReader, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    @abstractmethod
    def dataset(self) -> BaseDataset:
        pass

    def load_data(self, save_folder_path=None):
        """
        :param save_folder_path:    path in which to save the loaded dataset
                                    None    use default "dataset_name/original/"
                                    False   do not save
        :return:
        """

        # Use default e.g., "dataset_name/original/"
        if save_folder_path is None:
            save_folder_path = self.DATASET_SPLIT_ROOT_FOLDER + self._get_dataset_name_root() + self._get_dataset_name_data_subfolder()

        # If save_folder_path contains any path try to load a previously built split from it
        if save_folder_path is not False and not self.reload_from_original_data:

            try:
                loaded_dataset = BaseDataset.empty_dataset()
                loaded_dataset.load_data(save_folder_path)

                self._print("Verifying data consistency...")
                loaded_dataset.verify_data_consistency()
                self._print("Verifying data consistency... Passed!")

                loaded_dataset.print_statistics()
                return loaded_dataset

            except FileNotFoundError:

                self._print("Preloaded data not found, reading from original files...")

            except Exception:

                self._print("Reading split from {} caused the following exception...".format(save_folder_path))
                traceback.print_exc()
                raise Exception("{}: Exception while reading split".format(self._get_dataset_name()))

        self._print("Loading original data")
        loaded_dataset = self._load_from_original_file()

        self._print("Verifying data consistency...")
        loaded_dataset.verify_data_consistency()
        self._print("Verifying data consistency... Passed!")

        if save_folder_path not in [False]:

            # If directory does not exist, create
            if not os.path.exists(save_folder_path):
                self._print("Creating folder '{}'".format(save_folder_path))
                os.makedirs(save_folder_path)

            else:
                self._print("Found already existing folder '{}'".format(save_folder_path))

            loaded_dataset.save_data(save_folder_path)

            self._print("Saving complete!")

        loaded_dataset.print_statistics()
        return loaded_dataset
