import os
import pickle
from typing import Any, Type, Union

import numpy as np
from sklearn.svm import SVC
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

DEFAULT_FILENAME_SVC = "svm.pkl"  # For the SVC model
DEFAULT_FILENAME_INT64 = "int64.pkl"  # If you need to save numpy.int64

class SVCMaterializer(BaseMaterializer):
    """
    Custom materializer for SVC model
    """

    ASSOCIATED_TYPES = (
        SVC,
        np.ndarray,
        # Add other types if you want to include more
    )

    def handle_input(self, data_type: Type[Any]) -> Union[SVC, np.ndarray]:
        """
        Loads the model from the artifact and returns it.

        Args:
            data_type: The type of the model to be loaded
        """
        super().handle_input(data_type)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME_SVC)
        with fileio.open(filepath, "rb") as fid:
            model = pickle.load(fid)
        return model

    def handle_return(self, obj: Union[SVC, np.ndarray]) -> None:
        """
        Saves the model to the artifact store.

        Args:
            model: The model to be saved
        """
        super().handle_return(obj)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME_SVC)
        with fileio.open(filepath, "wb") as fid:
            pickle.dump(obj, fid)


class Int64Materializer(BaseMaterializer):
    """
    Custom materializer for numpy.int64
    """

    ASSOCIATED_TYPES = (np.int64,)

    def handle_input(self, data_type: Type[Any]) -> np.int64:
        super().handle_input(data_type)
        # If you're not saving to a file, just return the value directly
        return self.artifact.load()

    def handle_return(self, obj: np.int64) -> None:
        super().handle_return(obj)
        # Save it if necessary, otherwise just pass
        self.artifact.save(obj)
