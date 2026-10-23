"""Common type hints."""

from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
from narwhals.typing import IntoDataFrame
from scipy.sparse import spmatrix

# IntoDataFrame covers any narwhals-compatible DataFrame:
# pd.DataFrame, pl.DataFrame, cuDF DataFrame, Modin DataFrame, …
# Any covers Array API compatible arrays e.g., torch.Tensor or cupy.ndarray.
MatrixLike: TypeAlias = np.ndarray | IntoDataFrame | spmatrix | Any
ArrayLike: TypeAlias = npt.ArrayLike
