"""Common type hints."""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from narwhals.typing import IntoDataFrame
from scipy.sparse import spmatrix

# IntoDataFrame covers any narwhals-compatible DataFrame:
# pd.DataFrame, pl.DataFrame, cuDF DataFrame, Modin DataFrame, …
MatrixLike: TypeAlias  = np.ndarray| IntoDataFrame| spmatrix
ArrayLike: TypeAlias  = npt.ArrayLike
