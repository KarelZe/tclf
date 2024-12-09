"""Common type hints."""

from typing import Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import spmatrix

MatrixLike = Union[np.ndarray, pd.DataFrame, spmatrix]
ArrayLike = Union[npt.ArrayLike, pd.Series]
