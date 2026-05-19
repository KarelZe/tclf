"""Common type hints."""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import spmatrix

MatrixLike: TypeAlias = np.ndarray | pd.DataFrame | spmatrix
ArrayLike: TypeAlias = npt.ArrayLike | pd.Series
