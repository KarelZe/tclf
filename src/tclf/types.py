"""Common type hints."""

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import spmatrix

MatrixLike = np.ndarray | pd.DataFrame | spmatrix
ArrayLike = npt.ArrayLike | pd.Series
