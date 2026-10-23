"""Implements classical trade classification rules with a sklearn-like interface.

Both simple rules like quote rule or tick test or hybrids are included.

The classifier supports the Array API (see
https://scikit-learn.org/stable/modules/array_api.html). Inputs that are not
DataFrames, such as PyTorch tensors or CuPy arrays, are processed in their own
namespace and on their own device.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

if TYPE_CHECKING:
    from typing import TypeGuard

    from narwhals.typing import IntoDataFrame
    from sklearn.utils import Tags

import narwhals.stable.v2 as nw
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils._array_api import (
    check_same_namespace,
    get_namespace,
    get_namespace_and_device,
    move_to,
    supported_float_dtypes,
)
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    validate_data,
)

from tclf.types import ArrayLike, MatrixLike

ALLOWED_FUNC_LITERALS = Literal[
    "tick",
    "rev_tick",
    "quote",
    "lr",
    "rev_lr",
    "emo",
    "rev_emo",
    "clnv",
    "rev_clnv",
    "trade_size",
    "depth",
    "nan",
]
ALLOWED_FUNC_STR: tuple[ALLOWED_FUNC_LITERALS, ...] = get_args(ALLOWED_FUNC_LITERALS)


def _is_frame(X: object) -> TypeGuard[IntoDataFrame]:
    """Check if X is a narwhals-compatible eager DataFrame.

    Returns:
        TypeGuard[IntoDataFrame]: True if X is a narwhals-compatible DataFrame.
    """
    return nw.dependencies.is_into_dataframe(X)


def _required_columns(func_str: ALLOWED_FUNC_LITERALS, subset: str) -> list[str]:
    """Get the columns required by a classification rule.

    Args:
        func_str (ALLOWED_FUNC_LITERALS): rule e. g., 'tick'.
        subset (str): subset e. g., 'ex'.

    Returns:
        list[str]: required column names.
    """
    quote = ["trade_price", f"ask_{subset}", f"bid_{subset}"]
    sizes = [f"ask_size_{subset}", f"bid_size_{subset}"]
    lag, lead = f"price_{subset}_lag", f"price_{subset}_lead"
    mapping: dict[str, list[str]] = {
        "tick": ["trade_price", lag],
        "rev_tick": ["trade_price", lead],
        "quote": quote,
        "lr": [*quote, lag],
        "rev_lr": [*quote, lead],
        "emo": [*quote, lag],
        "rev_emo": [*quote, lead],
        "clnv": [*quote, lag],
        "rev_clnv": [*quote, lead],
        "trade_size": ["trade_size", *sizes],
        "depth": [*quote, *sizes],
        "nan": [],
    }
    return mapping[func_str]


def _isclose(a: Any, b: Any, xp: Any, atol: float = 1e-4, rtol: float = 1e-5) -> Any:
    """Array API compatible equivalent of `np.isclose`.

    `np.isclose` is not part of the Array API standard.

    Args:
        a (Any): first array.
        b (Any): second array.
        xp (Any): array namespace.
        atol (float, optional): absolute tolerance. Defaults to 1e-4.
        rtol (float, optional): relative tolerance. Defaults to 1e-5.

    Returns:
        Any: boolean array with result.
    """
    # as in `np.isclose`, non-finite values are only close if they are equal.
    equal = a == b
    finite = xp.isfinite(a) & xp.isfinite(b)
    # mask non-finite values to avoid warnings for e. g., inf - inf.
    a, b = xp.where(finite, a, xp.zeros_like(a)), xp.where(finite, b, xp.zeros_like(b))
    return equal | (finite & (xp.abs(a - b) <= atol + rtol * xp.abs(b)))


class _Frame:
    """Lightweight internal frame of 1-D columns sharing a namespace and device."""

    def __init__(self, data: Any, columns: list[str], xp: Any, device: Any) -> None:
        self._cols: dict[str, Any] = {col: data[:, i] for i, col in enumerate(columns)}
        self.shape = data.shape
        self.dtype = data.dtype
        self.xp = xp
        self.device = device

    def __getitem__(self, key: str) -> Any:
        return self._cols[key]

    def full(self, fill_value: float) -> Any:
        """Create a 1-D array filled with `fill_value` in the frame's namespace.

        Args:
            fill_value (float): fill value.

        Returns:
            Any: array of shape (n_samples,).
        """
        return self.xp.full(
            (self.shape[0],), fill_value, dtype=self.dtype, device=self.device
        )


class ClassicalClassifier(ClassifierMixin, BaseEstimator):
    """ClassicalClassifier implements several trade classification rules.

    Including:
    Tick test,
    Reverse tick test,
    Quote rule,
    LR algorithm,
    EMO algorithm,
    CLNV algorithm,
    Trade size rule,
    Depth rule,
    and nan

    Args:
        classifier mixin (ClassifierMixin): mixin for classifier functionality, such as `predict_proba()`
        base estimator (BaseEstimator): base estimator for basic functionality, such as `transform()`
    """

    X_: _Frame

    def __init__(
        self,
        layers: list[
            tuple[
                ALLOWED_FUNC_LITERALS,
                str,
            ]
        ]
        | None = None,
        *,
        features: list[str] | None = None,
        random_state: float | None = 42,
        strategy: Literal["random", "const"] = "random",
    ):
        """Initialize a ClassicalClassifier.

        Examples:
            >>> import numpy as np
            >>> X = np.array(
            ...     [
            ...         [1.5, 1, 3],
            ...         [2.5, 1, 3],
            ...         [1.5, 3, 1],
            ...         [2.5, 3, 1],
            ...         [1.0, np.nan, 1.0],
            ...         [3.0, np.nan, np.nan],
            ...     ]
            ... )
            >>> clf = ClassicalClassifier(
            ...     layers=[("quote", "ex")],
            ...     strategy="const",
            ...     features=["trade_price", "bid_ex", "ask_ex"],
            ... )
            >>> clf.fit(X)  # doctest: +ELLIPSIS
            ClassicalClassifier(...)
            >>> pred = clf.predict_proba(X)

        Args:
            layers (List[tuple[ALLOWED_FUNC_LITERALS, str]]): Layers of classical rule and subset name. Supported rules: "tick", "rev_tick", "quote", "lr", "rev_lr", "emo", "rev_emo", "trade_size", "depth", and "nan". Defaults to None, which results in classification by 'strategy' parameter.
            features (List[str] | None, optional): List of feature names in order of columns. Required to match columns in feature matrix with label. Can be `None`, if a DataFrame (e.g., `pd.DataFrame` or `pl.DataFrame`) or a dict of arrays is passed. Defaults to None.
            random_state (float | None, optional): random seed. Defaults to 42.
            strategy (Literal[&quot;random&quot;, &quot;const&quot;], optional): Strategy to fill unclassfied. Randomly with uniform probability or with constant 0. Defaults to &quot;random&quot;.
        """
        self.layers = layers
        self.random_state = random_state
        self.features = features
        self.strategy = strategy

    def __sklearn_tags__(self) -> Tags:
        """Set tags for sklearn.

        See: https://scikit-learn.org/stable/developers/develop.html#estimator-tags

        Returns:
            Tags: estimator tags
        """
        tags = super().__sklearn_tags__()
        tags.array_api_support = True
        tags.input_tags.allow_nan = True
        tags.target_tags.required = False
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.poor_score = True
        return tags

    def _label(self, buy: Any, sell: Any) -> Any:
        """Label trades as buy (1), sell (-1), or unclassified (nan).

        Args:
            buy (Any): boolean array, True for buyer-initiated trades.
            sell (Any): boolean array, True for seller-initiated trades.

        Returns:
            Any: labels. Can be np.NaN.
        """
        xp = self.X_.xp
        # scalars are allowed in `where` since Array API 2024.12, if one arg is an array.
        return xp.where(buy, 1.0, xp.where(sell, -1.0, self.X_.full(xp.nan)))

    def _tick(self, subset: str) -> Any:
        """Classify a trade as a buy (sell) if its trade price is above (below) the closest different price of a previous trade.

        Args:
            subset (str): subset i.e., 'all' or 'ex'.

        Returns:
            Any: result of tick rule. Can be np.NaN.
        """
        return self._label(
            self.X_["trade_price"] > self.X_[f"price_{subset}_lag"],
            self.X_["trade_price"] < self.X_[f"price_{subset}_lag"],
        )

    def _rev_tick(self, subset: str) -> Any:
        """Classify a trade as a sell (buy) if its trade price is below (above) the closest different price of a subsequent trade.

        Args:
            subset (str): subset i.e.,'all' or 'ex'.

        Returns:
            Any: result of reverse tick rule. Can be np.NaN.
        """
        return self._label(
            self.X_[f"price_{subset}_lead"] < self.X_["trade_price"],
            self.X_[f"price_{subset}_lead"] > self.X_["trade_price"],
        )

    def _quote(self, subset: str) -> Any:
        """Classify a trade as a buy (sell) if its trade price is above (below) the midpoint of the bid and ask spread. Trades executed at the midspread are not classified.

        Args:
            subset (str): subset i.e., 'ex' or 'best'.

        Returns:
            Any: result of quote rule. Can be np.NaN.
        """
        mid = self._mid(subset)
        return self._label(
            self.X_["trade_price"] > mid,
            self.X_["trade_price"] < mid,
        )

    def _lr(self, subset: str) -> Any:
        """Classify a trade as a buy (sell) if its price is above (below) the midpoint (quote rule), and use the tick test to classify midspread trades.

        Adapted from Lee and Ready (1991).

        Args:
            subset (str): subset i.e., 'ex' or 'best'.

        Returns:
            Any: result of the lee and ready algorithm with tick rule.
            Can be np.NaN.
        """
        xp = self.X_.xp
        q_r = self._quote(subset)
        return xp.where(~xp.isnan(q_r), q_r, self._tick(subset))

    def _rev_lr(self, subset: str) -> Any:
        """Classify a trade as a buy (sell) if its price is above (below) the midpoint (quote rule), and use the reverse tick test to classify midspread trades.

        Adapted from Lee and Ready (1991).

        Args:
            subset (str): subset i.e.,'ex' or 'best'.

        Returns:
            Any: result of the lee and ready algorithm with reverse tick
            rule. Can be np.NaN.
        """
        xp = self.X_.xp
        q_r = self._quote(subset)
        return xp.where(~xp.isnan(q_r), q_r, self._rev_tick(subset))

    def _mid(self, subset: str) -> Any:
        """Calculate the midpoint of the bid and ask spread.

        Midpoint is calculated as the average of the bid and ask spread if the spread is positive. Otherwise, np.NaN is returned.

        Args:
            subset (str): subset i.e.,
            'ex' or 'best'

        Returns:
            Any: midpoints. Can be np.NaN.
        """
        return self.X_.xp.where(
            self.X_[f"ask_{subset}"] >= self.X_[f"bid_{subset}"],
            0.5 * (self.X_[f"ask_{subset}"] + self.X_[f"bid_{subset}"]),
            self.X_.full(self.X_.xp.nan),
        )

    def _is_at_ask_xor_bid(self, subset: str) -> Any:
        """Check if the trade price is at the ask xor bid.

        Args:
            subset (str): subset i.e.,
            'ex' or 'best'.

        Returns:
            Any: boolean array with result.
        """
        xp = self.X_.xp
        at_ask = _isclose(self.X_["trade_price"], self.X_[f"ask_{subset}"], xp)
        at_bid = _isclose(self.X_["trade_price"], self.X_[f"bid_{subset}"], xp)
        return at_ask ^ at_bid

    def _is_at_upper_xor_lower_quantile(
        self, subset: str, quantiles: float = 0.3
    ) -> Any:
        """Check if the trade price is at the ask xor bid.

        Args:
            subset (str): subset i.e., 'ex'.
            quantiles (float, optional): percentage of quantiles. Defaults to 0.3.

        Returns:
            Any: boolean array with result.
        """
        in_upper = (
            (1.0 - quantiles) * self.X_[f"ask_{subset}"]
            + quantiles * self.X_[f"bid_{subset}"]
            <= self.X_["trade_price"]
        ) & (self.X_["trade_price"] <= self.X_[f"ask_{subset}"])
        in_lower = (self.X_[f"bid_{subset}"] <= self.X_["trade_price"]) & (
            self.X_["trade_price"]
            <= quantiles * self.X_[f"ask_{subset}"]
            + (1.0 - quantiles) * self.X_[f"bid_{subset}"]
        )
        return in_upper ^ in_lower

    def _emo(self, subset: str) -> Any:
        """Classify a trade as a buy (sell) if the trade takes place at the ask (bid) quote, and use the tick test to classify all other trades.

        Adapted from Ellis et al. (2000).

        Args:
            subset (Literal[&quot;ex&quot;, &quot;best&quot;]): subset i.e., 'ex' or 'best'.

        Returns:
            Any: result of the emo algorithm with tick rule. Can be
            np.NaN.
        """
        return self.X_.xp.where(
            self._is_at_ask_xor_bid(subset), self._quote(subset), self._tick(subset)
        )

    def _rev_emo(self, subset: str) -> Any:
        """Classify a trade as a buy (sell) if the trade takes place at the ask (bid) quote, and use the reverse tick test to classify all other trades.

        Adapted from Grauer et al. (2022).

        Args:
            subset (str): subset i.e., 'ex' or 'best'.

        Returns:
            Any: result of the emo algorithm with reverse tick rule.
            Can be np.NaN.
        """
        return self.X_.xp.where(
            self._is_at_ask_xor_bid(subset), self._quote(subset), self._rev_tick(subset)
        )

    def _clnv(self, subset: str) -> Any:
        """Classify a trade based on deciles of the bid and ask spread.

        Spread is divided into ten deciles and trades are classified as follows:
        - use quote rule for at ask until 30 % below ask (upper 3 deciles)
        - use quote rule for at bid until 30 % above bid (lower 3 deciles)
        - use tick rule for all other trades (±2 deciles from midpoint; outside
        bid or ask).

        Adapted from Chakrabarty et al. (2007).

        Args:
            subset (str): subset i.e.,'ex' or 'best'.

        Returns:
            Any: result of the emo algorithm with tick rule. Can be
            np.NaN.
        """
        return self.X_.xp.where(
            self._is_at_upper_xor_lower_quantile(subset),
            self._quote(subset),
            self._tick(subset),
        )

    def _rev_clnv(self, subset: str) -> Any:
        """Classify a trade based on deciles of the bid and ask spread.

        Spread is divided into ten deciles and trades are classified as follows:
        - use quote rule for at ask until 30 % below ask (upper 3 deciles)
        - use quote rule for at bid until 30 % above bid (lower 3 deciles)
        - use reverse tick rule for all other trades (±2 deciles from midpoint;
        outside bid or ask).

        Similar to extension of emo algorithm proposed Grauer et al. (2022).

        Args:
            subset (str): subset i.e., 'ex' or 'best'.

        Returns:
            Any: result of the emo algorithm with tick rule. Can be
            np.NaN.
        """
        return self.X_.xp.where(
            self._is_at_upper_xor_lower_quantile(subset),
            self._quote(subset),
            self._rev_tick(subset),
        )

    def _trade_size(self, subset: str) -> Any:
        """Classify a trade as a buy (sell) the trade size matches exactly either the bid (ask) quote size.

        Adapted from Grauer et al. (2022).

        Args:
            subset (str): subset i.e., 'ex' or 'best'.

        Returns:
            Any: result of the trade size rule. Can be np.NaN.
        """
        xp = self.X_.xp
        bid_eq_ask = _isclose(
            self.X_[f"ask_size_{subset}"], self.X_[f"bid_size_{subset}"], xp
        )

        ts_eq_bid = (
            _isclose(self.X_["trade_size"], self.X_[f"bid_size_{subset}"], xp)
            & ~bid_eq_ask
        )
        ts_eq_ask = (
            _isclose(self.X_["trade_size"], self.X_[f"ask_size_{subset}"], xp)
            & ~bid_eq_ask
        )

        return self._label(ts_eq_bid, ts_eq_ask)

    def _depth(self, subset: str) -> Any:
        """Classify midspread trades as buy (sell), if the ask size (bid size) exceeds the bid size (ask size).

        Adapted from Grauer et al. (2022).

        Args:
            subset (str): subset i.e., 'ex' or 'best'.

        Returns:
            Any: result of depth rule. Can be np.NaN.
        """
        at_mid = _isclose(self._mid(subset), self.X_["trade_price"], self.X_.xp)

        return self._label(
            at_mid & (self.X_[f"ask_size_{subset}"] > self.X_[f"bid_size_{subset}"]),
            at_mid & (self.X_[f"ask_size_{subset}"] < self.X_[f"bid_size_{subset}"]),
        )

    def _nan(self, subset: str) -> Any:
        """Classify nothing. Fast forward results from previous classifier.

        Returns:
            Any: result of the trade size rule. Can be np.NaN.
        """
        return self.X_.full(self.X_.xp.nan)

    def _validate_input(
        self, X: MatrixLike | Mapping[str, Any], *, reset: bool, method: str
    ) -> tuple[Any, list[str] | None, Any, Any]:
        """Validate X and resolve its column names.

        DataFrames are converted to NumPy, as in scikit-learn. Dicts of 1-D
        arrays and 2-D arrays keep their array namespace and device.

        Args:
            X (MatrixLike | Mapping[str, Any]): DataFrame, dict of 1-D arrays, or 2-D array.
            reset (bool): whether called from `fit`.
            method (str): name of the calling public method.

        Returns:
            tuple[Any, list[str] | None, Any, Any]: validated 2-D array, column
            names, array namespace and device.

        Raises:
            ValueError: columns missing in dict or column count mismatch.
        """
        columns: list[str] | None = self.features if reset else self.columns_
        if _is_frame(X):
            nw_X = nw.from_native(X, eager_only=True)
            if columns is None:
                columns = list(nw_X.columns)
            # selecting by str is fast; nw.col is needed for non-str names e. g., int.
            if all(isinstance(col, str) for col in columns):
                X = nw_X.select(columns).to_numpy()
            else:
                X = nw_X.select(nw.col(*columns)).to_numpy()
        elif isinstance(X, Mapping) and not sp.issparse(X):
            data = cast("Mapping[str, Any]", X)
            if columns is None:
                columns = list(data)
            missing_columns = sorted(set(columns) - set(data))
            if missing_columns:
                raise ValueError(
                    f"Expected to find columns: {missing_columns}. Check naming/presenence of columns. See: https://karelze.github.io/tclf/naming_conventions/"
                )
            xp, _, device = get_namespace_and_device(*(data[col] for col in columns))
            X = xp.stack(
                [xp.asarray(data[col], device=device) for col in columns], axis=1
            )

        if not reset:
            check_same_namespace(X, self, attribute="classes_", method=method)

        xp, _, device = get_namespace_and_device(X)
        X = validate_data(
            self,
            X,
            reset=reset,
            dtype=supported_float_dtypes(xp, device),
            accept_sparse=False,
            ensure_all_finite=False,
        )

        if columns and X.shape[1] != len(columns):
            raise ValueError(f"Expected {len(columns)} columns, got {X.shape[1]}.")

        return X, columns, xp, device

    def fit(
        self,
        X: MatrixLike | Mapping[str, Any],
        y: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
    ) -> ClassicalClassifier:
        """Fit the classifier.

        Args:
            X (MatrixLike | Mapping[str, Any]): features as DataFrame, dict of 1-D arrays, or 2-D array.
            y (ArrayLike | None, optional):  ignored, present here for API consistency by convention.
            sample_weight (ArrayLike | None, optional):  Sample weights. Defaults to None.

        Returns:
            ClassicalClassifier: Instance of itself.

        Raises:
            ValueError: Unknown function string e. g., 'lee-ready'
            ValueError: columns required by a layer are missing.
        """
        self._layers = self.layers if self.layers is not None else []
        for func_str, _ in self._layers:
            if func_str not in ALLOWED_FUNC_STR:
                raise ValueError(
                    f"Unknown function string: {func_str},"
                    f"expected one of {ALLOWED_FUNC_STR}."
                )

        X, columns, xp, device = self._validate_input(X, reset=True, method="fit")

        if sample_weight is not None:
            _check_sample_weight(sample_weight, X)

        # column names are only mandatory, if a layer requires them.
        required_columns = {
            col
            for func_str, subset in self._layers
            for col in _required_columns(func_str, subset)
        }
        missing_columns = sorted(required_columns - set(columns or []))
        if missing_columns:
            raise ValueError(
                f"Expected to find columns: {missing_columns}. Check naming/presenence of columns. See: https://karelze.github.io/tclf/naming_conventions/"
            )

        self.columns_ = columns
        self.func_mapping_ = dict(
            zip(
                ALLOWED_FUNC_STR,
                (
                    self._tick,
                    self._rev_tick,
                    self._quote,
                    self._lr,
                    self._rev_lr,
                    self._emo,
                    self._rev_emo,
                    self._clnv,
                    self._rev_clnv,
                    self._trade_size,
                    self._depth,
                    self._nan,
                ),
            )
        )
        # fitted attributes follow the namespace and device of X.
        self.classes_ = xp.asarray([-1, 1], device=device)
        return self

    def predict(self, X: MatrixLike | Mapping[str, Any]) -> Any:
        """Perform classification on test vectors `X`.

        Args:
            X (MatrixLike | Mapping[str, Any]): feature matrix.

        Returns:
            Any: Predicted target values for X in the namespace and on the device of X.
        """
        pred = self._predict_filled(X, method="predict")
        xp, _ = get_namespace(pred)
        return xp.astype(pred, self.classes_.dtype)

    def _predict_filled(self, X: MatrixLike | Mapping[str, Any], method: str) -> Any:
        """Predict with rule stack and fill unclassified trades by `strategy`.

        Args:
            X (MatrixLike | Mapping[str, Any]): feature matrix.
            method (str): name of the calling public method.

        Returns:
            Any: float predictions.
        """
        check_is_fitted(self)
        X, _, xp, device = self._validate_input(X, reset=False, method=method)

        self.X_ = _Frame(X, self.columns_ or [], xp, device)
        pred = self._predict()
        # reset self.X_ to avoid persisting it
        del self.X_

        # fill NaNs randomly with -1 and 1 or with constant zero
        if self.strategy == "random":
            # draw on host for identical results across namespaces and devices.
            rs = check_random_state(self.random_state)
            fill = rs.choice(np.array([-1, 1], dtype=np.int8), pred.shape[0])
            fill = xp.astype(move_to(fill, xp=xp, device=device), pred.dtype)
        else:
            fill = xp.zeros_like(pred)
        return xp.where(xp.isnan(pred), fill, pred)

    def _predict(self) -> Any:
        """Predict with rule stack.

        Returns:
            Any: prediction
        """
        xp = self.X_.xp
        pred = self.X_.full(xp.nan)
        for func_str, subset in self._layers:
            func = self.func_mapping_[func_str]
            pred = xp.where(
                xp.isnan(pred),
                func(subset=subset),
                pred,
            )
        return pred

    def predict_proba(self, X: MatrixLike | Mapping[str, Any]) -> Any:
        """Predict class probabilities for X.

        Probabilities are either 0 or 1 depending on the class.

        For strategy 'constant' probabilities are (0.5,0.5) for unclassified classes.

        Args:
            X (MatrixLike | Mapping[str, Any]): feature matrix

        Returns:
            Any: probabilities in the namespace and on the device of X.
        """
        pred = self._predict_filled(X, method="predict_proba")
        xp, _ = get_namespace(pred)

        # one-hot encode predictions in order of `classes_` i.e., (-1, 1).
        sell = xp.astype(pred < 0.0, pred.dtype)
        buy = xp.astype(pred > 0.0, pred.dtype)
        # For strategy 'constant' probabilities of unclassified are (0.5,0.5).
        unclassified = xp.reshape(1.0 - sell - buy, (-1, 1))
        return xp.stack([sell, buy], axis=1) + 0.5 * unclassified
