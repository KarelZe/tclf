"""Implements classical trade classification rules with a sklearn-like interface.

Both simple rules like quote rule or tick test or hybrids are included.
"""

from __future__ import annotations

from typing import Literal, get_args

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
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
            >>> X = pd.DataFrame(
            ... [
            ...     [1.5, 1, 3],
            ...     [2.5, 1, 3],
            ...     [1.5, 3, 1],
            ...     [2.5, 3, 1],
            ...     [1, np.nan, 1],
            ...     [3, np.nan, np.nan],
            ... ],
            ... columns=["trade_price", "bid_ex", "ask_ex"],
            ... )
            >>> clf = ClassicalClassifier(layers=[("quote", "ex")], strategy="const")
            >>> clf.fit(X)
            ClassicalClassifier(layers=[('quote', 'ex')], strategy='const')
            >>> pred = clf.predict_proba(X)

        Args:
            layers (List[tuple[ALLOWED_FUNC_LITERALS, str]]): Layers of classical rule and subset name. Supported rules: "tick", "rev_tick", "quote", "lr", "rev_lr", "emo", "rev_emo", "trade_size", "depth", and "nan". Defaults to None, which results in classification by 'strategy' parameter.
            features (List[str] | None, optional): List of feature names in order of columns. Required to match columns in feature matrix with label. Can be `None`, if `pd.DataFrame` is passed. Defaults to None.
            random_state (float | None, optional): random seed. Defaults to 42.
            strategy (Literal[&quot;random&quot;, &quot;const&quot;], optional): Strategy to fill unclassfied. Randomly with uniform probability or with constant 0. Defaults to &quot;random&quot;.
        """
        self.layers = layers
        self.random_state = random_state
        self.features = features
        self.strategy = strategy

    def _more_tags(self) -> dict[str, bool | dict[str, str]]:
        """Set tags for sklearn.

        See: https://scikit-learn.org/stable/developers/develop.html#estimator-tags
        """
        return {
            "allow_nan": True,
            "binary_only": True,
            "requires_y": False,
            "poor_score": True,
            "_xfail_checks": {
                "check_classifiers_classes": "Disabled due to partly random classification.",
                "check_classifiers_train": "No check, as unsupervised classifier.",
                "check_classifiers_one_label": "Disabled due to partly random classification.",
                "check_methods_subset_invariance": "No check, as unsupervised classifier.",
                "check_methods_sample_order_invariance": "No check, as unsupervised classifier.",
                "check_supervised_y_no_nan": "No check, as unsupervised classifier.",
                "check_supervised_y_2d": "No check, as unsupervised classifier.",
                "check_classifiers_regression_target": "No check, as unsupervised classifier.",
            },
        }

    def _tick(self, subset: str) -> npt.NDArray:
        """Classify a trade as a buy (sell) if its trade price is above (below) the closest different price of a previous trade.

        Args:
            subset (str): subset i.e., 'all' or 'ex'.

        Returns:
            npt.NDArray: result of tick rule. Can be np.NaN.
        """
        return np.where(
            self.X_["trade_price"] > self.X_[f"price_{subset}_lag"],
            1,
            np.where(
                self.X_["trade_price"] < self.X_[f"price_{subset}_lag"], -1, np.nan
            ),
        )

    def _rev_tick(self, subset: str) -> npt.NDArray:
        """Classify a trade as a sell (buy) if its trade price is below (above) the closest different price of a subsequent trade.

        Args:
            subset (str): subset i.e.,'all' or 'ex'.

        Returns:
            npt.NDArray: result of reverse tick rule. Can be np.NaN.
        """
        return np.where(
            self.X_[f"price_{subset}_lead"] > self.X_["trade_price"],
            -1,
            np.where(
                self.X_[f"price_{subset}_lead"] < self.X_["trade_price"], 1, np.nan
            ),
        )

    def _quote(self, subset: str) -> npt.NDArray:
        """Classify a trade as a buy (sell) if its trade price is above (below) the midpoint of the bid and ask spread. Trades executed at the midspread are not classified.

        Args:
            subset (str): subset i.e., 'ex' or 'best'.

        Returns:
            npt.NDArray: result of quote rule. Can be np.NaN.
        """
        mid = self._mid(subset)

        return np.where(
            self.X_["trade_price"] > mid,
            1,
            np.where(self.X_["trade_price"] < mid, -1, np.nan),
        )

    def _lr(self, subset: str) -> npt.NDArray:
        """Classify a trade as a buy (sell) if its price is above (below) the midpoint (quote rule), and use the tick test to classify midspread trades.

        Adapted from Lee and Ready (1991).

        Args:
            subset (str): subset i.e., 'ex' or 'best'.

        Returns:
            npt.ndarray: result of the lee and ready algorithm with tick rule.
            Can be np.NaN.
        """
        q_r = self._quote(subset)
        return np.where(~np.isnan(q_r), q_r, self._tick(subset))

    def _rev_lr(self, subset: str) -> npt.NDArray:
        """Classify a trade as a buy (sell) if its price is above (below) the midpoint (quote rule), and use the reverse tick test to classify midspread trades.

        Adapted from Lee and Ready (1991).

        Args:
            subset (str): subset i.e.,'ex' or 'best'.

        Returns:
            npt.NDArray: result of the lee and ready algorithm with reverse tick
            rule. Can be np.NaN.
        """
        q_r = self._quote(subset)
        return np.where(~np.isnan(q_r), q_r, self._rev_tick(subset))

    def _mid(self, subset: str) -> npt.NDArray:
        """Calculate the midpoint of the bid and ask spread.

        Midpoint is calculated as the average of the bid and ask spread if the spread is positive. Otherwise, np.NaN is returned.

        Args:
            subset (str): subset i.e.,
            'ex' or 'best'
        Returns:
            npt.NDArray: midpoints. Can be np.NaN.
        """
        return np.where(
            self.X_[f"ask_{subset}"] >= self.X_[f"bid_{subset}"],
            0.5 * (self.X_[f"ask_{subset}"] + self.X_[f"bid_{subset}"]),
            np.nan,
        )

    def _is_at_ask_xor_bid(self, subset: str) -> pd.Series:
        """Check if the trade price is at the ask xor bid.

        Args:
            subset (str): subset i.e.,
            'ex' or 'best'.

        Returns:
            pd.Series: boolean series with result.
        """
        at_ask = np.isclose(self.X_["trade_price"], self.X_[f"ask_{subset}"], atol=1e-4)
        at_bid = np.isclose(self.X_["trade_price"], self.X_[f"bid_{subset}"], atol=1e-4)
        return at_ask ^ at_bid

    def _is_at_upper_xor_lower_quantile(
        self, subset: str, quantiles: float = 0.3
    ) -> pd.Series:
        """Check if the trade price is at the ask xor bid.

        Args:
            subset (str): subset i.e., 'ex'.
            quantiles (float, optional): percentage of quantiles. Defaults to 0.3.

        Returns:
            pd.Series: boolean series with result.
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

    def _emo(self, subset: str) -> npt.NDArray:
        """Classify a trade as a buy (sell) if the trade takes place at the ask (bid) quote, and use the tick test to classify all other trades.

        Adapted from Ellis et al. (2000).

        Args:
            subset (Literal[&quot;ex&quot;, &quot;best&quot;]): subset i.e., 'ex' or 'best'.

        Returns:
            npt.NDArray: result of the emo algorithm with tick rule. Can be
            np.NaN.
        """
        return np.where(
            self._is_at_ask_xor_bid(subset), self._quote(subset), self._tick(subset)
        )

    def _rev_emo(self, subset: str) -> npt.NDArray:
        """Classify a trade as a buy (sell) if the trade takes place at the ask (bid) quote, and use the reverse tick test to classify all other trades.

        Adapted from Grauer et al. (2022).

        Args:
            subset (str): subset i.e., 'ex' or 'best'.

        Returns:
            npt.NDArray: result of the emo algorithm with reverse tick rule.
            Can be np.NaN.
        """
        return np.where(
            self._is_at_ask_xor_bid(subset), self._quote(subset), self._rev_tick(subset)
        )

    def _clnv(self, subset: str) -> npt.NDArray:
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
            npt.NDArray: result of the emo algorithm with tick rule. Can be
            np.NaN.
        """
        return np.where(
            self._is_at_upper_xor_lower_quantile(subset),
            self._quote(subset),
            self._tick(subset),
        )

    def _rev_clnv(self, subset: str) -> npt.NDArray:
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
            npt.NDArray: result of the emo algorithm with tick rule. Can be
            np.NaN.
        """
        return np.where(
            self._is_at_upper_xor_lower_quantile(subset),
            self._quote(subset),
            self._rev_tick(subset),
        )

    def _trade_size(self, subset: str) -> npt.NDArray:
        """Classify a trade as a buy (sell) the trade size matches exactly either the bid (ask) quote size.

        Adapted from Grauer et al. (2022).

        Args:
            subset (str): subset i.e., 'ex' or 'best'.

        Returns:
            npt.NDArray: result of the trade size rule. Can be np.NaN.
        """
        bid_eq_ask = np.isclose(
            self.X_[f"ask_size_{subset}"], self.X_[f"bid_size_{subset}"], atol=1e-4
        )

        ts_eq_bid = (
            np.isclose(self.X_["trade_size"], self.X_[f"bid_size_{subset}"], atol=1e-4)
            & ~bid_eq_ask
        )
        ts_eq_ask = (
            np.isclose(self.X_["trade_size"], self.X_[f"ask_size_{subset}"], atol=1e-4)
            & ~bid_eq_ask
        )

        return np.where(ts_eq_bid, 1, np.where(ts_eq_ask, -1, np.nan))

    def _depth(self, subset: str) -> npt.NDArray:
        """Classify midspread trades as buy (sell), if the ask size (bid size) exceeds the bid size (ask size).

        Adapted from Grauer et al. (2022).

        Args:
            subset (str): subset i.e., 'ex' or 'best'.

        Returns:
            npt.NDArray: result of depth rule. Can be np.NaN.
        """
        at_mid = np.isclose(self._mid(subset), self.X_["trade_price"], atol=1e-4)

        return np.where(
            at_mid & (self.X_[f"ask_size_{subset}"] > self.X_[f"bid_size_{subset}"]),
            1,
            np.where(
                at_mid
                & (self.X_[f"ask_size_{subset}"] < self.X_[f"bid_size_{subset}"]),
                -1,
                np.nan,
            ),
        )

    def _nan(self, subset: str) -> npt.NDArray:
        """Classify nothing. Fast forward results from previous classifier.

        Returns:
            npt.NDArray: result of the trade size rule. Can be np.NaN.
        """
        return np.full(shape=(self.X_.shape[0],), fill_value=np.nan)

    def _validate_columns(self, found_cols: list[str]) -> None:
        """Validate if all required columns are present.

        Args:
            found_cols (list[str]): columns present in dataframe.
        """

        def lookup_columns(func_str: str, sub: str) -> list[str]:
            LR_LIKE = [
                "trade_price",
                f"price_{sub}_lag",
                f"ask_{sub}",
                f"bid_{sub}",
            ]
            REV_LR_LIKE = [
                "trade_price",
                f"price_{sub}_lead",
                f"ask_{sub}",
                f"bid_{sub}",
            ]

            LUT_REQUIRED_COLUMNS: dict[str, list[str]] = {
                "nan": [],
                "clnv": LR_LIKE,
                "depth": [
                    "trade_price",
                    f"ask_{sub}",
                    f"bid_{sub}",
                    f"ask_size_{sub}",
                    f"bid_size_{sub}",
                ],
                "emo": LR_LIKE,
                "lr": LR_LIKE,
                "quote": ["trade_price", f"ask_{sub}", f"bid_{sub}"],
                "rev_clnv": REV_LR_LIKE,
                "rev_emo": REV_LR_LIKE,
                "rev_lr": REV_LR_LIKE,
                "rev_tick": ["trade_price", f"price_{sub}_lead"],
                "tick": ["trade_price", f"price_{sub}_lag"],
                "trade_size": ["trade_size", f"ask_size_{sub}", f"bid_size_{sub}"],
            }
            return LUT_REQUIRED_COLUMNS[func_str]

        required_cols_set = set()
        for func_str, sub in self._layers:
            func_col = lookup_columns(func_str, sub)
            required_cols_set.update(func_col)

        missing_cols = sorted(required_cols_set - set(found_cols))
        if missing_cols:
            raise ValueError(
                f"Expected to find columns: {missing_cols}. Check naming/presenence of columns. See: https://karelze.github.io/tclf/naming_conventions/"
            )

    def fit(
        self,
        X: MatrixLike,
        y: ArrayLike | None = None,
        sample_weight: npt.NDArray | None = None,
    ) -> ClassicalClassifier:
        """Fit the classifier.

        Args:
            X (MatrixLike): features
            y (ArrayLike | None, optional):  ignored, present here for API consistency by convention.
            sample_weight (npt.NDArray | None, optional):  Sample weights. Defaults to None.

        Raises:
            ValueError: Unknown subset e. g., 'ise'
            ValueError: Unknown function string e. g., 'lee-ready'
            ValueError: Multi output is not supported.

        Returns:
            ClassicalClassifier: Instance of itself.
        """
        _check_sample_weight(sample_weight, X)

        funcs = (
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
        )

        self.func_mapping_ = dict(zip(ALLOWED_FUNC_STR, funcs))

        # create working copy to be altered and try to get columns from df
        self.columns_ = self.features
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns.tolist()

        X = self._validate_data(
            X,
            y="no_validation",
            dtype=[np.float64, np.float32],
            accept_sparse=False,
            force_all_finite=False,
        )

        self.classes_ = np.array([-1, 1])

        # if no features are provided or inferred, use default
        if not self.columns_:
            self.columns_ = [str(i) for i in range(X.shape[1])]

        if len(self.columns_) > 0 and X.shape[1] != len(self.columns_):
            raise ValueError(
                f"Expected {len(self.columns_)} columns, got {X.shape[1]}."
            )

        self._layers = self.layers if self.layers is not None else []
        for func_str, _ in self._layers:
            if func_str not in ALLOWED_FUNC_STR:
                raise ValueError(
                    f"Unknown function string: {func_str},"
                    f"expected one of {ALLOWED_FUNC_STR}."
                )

        columns = self.columns_
        self._validate_columns(columns)

        return self

    def predict(self, X: MatrixLike) -> npt.NDArray:
        """Perform classification on test vectors `X`.

        Args:
            X (MatrixLike): feature matrix.

        Returns:
            npt.NDArray: Predicted traget values for X.
        """
        check_is_fitted(self)
        X = self._validate_data(
            X,
            dtype=[np.float64, np.float32],
            accept_sparse=False,
            force_all_finite=False,
        )

        rs = check_random_state(self.random_state)

        self.X_ = pd.DataFrame(data=X, columns=self.columns_)
        pred = np.full(shape=(X.shape[0],), fill_value=np.nan)

        for func_str, subset in self._layers:
            func = self.func_mapping_[func_str]
            pred = np.where(
                np.isnan(pred),
                func(subset=subset),
                pred,
            )

        # fill NaNs randomly with -1 and 1 or with constant zero
        mask = np.isnan(pred)
        if self.strategy == "random":
            pred[mask] = rs.choice(self.classes_, pred.shape)[mask]
        else:
            pred[mask] = np.zeros(pred.shape)[mask]

        # reset self.X_ to avoid persisting it
        del self.X_
        return pred

    def predict_proba(self, X: MatrixLike) -> npt.NDArray:
        """Predict class probabilities for X.

        Probabilities are either 0 or 1 depending on the class.

        For strategy 'constant' probabilities are (0.5,0.5) for unclassified classes.

        Args:
            X (MatrixLike): feature matrix

        Returns:
            npt.NDArray: probabilities
        """
        # assign 0.5 to all classes. Required for strategy 'constant'.
        prob = np.full((len(X), 2), 0.5)

        # Class can be assumed to be -1 or 1 for strategy 'random'.
        # Class might be zero though for strategy constant. Mask non-zeros.
        preds = self.predict(X)
        mask = np.flatnonzero(preds)

        # get index of predicted class and one-hot encode it
        indices = np.nonzero(preds[mask, None] == self.classes_[None, :])[1]
        n_classes = np.max(self.classes_) + 1

        # overwrite defaults with one-hot encoded classes.
        # For strategy 'constant' probabilities are (0.5,0.5).
        prob[mask] = np.identity(n_classes)[indices]
        return prob
