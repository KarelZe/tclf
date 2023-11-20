"""Implements classical trade classification rules with a sklearn-like interface.

Both simple rules like quote rule or tick test or hybrids are included.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _check_sample_weight, check_is_fitted, check_X_y

allowed_func_str = (
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
)

allowed_subsets = ("all", "ex", "best")


class ClassicalClassifier(ClassifierMixin, BaseEstimator):
    """ClassicalClassifier implements several trade classification rules.

    Including:
    * Tick test
    * Reverse tick test
    * Quote rule
    * LR algorithm
    * LR algorithm with reverse tick test
    * EMO algorithm
    * EMO algorithm with reverse tick test
    * CLNV algorithm
    * CLNV algorithm with reverse tick test
    * Trade size rule
    * Depth rule
    * nan

    Args:
    ----
        ClassifierMixin (_type_): ClassifierMixin
        BaseEstimator (_type_): Baseestimator
    """

    def __init__(
        self,
        *,
        layers: list[
            tuple[
                str,
                str,
            ]
        ],
        features: list[str] | None = None,
        random_state: float | None = 42,
        strategy: Literal["random", "const"] = "random",
    ):
        """Initialize a ClassicalClassifier.

        Args:
            layers (List[ tuple[ str, str, ] ]): Layers of classical rule.
            features (List[str] | None, optional): List of feature names in order of
            columns. Required to match columns in feature matrix with label.
            Can be `None`, if `pd.DataFrame` is passed. Defaults to None.
            random_state (float | None, optional): random seed. Defaults to 42.
            strategy (Literal[&quot;random&quot;, &quot;const&quot;], optional): Strategy to fill unclassfied. Randomly with uniform probability or with constant 0. Defaults to &quot;random&quot;.
        """
        self.layers = layers
        self.random_state = random_state
        self.features = features
        self.strategy = strategy

    def _more_tags(self) -> dict[str, bool]:
        """Set tags for sklearn.

        See: https://scikit-learn.org/stable/developers/develop.html#estimator-tags
        """
        # FIXME: Try enabling _skip_test again. Skip tests, as prediction is not
        # invariant and parameters immutable.
        return {
            "allow_nan": True,
            "binary_only": True,
            "_skip_test": True,
            "poor_score": True,
        }

    def _tick(self, subset: Literal["all", "ex"]) -> npt.NDArray:
        """Classify a trade as a buy (sell) if its trade price is above (below) the closest different price of a previous trade.

        Args:
            subset (Literal[&quot;all&quot;, &quot;ex&quot;]): subset i. e.,
            'all' or 'ex'.

        Returns:
            npt.NDArray: result of tick rule. Can be np.NaN.
        """
        return np.where(
            self.X_["TRADE_PRICE"] > self.X_[f"price_{subset}_lag"],
            1,
            np.where(
                self.X_["TRADE_PRICE"] < self.X_[f"price_{subset}_lag"], -1, np.nan
            ),
        )

    def _rev_tick(self, subset: Literal["all", "ex"]) -> npt.NDArray:
        """Classify a trade as a sell (buy) if its trade price is below (above) the closest different price of a subsequent trade.

        Args:
            subset (Literal[&quot;all&quot;, &quot;ex&quot;]): subset i. e.,
            'all' or 'ex'.

        Returns:
            npt.NDArray: result of reverse tick rule. Can be np.NaN.
        """
        return np.where(
            self.X_[f"price_{subset}_lead"] > self.X_["TRADE_PRICE"],
            -1,
            np.where(
                self.X_[f"price_{subset}_lead"] < self.X_["TRADE_PRICE"], 1, np.nan
            ),
        )

    def _quote(self, subset: Literal["best", "ex"]) -> npt.NDArray:
        """Classify a trade as a buy (sell) if its trade price is above (below) the midpoint of the bid and ask spread. Trades executed at the midspread are not classified.

        Args:
            subset (Literal[&quot;ex&quot;, &quot;best&quot;]): subset i. e.,
            'ex' or 'best'.

        Returns:
            npt.NDArray: result of quote rule. Can be np.NaN.
        """
        mid = self._mid(subset)

        return np.where(
            self.X_["TRADE_PRICE"] > mid,
            1,
            np.where(self.X_["TRADE_PRICE"] < mid, -1, np.nan),
        )

    def _lr(self, subset: Literal["best", "ex"]) -> npt.NDArray:
        """Classify a trade as a buy (sell) if its price is above (below) the midpoint (quote rule), and use the tick test (all) to classify midspread trades.

        Adapted from Lee and Ready (1991).

        Args:
            subset (Literal[&quot;ex&quot;, &quot;best&quot;]): subset i. e.,
            'ex' or 'best'.

        Returns:
            npt.ndarray: result of the lee and ready algorithm with tick rule.
            Can be np.NaN.
        """
        q_r = self._quote(subset)
        return np.where(~np.isnan(q_r), q_r, self._tick("all"))

    def _rev_lr(self, subset: Literal["best", "ex"]) -> npt.NDArray:
        """Classify a trade as a buy (sell) if its price is above (below) the midpoint (quote rule), and use the reverse tick test (all) to classify midspread trades.

        Adapted from Lee and Ready (1991).

        Args:
            subset (Literal[&quot;ex&quot;, &quot;best&quot;]): subset i. e.,
            'ex' or 'best'.

        Returns:
            npt.NDArray: result of the lee and ready algorithm with reverse tick
            rule. Can be np.NaN.
        """
        q_r = self._quote(subset)
        return np.where(~np.isnan(q_r), q_r, self._rev_tick("all"))

    def _mid(self, subset: Literal["best", "ex"]) -> npt.NDArray:
        """Calculate the midpoint of the bid and ask spread.

        Midpoint is calculated as the average of the bid and ask spread if the spread is positive. Otherwise, np.NaN is returned.

        Args:
            subset (Literal[&quot;best&quot;, &quot;ex&quot;]): subset i. e.,
            'ex' or 'best'
        Returns:
            npt.NDArray: midpoints. Can be np.NaN.
        """
        return np.where(
            self.X_[f"ask_{subset}"] >= self.X_[f"bid_{subset}"],
            0.5 * (self.X_[f"ask_{subset}"] + self.X_[f"bid_{subset}"]),
            np.nan,
        )

    def _is_at_ask_xor_bid(self, subset: Literal["best", "ex"]) -> pd.Series:
        """Check if the trade price is at the ask xor bid.

        Args:
            subset (Literal[&quot;ex&quot;, &quot;best&quot;]): subset i. e.,
            'ex' or 'best'.

        Returns:
            pd.Series: boolean series with result.
        """
        at_ask = np.isclose(self.X_["TRADE_PRICE"], self.X_[f"ask_{subset}"], atol=1e-4)
        at_bid = np.isclose(self.X_["TRADE_PRICE"], self.X_[f"bid_{subset}"], atol=1e-4)
        return at_ask ^ at_bid

    def _is_at_upper_xor_lower_quantile(
        self, subset: Literal["best", "ex"], quantiles: float = 0.3
    ) -> pd.Series:
        """Check if the trade price is at the ask xor bid.

        Args:
            subset (Literal[&quot;best&quot;, &quot;ex&quot;]): subset i. e., 'ex'.
            quantiles (float, optional): percentage of quantiles. Defaults to 0.3.

        Returns:
            pd.Series: boolean series with result.
        """
        in_upper = (
            (1.0 - quantiles) * self.X_[f"ask_{subset}"]
            + quantiles * self.X_[f"bid_{subset}"]
            <= self.X_["TRADE_PRICE"]
        ) & (self.X_["TRADE_PRICE"] <= self.X_[f"ask_{subset}"])
        in_lower = (self.X_[f"bid_{subset}"] <= self.X_["TRADE_PRICE"]) & (
            self.X_["TRADE_PRICE"]
            <= quantiles * self.X_[f"ask_{subset}"]
            + (1.0 - quantiles) * self.X_[f"bid_{subset}"]
        )
        return in_upper ^ in_lower

    def _emo(self, subset: Literal["best", "ex"]) -> npt.NDArray:
        """Classify a trade as a buy (sell) if the trade takes place at the ask (bid) quote, and use the tick test (all) to classify all other trades.

        Adapted from Ellis et al. (2000).

        Args:
            subset (Literal[&quot;ex&quot;, &quot;best&quot;]): subset i. e.,
            'ex' or 'best'.

        Returns:
            npt.NDArray: result of the emo algorithm with tick rule. Can be
            np.NaN.
        """
        return np.where(
            self._is_at_ask_xor_bid(subset), self._quote(subset), self._tick("all")
        )

    def _rev_emo(self, subset: Literal["best", "ex"]) -> npt.NDArray:
        """Classify a trade as a buy (sell) if the trade takes place at the ask (bid) quote, and use the reverse tick test (all) to classify all other trades.

        Adapted from Grauer et al. (2022).

        Args:
            subset (Literal[&quot;ex&quot;, &quot;best&quot;]): subset
            i. e., 'ex' or 'best'.

        Returns:
            npt.NDArray: result of the emo algorithm with reverse tick rule.
            Can be np.NaN.
        """
        return np.where(
            self._is_at_ask_xor_bid(subset), self._quote(subset), self._rev_tick("all")
        )

    def _clnv(self, subset: Literal["best", "ex"]) -> npt.NDArray:
        """Classify a trade based on deciles of the bid and ask spread.

        Spread is divided into ten deciles and trades are classified as follows:
        - use quote rule for at ask until 30 % below ask (upper 3 deciles)
        - use quote rule for at bid until 30 % above bid (lower 3 deciles)
        - use tick rule (all) for all other trades (±2 deciles from midpoint; outside
        bid or ask).

        Adapted from Chakrabarty et al. (2007).

        Args:
            subset (Literal[&quot;ex&quot;, &quot;best&quot;]): subset i. e.,
            'ex' or 'best'.

        Returns:
            npt.NDArray: result of the emo algorithm with tick rule. Can be
            np.NaN.
        """
        return np.where(
            self._is_at_upper_xor_lower_quantile(subset),
            self._quote(subset),
            self._tick("all"),
        )

    def _rev_clnv(self, subset: Literal["best", "ex"]) -> npt.NDArray:
        """Classify a trade based on deciles of the bid and ask spread.

        Spread is divided into ten deciles and trades are classified as follows:
        - use quote rule for at ask until 30 % below ask (upper 3 deciles)
        - use quote rule for at bid until 30 % above bid (lower 3 deciles)
        - use reverse tick rule (all) for all other trades (±2 deciles from midpoint;
        outside bid or ask).

        Similar to extension of emo algorithm proposed Grauer et al. (2022).

        Args:
            subset (Literal[&quot;ex&quot;, &quot;best&quot;]): subset i. e.,
            'ex' or 'best'.

        Returns:
            npt.NDArray: result of the emo algorithm with tick rule. Can be
            np.NaN.
        """
        return np.where(
            self._is_at_upper_xor_lower_quantile(subset),
            self._quote(subset),
            self._rev_tick("all"),
        )

    def _trade_size(self, *args: Any) -> npt.NDArray:
        """Classify a trade as a buy (sell) the trade size matches exactly either the bid (ask) quote size.

        Adapted from Grauer et al. (2022).

        Returns:
            npt.NDArray: result of the trade size rule. Can be np.NaN.
        """
        bid_eq_ask = np.isclose(
            self.X_["ask_size_ex"], self.X_["bid_size_ex"], atol=1e-4
        )

        ts_eq_bid = (
            np.isclose(self.X_["TRADE_SIZE"], self.X_["bid_size_ex"], atol=1e-4)
            & ~bid_eq_ask
        )
        ts_eq_ask = (
            np.isclose(self.X_["TRADE_SIZE"], self.X_["ask_size_ex"], atol=1e-4)
            & ~bid_eq_ask
        )

        return np.where(ts_eq_bid, 1, np.where(ts_eq_ask, -1, np.nan))

    def _depth(self, subset: Literal["best", "ex"]) -> npt.NDArray:
        """Classify midspread trades as buy (sell), if the ask size (bid size) exceeds the bid size (ask size).

        Adapted from Grauer et al. (2022).

        Args:
            subset (Literal[&quot;best&quot;, &quot;ex&quot;]): subset

        Returns:
            npt.NDArray: result of depth rule. Can be np.NaN.
        """
        at_mid = np.isclose(self._mid(subset), self.X_["TRADE_PRICE"], atol=1e-4)

        return np.where(
            at_mid & (self.X_["ask_size_ex"] > self.X_["bid_size_ex"]),
            1,
            np.where(
                at_mid & (self.X_["ask_size_ex"] < self.X_["bid_size_ex"]),
                -1,
                np.nan,
            ),
        )

    def _nan(self, *args: Any) -> npt.NDArray:
        """Classify nothing. Fast forward results from previous classifier.

        Returns:
            npt.NDArray: result of the trade size rule. Can be np.NaN.
        """
        return np.full(shape=(self.X_.shape[0],), fill_value=np.nan)

    def fit(
        self,
        X: npt.NDArray | pd.DataFrame,
        y: npt.NDArray | pd.Series,
        sample_weight: npt.NDArray | None = None,
    ) -> ClassicalClassifier:
        """Fit the classifier.

        Args:
            X (npt.NDArray | pd.DataFrame): features
            y (npt.NDArray | pd.Series): ground truth (ignored)
            sample_weight (npt.NDArray | None, optional):  Sample weights.
            Defaults to None.

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

        self.func_mapping_ = dict(zip(allowed_func_str, funcs))

        # create working copy to be altered and try to get columns from df
        self.columns_ = self.features
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns.tolist()

        check_classification_targets(y)

        X, y = check_X_y(
            X, y, multi_output=False, accept_sparse=False, force_all_finite=False
        )

        # FIXME: make flexible if open-sourced
        # self.classes_ = np.unique(y)
        self.classes_ = np.array([-1, 1])

        # if no features are provided or inferred, use default
        if not self.columns_:
            self.columns_ = [str(i) for i in range(X.shape[1])]

        if len(self.columns_) > 0 and X.shape[1] != len(self.columns_):
            raise ValueError(
                f"Expected {len(self.columns_)} columns, got {X.shape[1]}."
            )

        for func_str, subset in self.layers:
            if subset not in allowed_subsets:
                raise ValueError(
                    f"Unknown subset: {subset}, expected one of {allowed_subsets}."
                )
            if func_str not in allowed_func_str:
                raise ValueError(
                    f"Unknown function string: {func_str},"
                    f"expected one of {allowed_func_str}."
                )

        return self

    def predict(self, X: npt.NDArray | pd.DataFrame) -> npt.NDArray:
        """Perform classification on test vectors `X`.

        Args:
            X (npt.NDArray | pd.DataFrame): feature matrix.

        Returns:
            npt.NDArray: Predicted traget values for X.
        """
        check_is_fitted(self)

        rs = check_random_state(self.random_state)

        self.X_ = pd.DataFrame(data=X, columns=self.columns_)

        mapping_cols = {"BEST_ASK": "ask_best", "BEST_BID": "bid_best"}

        self.X_ = self.X_.rename(columns=mapping_cols)

        pred = np.full(shape=(X.shape[0],), fill_value=np.nan)

        for func_str, subset in self.layers:
            func = self.func_mapping_[func_str]
            pred = np.where(
                np.isnan(pred),
                func(subset),
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

    def predict_proba(self, X: npt.NDArray | pd.DataFrame) -> npt.NDArray:
        """Predict class probabilities for X.

        Probabilities are either 0 or 1 depending on the class.

        For strategy 'constant' probabilities are (0.5,0.5) for unclassified classes.

        Args:
            X (npt.NDArray | pd.DataFrame): feature matrix

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
        indices = np.where(preds[mask, None] == self.classes_[None, :])[1]
        n_classes = np.max(self.classes_) + 1

        # overwrite defaults with one-hot encoded classes.
        # For strategy 'constant' probabilities are (0.5,0.5).
        prob[mask] = np.identity(n_classes)[indices]
        return prob
