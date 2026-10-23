"""Tests for the classical classifier."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn import config_context
from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import check_is_fitted

from tclf.classical_classifier import (
    ALLOWED_FUNC_LITERALS,
    ALLOWED_FUNC_STR,
    ClassicalClassifier,
)

UNSUPERVISED = "No check, as unsupervised classifier."
RANDOM = "Disabled due to partly random classification."
EXPECTED_FAILED_CHECKS = {
    "check_classifiers_classes": RANDOM,
    "check_classifiers_one_label": RANDOM,
    "check_classifiers_train": UNSUPERVISED,
    "check_classifiers_regression_target": UNSUPERVISED,
    "check_classifier_not_supporting_multiclass": UNSUPERVISED,
    "check_methods_subset_invariance": UNSUPERVISED,
    "check_methods_sample_order_invariance": UNSUPERVISED,
    "check_supervised_y_no_nan": UNSUPERVISED,
    "check_supervised_y_2d": UNSUPERVISED,
}

# (namespace, device, dtype) combinations for array API tests. Skipped if unavailable.
ARRAY_API_PARAMS = [
    pytest.param("array_api_strict", "CPU_DEVICE", "float64", id="strict-cpu"),
    pytest.param("array_api_strict", "device1", "float32", id="strict-device1"),
    pytest.param("torch", "cpu", "float64", id="torch-cpu"),
    pytest.param("torch", "mps", "float32", id="torch-mps"),
    pytest.param("torch", "cuda", "float64", id="torch-cuda"),
]


def _get_namespace(name: str, device_name: str) -> tuple[Any, Any]:
    """Get array namespace and device or skip test, if unavailable.

    Args:
        name (str): name of namespace e. g., 'torch'.
        device_name (str): name of device e. g., 'cpu'.

    Returns:
        tuple[Any, Any]: namespace and device.
    """
    xp = pytest.importorskip(name)
    if name == "array_api_strict":
        return xp, xp.Device(device_name)
    if device_name == "cuda" and not xp.cuda.is_available():
        pytest.skip("CUDA not available.")
    if device_name == "mps" and not xp.backends.mps.is_available():
        pytest.skip("MPS not available.")
    return xp, xp.device(device_name)


def _to_numpy(array: Any) -> np.ndarray:
    """Convert an array of any namespace and device to numpy.

    Args:
        array (Any): array.

    Returns:
        np.ndarray: numpy array.
    """
    if hasattr(array, "cpu"):
        return array.cpu().numpy()
    if hasattr(array, "to_device"):
        import array_api_strict

        array = array.to_device(array_api_strict.Device("CPU_DEVICE"))
    return np.asarray(array)


class TestClassicalClassifier:
    """Perform automated tests for ClassicalClassifier.

    Args:
        unittest (_type_): unittest module
    """

    @pytest.fixture
    def x_train(self) -> pd.DataFrame:
        """Training set fixture.

        Returns:
            pd.DataFrame: training set
        """
        return pd.DataFrame(
            np.zeros(shape=(1, 14)),
            columns=[
                "ask_size_ex",
                "bid_size_ex",
                "ask_best",
                "bid_best",
                "ask_ex",
                "bid_ex",
                "trade_price",
                "trade_size",
                "price_ex_lag",
                "price_ex_lead",
                "price_best_lag",
                "price_best_lead",
                "price_all_lag",
                "price_all_lead",
            ],
        )

    @pytest.fixture
    def x_test(self) -> pd.DataFrame:
        """Test set fixture.

        Returns:
            pd.DataFrame: test set
        """
        return pd.DataFrame(
            [[1, 2], [3, 4], [1, 2], [3, 4]],
            columns=["ask_best", "bid_best"],
        )

    @pytest.fixture
    def y_test(self) -> pd.Series:
        """Test target fixture.

        Returns:
            pd.Series: test target
        """
        return pd.Series([1, -1, 1, -1])

    @pytest.fixture
    def clf(self, x_train: pd.DataFrame) -> ClassicalClassifier:
        """Classifier fixture with random classification.

        Args:
            x_train (pd.DataFrame): train set

        Returns:
            ClassicalClassifier: fitted clf
        """
        return ClassicalClassifier().fit(x_train[["ask_best", "bid_best"]])

    @parametrize_with_checks(
        [ClassicalClassifier()],
        expected_failed_checks=lambda _: EXPECTED_FAILED_CHECKS,
    )
    def test_sklearn_compatibility(
        self, estimator: BaseEstimator, check: Callable
    ) -> None:
        """Test, if classifier is compatible with sklearn."""
        check(estimator)

    @pytest.fixture
    def x_rand(self) -> pd.DataFrame:
        """Random test set with all columns, ties, nans and infs.

        Returns:
            pd.DataFrame: test set
        """
        rng = np.random.default_rng(0)
        n = 500
        data: dict[str, np.ndarray] = {
            "trade_price": rng.choice([1.0, 1.1, 1.2, 1.3], n),
            "trade_size": rng.choice([1.0, 2.0, 3.0], n),
        }
        for subset in ("ex", "best", "all"):
            bid = rng.choice([1.0, 1.1, 1.2], n)
            data[f"bid_{subset}"] = bid
            data[f"ask_{subset}"] = bid + rng.choice([-0.1, 0.0, 0.1, 0.2], n)
            data[f"bid_size_{subset}"] = rng.choice([1.0, 2.0, 3.0], n)
            data[f"ask_size_{subset}"] = rng.choice([1.0, 2.0, 3.0], n)
            data[f"price_{subset}_lag"] = rng.choice([1.0, 1.1, 1.2, 1.3], n)
            data[f"price_{subset}_lead"] = rng.choice([1.0, 1.1, 1.2, 1.3], n)
        x = pd.DataFrame(data)
        # inject missing and infinite values
        mask = rng.random(x.shape) < 0.05
        return x.mask(mask, rng.choice([np.nan, np.inf, -np.inf], x.shape))

    @staticmethod
    def _all_layers() -> list[tuple[ALLOWED_FUNC_LITERALS, str]]:
        """All rules on all subsets, to exercise every code path.

        Returns:
            list[tuple[ALLOWED_FUNC_LITERALS, str]]: layers
        """
        return [
            (func_str, subset)
            for func_str in ALLOWED_FUNC_STR
            for subset in ("ex", "best", "all")
        ]

    def test_shapes(
        self, clf: ClassicalClassifier, x_test: pd.DataFrame, y_test: pd.Series
    ) -> None:
        """Test, if shapes of the classifier equal the targets.

        Shapes are usually [no. of samples, 1].
        """
        y_pred = clf.predict(x_test)

        assert y_test.shape == y_pred.shape

    def test_proba(self, clf: ClassicalClassifier, x_test: pd.DataFrame) -> None:
        """Test, if probabilities are in [0, 1]."""
        y_pred = clf.predict_proba(x_test)
        assert (y_pred >= 0).all()
        assert (y_pred <= 1).all()

    def test_score(
        self, clf: ClassicalClassifier, x_test: pd.DataFrame, y_test: pd.Series
    ) -> None:
        """Test, if score is correctly calculated..

        For a random classification i. e., `layers=[("nan", "ex")]`, the score
        should be around 0.5.
        """
        accuracy = clf.score(x_test, y_test)
        assert 0.0 <= accuracy <= 1.0

    def test_random_state(self, x_train: pd.DataFrame, x_test: pd.DataFrame) -> None:
        """Test, if random state is correctly set.

        Two classifiers with the same random state should give the same results.
        """
        columns = ["ask_best", "bid_best"]
        first_classifier = ClassicalClassifier(
            layers=[("nan", "ex")],
            random_state=50,
        ).fit(x_train[columns])
        first_y_pred = first_classifier.predict(x_test)

        second_classifier = ClassicalClassifier(
            layers=[("nan", "ex")],
            random_state=50,
        ).fit(x_train[columns])
        second_y_pred = second_classifier.predict(x_test)

        assert (first_y_pred == second_y_pred).all()

    def test_fit(self, x_train: pd.DataFrame) -> None:
        """Test, if fit works.

        A fitted classifier should have an attribute `layers_`.
        """
        fitted_classifier = ClassicalClassifier(
            layers=[("nan", "ex")],
            random_state=42,
        ).fit(x_train[["ask_best", "bid_best"]])
        assert check_is_fitted(fitted_classifier) is None

    def test_strategy_const(self, x_train: pd.DataFrame, x_test: pd.DataFrame) -> None:
        """Test, if strategy 'const' returns correct probabilities.

        A classifier with strategy 'constant' should return class probabilities
        of (0.5, 0.5), if a trade can not be classified.
        """
        columns = ["ask_best", "bid_best"]
        fitted_classifier = ClassicalClassifier(
            layers=[("nan", "ex")], strategy="const"
        ).fit(x_train[columns])
        assert_allclose(
            fitted_classifier.predict_proba(x_test[columns]),
            0.5,
            rtol=1e-09,
            atol=1e-09,
        )

    def test_invalid_func(self, x_train: pd.DataFrame) -> None:
        """Test, if only valid function strings can be passed.

        An exception should be raised for invalid function strings.
        Test for 'foo', which is no valid rule.
        """
        classifier = ClassicalClassifier(
            layers=[("foo", "all")],  # ty: ignore[invalid-argument-type]  # intentional invalid input
            random_state=42,
        )
        with pytest.raises(ValueError, match=r"Unknown function string"):
            classifier.fit(x_train)

    def test_missing_columns(self, x_train: pd.DataFrame) -> None:
        """Test, if an error is raised, if required columns are missing.

        An exception should be raised if required features are missing,
        including the columns required for classification.
        """
        classifier = ClassicalClassifier(
            layers=[("tick", "all"), ("quote", "ex")], random_state=42
        )
        with pytest.raises(
            ValueError,
            match=r"Expected to find columns: ['ask_ex', 'bid_ex', 'price_all_lag']*",
        ):
            classifier.fit(x_train[["trade_price", "trade_size"]])

    def test_invalid_col_length(self, x_train: pd.DataFrame) -> None:
        """Test, if only valid column length can be passed.

        An exception should be raised if length of columns list does not match
        the number of columns in the data. `features` is only used if, data is
        not passed as `pd.DataFrame`.Test for columns list of length 2, which
        does not match the data.
        """
        classifier = ClassicalClassifier(
            layers=[("tick", "all")], random_state=42, features=["one"]
        )
        with pytest.raises(ValueError, match=r"Expected"):
            classifier.fit(x_train.to_numpy())

    def test_override(self, x_train: pd.DataFrame) -> None:
        """Test, if classifier does not override valid results from layer one.

        If all data can be classified using first rule, first rule should
        only be applied.
        """
        columns = ["trade_price", "price_ex_lag", "price_all_lead"]
        x_test = pd.DataFrame(
            [[1, 2, 0], [2, 1, 3]],
            columns=columns,
        )
        y_test = pd.Series([-1, 1])
        y_pred = (
            ClassicalClassifier(
                layers=[("tick", "ex"), ("rev_tick", "all")],
                random_state=7,
            )
            .fit(x_train[columns])
            .predict(x_test)
        )
        assert (y_pred == y_test).all()

    def test_np_array(self, x_train: pd.DataFrame) -> None:
        """Test, if classifier works, if only np.ndarrays are provided.

        If only np.ndarrays are provided, the classifier should work, by constructing
        a dataframe from the arrays and the `columns` list.
        """
        x_test = np.array([[1, 2, 0], [2, 1, 3]])
        y_test = np.array([-1, 1])

        columns = ["trade_price", "price_ex_lag", "price_ex_lead"]
        y_pred = (
            ClassicalClassifier(
                layers=[("tick", "ex"), ("rev_tick", "ex")],
                random_state=7,
                features=columns,
            )
            .fit(x_train[columns].to_numpy())
            .predict(x_test)
        )
        assert (y_pred == y_test).all()

    @pytest.mark.parametrize("subset", ["best", "ex"])
    def test_mid(self, x_train: pd.DataFrame, subset: str) -> None:
        """Test, if no mid is calculated, if bid exceeds ask etc.

        Args:
            x_train (pd.DataFrame): train set
            subset (str): subset
        """
        columns = ["trade_price", f"bid_{subset}", f"ask_{subset}"]

        # first two by rule, all other by random chance.
        x_test = pd.DataFrame(
            [
                [1.5, 1, 3],
                [2.5, 1, 3],
                [1.5, 3, 1],  # bid > ask
                [2.5, 3, 1],  # bid > ask
                [1, np.nan, 1],  # missing data
                [3, np.nan, np.nan],  # missing_data
            ],
            columns=columns,
        )
        y_test = pd.Series([-1, 1, 1, -1, -1, 1])
        y_pred = (
            ClassicalClassifier(layers=[("quote", subset)], random_state=45)
            .fit(x_train[columns])
            .predict(x_test)
        )
        assert (y_pred == y_test).all()

    def _apply_rule(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        layers: list[tuple[ALLOWED_FUNC_LITERALS, str]],
        random_state: int = 7,
    ) -> None:
        """Apply rule-based classification.

        Args:
            x_train (pd.DataFrame): training features
            x_test (pd.DataFrame): test features
            y_test (pd.Series): true labels
            layers (list[tuple[ALLOWED_FUNC_LITERALS, str]]): layers
            random_state (int, optional): random state. Defaults to 7.
        """
        y_pred = (
            ClassicalClassifier(
                layers=layers,
                random_state=random_state,
            )
            .fit(x_train[x_test.columns])
            .predict(x_test)
        )
        assert (y_pred == y_test).all()

    @pytest.mark.benchmark
    @pytest.mark.parametrize("subset", ["all", "ex"])
    def test_tick_rule(self, x_train: pd.DataFrame, subset: str) -> None:
        """Test, if tick rule is correctly applied.

        Tests cases where prev. trade price is higher, lower, equal or missing.

        Args:
            x_train (pd.DataFrame): training set
            subset (str): subset e. g., 'ex'
        """
        x_test = pd.DataFrame(
            [[1, 2], [2, 1], [1, 1], [1, np.nan]],
            columns=["trade_price", f"price_{subset}_lag"],
        )

        # first two by rule (see p. 28 Grauer et al.), remaining two by random chance.
        y_test = pd.Series([-1, 1, 1, -1])
        self._apply_rule(x_train, x_test, y_test, [("tick", subset)], 7)

    @pytest.mark.benchmark
    @pytest.mark.parametrize("subset", ["all", "ex"])
    def test_rev_tick_rule(self, x_train: pd.DataFrame, subset: str) -> None:
        """Test, if rev. tick rule is correctly applied.

        Tests cases where suc. trade price is higher, lower, equal or missing.

        Args:
            x_train (pd.DataFrame): training set
            subset (str): subset e. g., 'ex'
        """
        x_test = pd.DataFrame(
            [[1, 2], [2, 1], [1, 1], [1, np.nan]],
            columns=["trade_price", f"price_{subset}_lead"],
        )
        # first two by rule (see p. 28 Grauer et al.), remaining two by random chance.
        y_test = pd.Series([-1, 1, 1, -1])
        self._apply_rule(x_train, x_test, y_test, [("rev_tick", subset)], 7)

    @pytest.mark.benchmark
    @pytest.mark.parametrize("subset", ["best", "ex"])
    def test_quote_rule(self, x_train: pd.DataFrame, subset: str) -> None:
        """Test, if quote rule is correctly applied.

        Tests cases where prev. trade price is higher, lower, equal or missing.

        Args:
            x_train (pd.DataFrame): training set
            subset (str): subset e. g., 'ex'
        """
        # first two by rule (see p. 28 Grauer et al.), remaining four by random chance.
        x_test = pd.DataFrame(
            [
                [1, 1, 3],
                [3, 1, 3],
                [1, 1, 1],
                [3, 2, 4],
                [1, np.nan, 1],
                [3, np.nan, np.nan],
            ],
            columns=["trade_price", f"bid_{subset}", f"ask_{subset}"],
        )
        y_test = pd.Series([-1, 1, 1, -1, -1, 1])
        self._apply_rule(x_train, x_test, y_test, [("quote", subset)], 45)

    @pytest.mark.benchmark
    @pytest.mark.parametrize("subset", ["best", "ex"])
    def test_lr(self, x_train: pd.DataFrame, subset: str) -> None:
        """Test, if the lr algorithm is correctly applied.

        Tests cases where both quote rule and tick rule all are used.

        Args:
            x_train (pd.DataFrame): training set
            subset (str): subset e. g., 'ex'
        """
        # first two by quote rule, remaining two by tick rule.
        x_test = pd.DataFrame(
            [[1, 1, 3, 0], [3, 1, 3, 0], [1, 1, 1, 0], [3, 2, 4, 4]],
            columns=[
                "trade_price",
                f"bid_{subset}",
                f"ask_{subset}",
                f"price_{subset}_lag",
            ],
        )
        y_test = pd.Series([-1, 1, 1, -1])
        self._apply_rule(x_train, x_test, y_test, [("lr", subset)], 7)

    @pytest.mark.benchmark
    @pytest.mark.parametrize("subset", ["best", "ex"])
    def test_rev_lr(self, x_train: pd.DataFrame, subset: str) -> None:
        """Test, if the rev. lr algorithm is correctly applied.

        Tests cases where both quote rule and tick rule all are used.

        Args:
            x_train (pd.DataFrame): training set
            subset (str): subset e. g., 'ex'
        """
        # first two by quote rule, two by tick rule, and two by random chance.
        x_test = pd.DataFrame(
            [
                [1, 1, 3, 0],
                [3, 1, 3, 0],
                [1, 1, 1, 0],
                [3, 2, 4, 4],
                [1, 1, np.nan, np.nan],
                [1, 1, np.nan, np.nan],
            ],
            columns=[
                "trade_price",
                f"bid_{subset}",
                f"ask_{subset}",
                f"price_{subset}_lead",
            ],
        )
        y_test = pd.Series([-1, 1, 1, -1, -1, 1])
        self._apply_rule(x_train, x_test, y_test, [("rev_lr", subset)], 42)

    @pytest.mark.benchmark
    @pytest.mark.parametrize("subset", ["best", "ex"])
    def test_emo(self, x_train: pd.DataFrame, subset: str) -> None:
        """Test, if the emo algorithm is correctly applied.

        Tests cases where both quote rule at bid or ask and tick rule all are used.

        Args:
            x_train (pd.DataFrame): training set
            subset (str): subset e.g., best
        """
        # first two by quote rule, two by tick rule, two by random chance.
        x_test = pd.DataFrame(
            [
                [1, 1, 3, 0],
                [3, 1, 3, 0],
                [1, 1, 1, 0],
                [3, 2, 4, 4],
                [1, 1, np.inf, np.nan],
                [1, 1, np.nan, np.nan],
            ],
            columns=[
                "trade_price",
                f"bid_{subset}",
                f"ask_{subset}",
                f"price_{subset}_lag",
            ],
        )
        y_test = pd.Series([-1, 1, 1, -1, -1, 1])
        self._apply_rule(x_train, x_test, y_test, [("emo", subset)], 42)

    @pytest.mark.benchmark
    @pytest.mark.parametrize("subset", ["best", "ex"])
    def test_rev_emo(self, x_train: pd.DataFrame, subset: str) -> None:
        """Test, if the rev. emo algorithm is correctly applied.

        Tests cases where both quote rule at bid or ask and rev. tick rule all are used.

        Args:
            x_train (pd.DataFrame): training set
            subset (str): subset e. g., 'ex'
        """
        # first two by quote rule, two by tick rule, two by random chance.
        x_test = pd.DataFrame(
            [
                [1, 1, 3, 0],
                [3, 1, 3, 0],
                [1, 1, 1, 0],
                [3, 2, 4, 4],
                [1, 1, np.inf, np.nan],
                [1, 1, np.nan, np.nan],
            ],
            columns=[
                "trade_price",
                f"bid_{subset}",
                f"ask_{subset}",
                f"price_{subset}_lead",
            ],
        )
        y_test = pd.Series([-1, 1, 1, -1, -1, 1])
        self._apply_rule(x_train, x_test, y_test, [("rev_emo", subset)], 42)

    @pytest.mark.benchmark
    @pytest.mark.parametrize("subset", ["best", "ex"])
    def test_clnv(self, x_train: pd.DataFrame, subset: str) -> None:
        """Test, if the clnv algorithm is correctly applied.

        Tests cases where both quote rule and  tick rule all are used.

        Args:
            x_train (pd.DataFrame): training set
            subset (str): subset e. g., 'ex'
        """
        # first two by quote rule, two by tick rule, two by random chance.
        x_test = pd.DataFrame(
            [
                [5, 3, 1, 0],  # tick rule
                [0, 3, 1, 1],  # tick rule
                [2.9, 3, 1, 1],  # quote rule
                [2.3, 3, 1, 3],  # tick rule
                [1.7, 3, 1, 0],  # tick rule
                [1.3, 3, 1, 1],  # quote rule
            ],
            columns=[
                "trade_price",
                f"ask_{subset}",
                f"bid_{subset}",
                f"price_{subset}_lag",
            ],
        )
        y_test = pd.Series([1, -1, 1, -1, 1, -1])
        self._apply_rule(x_train, x_test, y_test, [("clnv", subset)], 42)

    @pytest.mark.benchmark
    @pytest.mark.parametrize("subset", ["best", "ex"])
    def test_rev_clnv(self, x_train: pd.DataFrame, subset: str) -> None:
        """Test, if the rev. clnv algorithm is correctly applied.

        Tests cases where both quote rule and rev. tick rule all are used.

        Args:
            x_train (pd.DataFrame): training set
            subset (str): subset e. g., 'ex'
        """
        x_test = pd.DataFrame(
            [
                [5, 3, 1, 0],  # rev tick rule
                [0, 3, 1, 1],  # rev tick rule
                [2.9, 3, 1, 1],  # quote rule
                [2.3, 3, 1, 3],  # rev tick rule
                [1.7, 3, 1, 0],  # rev tick rule
                [1.3, 3, 1, 1],  # quote rule
            ],
            columns=[
                "trade_price",
                f"ask_{subset}",
                f"bid_{subset}",
                f"price_{subset}_lead",
            ],
        )
        y_test = pd.Series([1, -1, 1, -1, 1, -1])
        self._apply_rule(x_train, x_test, y_test, [("rev_clnv", subset)], 5)

    @pytest.mark.benchmark
    def test_trade_size(self, x_train: pd.DataFrame) -> None:
        """Test, if the trade size algorithm is correctly applied.

        Tests cases where relevant data is present or missing.

        Args:
            x_train (pd.DataFrame): training set
        """
        # first two by trade size, random, at bid size, random, random.
        x_test = pd.DataFrame(
            [
                [1, 1, 3],
                [3, 1, 3],
                [1, 1, 1],
                [3, np.nan, 3],
                [1, np.inf, 2],
                [1, np.inf, 2],
            ],
            columns=["trade_size", "ask_size_ex", "bid_size_ex"],
        )
        y_test = pd.Series([-1, 1, -1, 1, -1, 1])
        self._apply_rule(x_train, x_test, y_test, [("trade_size", "ex")], 42)

    @pytest.mark.benchmark
    def test_depth(self, x_train: pd.DataFrame) -> None:
        """Test, if the depth rule is correctly applied.

        Tests cases where relevant data is present or missing.

        Args:
            x_train (pd.DataFrame): training set
        """
        # first three by depth, all other random as mid is different from trade price.
        x_test = pd.DataFrame(
            [
                [2, 1, 2, 4, 3],
                [1, 2, 2, 4, 3],
                [2, 1, 4, 4, 4],
                [2, 1, 2, 4, 2],
                [2, 1, 2, 4, 2],
            ],
            columns=[
                "ask_size_ex",
                "bid_size_ex",
                "ask_ex",
                "bid_ex",
                "trade_price",
            ],
        )
        y_test = pd.Series([1, -1, 1, 1, -1])
        self._apply_rule(x_train, x_test, y_test, [("depth", "ex")], 5)

    def test_dict_input(self, x_rand: pd.DataFrame) -> None:
        """Test, if a dict of arrays gives the same results as a DataFrame."""
        layers = self._all_layers()
        x_dict = {col: x_rand[col].to_numpy() for col in x_rand.columns}
        y_frame = ClassicalClassifier(layers=layers).fit(x_rand).predict_proba(x_rand)
        y_dict = ClassicalClassifier(layers=layers).fit(x_dict).predict_proba(x_dict)
        assert_allclose(y_frame, y_dict)

    def test_dict_missing_columns(self) -> None:
        """Test, if an error is raised, if columns of `features` are missing in dict."""
        classifier = ClassicalClassifier(features=["trade_price", "price_ex_lag"])
        with pytest.raises(
            ValueError, match=r"Expected to find columns: \['price_ex_lag'\]"
        ):
            classifier.fit({"trade_price": np.array([1.0, 2.0])})

    def test_np_array_without_features(self) -> None:
        """Test, if column names are required, if a layer needs them.

        Without layers, no names are required.
        """
        x = np.array([[1.0, 2.0], [2.0, 1.0]])
        with pytest.raises(ValueError, match=r"Expected to find columns"):
            ClassicalClassifier(layers=[("tick", "ex")]).fit(x)
        assert ClassicalClassifier().fit(x).predict(x).shape == (2,)

    def test_int_column_names(self) -> None:
        """Test, if DataFrames with non-string column names are supported."""
        x = pd.DataFrame(np.ones((3, 2)))
        assert ClassicalClassifier().fit(x).predict(x).shape == (3,)

    @pytest.mark.parametrize("strategy", ["random", "const"])
    def test_predict_dtype(self, x_rand: pd.DataFrame, strategy: str) -> None:
        """Test, if predictions are in dtype of `classes_` and probabilities in dtype of X."""
        x = x_rand.astype(np.float32)
        clf = ClassicalClassifier(layers=self._all_layers(), strategy=strategy).fit(x)  # ty: ignore[invalid-argument-type]
        assert clf.predict(x).dtype == clf.classes_.dtype
        assert clf.predict_proba(x).dtype == np.float32

    @pytest.mark.parametrize(("namespace", "device_name", "dtype"), ARRAY_API_PARAMS)
    @pytest.mark.parametrize("strategy", ["random", "const"])
    def test_array_api(
        self,
        x_rand: pd.DataFrame,
        namespace: str,
        device_name: str,
        dtype: str,
        strategy: str,
    ) -> None:
        """Test, if Array API inputs give the same results as numpy.

        Results must be in the namespace and on the device of the input.
        """
        xp, device = _get_namespace(namespace, device_name)
        x = x_rand.astype(dtype)
        x_xp = {col: xp.asarray(x[col].to_numpy(), device=device) for col in x.columns}
        layers = self._all_layers()

        clf = ClassicalClassifier(layers=layers, strategy=strategy).fit(x)  # ty: ignore[invalid-argument-type]
        with config_context(array_api_dispatch=True):
            clf_xp = ClassicalClassifier(layers=layers, strategy=strategy).fit(x_xp)  # ty: ignore[invalid-argument-type]
            pred_xp = clf_xp.predict(x_xp)
            proba_xp = clf_xp.predict_proba(x_xp)

        reference = x_xp["trade_price"]
        for result in (clf_xp.classes_, pred_xp, proba_xp):
            assert type(result) is type(reference)
            assert result.device == reference.device
        assert_allclose(_to_numpy(clf_xp.classes_), clf.classes_)
        assert_allclose(_to_numpy(pred_xp), clf.predict(x))
        assert_allclose(_to_numpy(proba_xp), clf.predict_proba(x))

    def test_array_api_namespace_mismatch(self, x_rand: pd.DataFrame) -> None:
        """Test, if an error is raised, if namespaces of fit and predict differ."""
        xp, device = _get_namespace("array_api_strict", "CPU_DEVICE")
        x_xp = {
            col: xp.asarray(x_rand[col].to_numpy(), device=device)
            for col in x_rand.columns
        }
        with config_context(array_api_dispatch=True):
            clf = ClassicalClassifier(layers=[("quote", "ex")]).fit(x_xp)
            with pytest.raises(ValueError, match=r"must use the same namespace"):
                clf.predict(x_rand)
