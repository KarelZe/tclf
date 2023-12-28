"""Tests for the classical classifier."""

from typing import Callable

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import check_is_fitted

from tclf.classical_classifier import ALLOWED_FUNC_LITERALS, ClassicalClassifier


class TestClassicalClassifier:
    """Perform automated tests for ClassicalClassifier.

    Args:
        unittest (_type_): unittest module
    """

    @pytest.fixture()
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

    @pytest.fixture()
    def x_test(self) -> pd.DataFrame:
        """Test set fixture.

        Returns:
            pd.DataFrame: test set
        """
        return pd.DataFrame(
            [[1, 2], [3, 4], [1, 2], [3, 4]], columns=["ask_best", "bid_best"]
        )

    @pytest.fixture()
    def y_test(self) -> pd.Series:
        """Test target fixture.

        Returns:
            pd.Series: test target
        """
        return pd.Series([1, -1, 1, -1])

    @pytest.fixture()
    def clf(self, x_train: pd.DataFrame) -> ClassicalClassifier:
        """Classifier fixture with random classification.

        Args:
            x_train (pd.DataFrame): train set

        Returns:
            ClassicalClassifier: fitted clf
        """
        return ClassicalClassifier().fit(x_train[["ask_best", "bid_best"]])

    @parametrize_with_checks([ClassicalClassifier()])
    def test_sklearn_compatibility(
        self, estimator: BaseEstimator, check: Callable
    ) -> None:
        """Test, if classifier is compatible with sklearn."""
        check(estimator)

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
        """Test, if strategy 'const' returns correct proabilities.

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
            layers=[("foo", "all")],  # type: ignore [list-item]
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
        y_test: pd.DataFrame,
        layers: list[tuple[ALLOWED_FUNC_LITERALS, str]],
        random_state: int = 7,
    ) -> None:
        """Apply rule-based classification.

        Args:
            x_train (pd.DataFrame): training features
            x_test (pd.DataFrame): test features
            y_test (pd.DataFrame): true labels
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

    def test_trade_size(self, x_train: pd.DataFrame) -> None:
        """Test, if the trade size algorithm is correctly applied.

        Tests cases where relevant data is present or missing.
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

    def test_depth(self, x_train: pd.DataFrame) -> None:
        """Test, if the depth rule is correctly applied.

        Tests cases where relevant data is present or missing.
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
