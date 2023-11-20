"""Tests for Neural networks.

See:
https://thenerdstation.medium.com/how-to-unit-test-machine-learning-code-57cf6fd81765
http://karpathy.github.io/2019/04/25/recipe/
https://krokotsch.eu/posts/deep-learning-unit-tests/
"""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import check_estimator


class ClassifierMixin:
    """Perform automated tests for Classifiers.

    Args:
    ----
        unittest (_type_): unittest module
    """

    clf: BaseEstimator
    x_test: pd.DataFrame
    y_test: pd.Series

    def test_sklearn_compatibility(self) -> None:
        """Test, if classifier is compatible with sklearn."""
        check_estimator(self.clf)

    def test_shapes(self) -> None:
        """Test, if shapes of the classifier equal the targets.

        Shapes are usually [no. of samples, 1].
        """
        y_pred = self.clf.predict(self.x_test)

        assert self.y_test.shape == y_pred.shape

    def test_proba(self) -> None:
        """Test, if probabilities are in [0, 1]."""
        y_pred = self.clf.predict_proba(self.x_test)
        assert (y_pred >= 0).all()
        assert (y_pred <= 1).all()

    def test_score(self) -> None:
        """Test, if score is correctly calculated..

        For a random classification i. e., `layers=[("nan", "ex")]`, the score
        should be around 0.5.
        """
        accuracy = self.clf.score(self.x_test, self.y_test)
        assert 0.0 <= accuracy <= 1.0
