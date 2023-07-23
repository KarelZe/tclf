import pytest

from sklearn.utils.estimator_checks import check_estimator

from tclf import TemplateEstimator
from tclf import TemplateClassifier
from tclf import TemplateTransformer


@pytest.mark.parametrize(
    "estimator",
    [TemplateEstimator(), TemplateTransformer(), TemplateClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
