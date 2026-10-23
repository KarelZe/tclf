# Array API support

`tclf` supports scikit-learn's [Array API dispatch](https://scikit-learn.org/stable/modules/array_api.html) (experimental). It classifies trades directly on [Array API](https://data-apis.org/array-api/latest/) compatible arrays, such as PyTorch tensors or CuPy arrays, on CPU or GPU.

## Enabling Array API support

Set the environment variable `SCIPY_ARRAY_API=1` *before* importing scipy or scikit-learn, and enable dispatch with `array_api_dispatch=True`:

```console
export SCIPY_ARRAY_API=1
```

## Passing column names

The Array API standard has no notion of column names, but `tclf` needs them to map columns to rules (see [naming conventions](naming_conventions.md)). Pass Array API inputs as either:

- a `dict` of 1-D arrays keyed by column name, or
- a 2-D array together with `ClassicalClassifier(features=[...])`.

```python title="main.py"
import torch
from sklearn import config_context

from tclf.classical_classifier import ClassicalClassifier

X = {
    "trade_price": torch.tensor([1.5, 2.5, 1.5, 2.5], device="cuda"),
    "bid_ex": torch.tensor([1.0, 1.0, 3.0, 3.0], device="cuda"),
    "ask_ex": torch.tensor([3.0, 3.0, 1.0, 1.0], device="cuda"),
}

with config_context(array_api_dispatch=True):
    clf = ClassicalClassifier(layers=[("quote", "ex")], strategy="random")
    clf.fit(X)
    probs = clf.predict_proba(X)  # torch.Tensor on cuda
```

## Semantics

- **Everything follows `X`.** Fitted attributes (e.g., `classes_`) and outputs of `predict` and `predict_proba` use the namespace and device of `X`. Inputs passed to `predict` must use the same namespace and device as those passed to `fit`; otherwise a `ValueError` is raised.
- **DataFrames are converted to NumPy.** As in scikit-learn, DataFrames (pandas, Polars, cuDF, …) are always processed with NumPy on the CPU.
- **dtypes.** Floating point inputs keep their precision, e.g. `float32` inputs on devices without `float64` support (such as Apple MPS) stay `float32`. `predict` returns labels in the dtype of `classes_`, `predict_proba` returns probabilities in the floating point dtype of `X`.
- **Reproducibility.** With `strategy="random"`, unclassified trades are filled with random draws from NumPy's random number generator. Predictions are therefore identical across namespaces and devices for the same `random_state`.
