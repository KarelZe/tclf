# better DataFrame support

## task

- I'd like to provide support for a wide range of dataframe apis, so that tclf can be used with Polars, cuDF, pandas etc.
- I'd propose to use `narwhals` for it.
- Previously, I wanted to implement using the `__dataframe__` protocol (see here https://github.com/KarelZe/tclf/pull/103), which seems to be fallen out of favour.

## background
- The ecosystem is shifting away from the `__dataframe__` interchange protocol due to its complexity and brittleness. Notably:
- **Scikit-learn** and **skrub** are currently discussing or implementing Narwhals to simplify DataFrame input/output handling.
- **Polars** has de-emphasized the interchange protocol in favor of the **Arrow PyCapsule Interface**. As of version 1.23.0, Polars only uses the old protocol as a fallback, citing implementation inconsistencies and bugs in external libraries (like pandas < 2.2) that led to unreliable results.
- Narwhals provides a lightweight, transparent wrapper that allows `tclf` to treat various DataFrames as a unified object without adding heavy dependencies or sacrificing performance.
**references**
- https://github.com/scikit-learn/scikit-learn/issues/31049
- https://github.com/skrub-data/skrub/issues/1818

## Potential Implementation
```python
# https://github.com/scikit-learn/scikit-learn/pull/31127/changes#diff-9ae0e7df7811fcd30e802947898d26c1d83267255d7abf79b20a25233f86cabeR370
import narwhals.stable.v2 as nw

        if nw.dependencies.is_into_dataframe(X):
            X = nw.from_native(X)
            dtypes = X.schema.dtypes()
```

## tasks

- remove `pandas` and maybe even `numpy` as a runtime dependency
- update implementation in `ClassicalClassifier` to use `narwhals`
- add additional tests for `polars` and maybe `cuDF`.
- update documentation
- run benchmarks before and after test
