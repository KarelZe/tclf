## Frequently Asked Questions

**How are `NaN` values handled in by `tclf`?**

We take care to treat `NaN` values correctly. If features relevant for classification like the trade price or quoted bid/ask prices are missing, no classification is performed and classification of the trade is deferred to the subsequent rule or fallback strategy.

Alternatively, you can provide imputed data. See [`sklearn.impute`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.impute) for details.
