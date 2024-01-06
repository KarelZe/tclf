# Trade classification with python 🐍

[![GitHubActions](https://github.com/karelze/tclf//actions/workflows/tests.yaml/badge.svg)](https://github.com/KarelZe/tclf/actions)
[![codecov](https://codecov.io/gh/KarelZe/tclf/branch/main/graph/badge.svg?token=CBM1RXGI86)](https://codecov.io/gh/KarelZe/tclf/tree/main/graph)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=KarelZe_tclf&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=KarelZe_tclf)


**Documentation ✒️:** [https://karelze.github.io/tclf/](https://karelze.github.io/tclf/)

**Source Code 🐍:** [https://github.com/KarelZe/tclf](https://github.com/KarelZe/tclf)


`tclf` is a [`scikit-learn`](https://scikit-learn.org/stable/)-compatible implementation of trade classification algorithms to classify financial markets transactions into buyer- and seller-initiated trades.

The key features are:

* **Easy**: Easy to use and learn.
* **Sklearn-compatible**: Compatible to the sklearn API. Use sklearn metrics and visualizations.
* **Feature complete**: Wide range of supported algorithms. Use the algorithms individually or stack them like LEGO blocks.

## Installation
```console
python -m pip install tclf
```

## Supported Algorithms

- (Rev.) CLNV rule[^1]
- (Rev.) EMO rule[^2]
- (Rev.) LR algorithm[^6]
- (Rev.) Tick test[^5]
- Depth rule[^3]
- Quote rule[^4]
- Tradesize rule[^3]

## Minimal Example

Let's start simple: classify all trades by the quote rule and all other trades, which cannot be classified by the quote rule, randomly.

Create a `main.py` with:
```python title="main.py"
import numpy as np
import pandas as pd

from tclf.classical_classifier import ClassicalClassifier

X = pd.DataFrame(
    [
        [1.5, 1, 3],
        [2.5, 1, 3],
        [1.5, 3, 1],
        [2.5, 3, 1],
        [1, np.nan, 1],
        [3, np.nan, np.nan],
    ],
    columns=["trade_price", "bid_ex", "ask_ex"],
)

clf = ClassicalClassifier(layers=[("quote", "ex")], strategy="random")
clf.fit(X)
probs = clf.predict_proba(X)
```
Run your script with
```console
$ python main.py
```
In this example, input data is available as a pd.DataFrame with columns conforming to our [naming conventions](https://karelze.github.io/tclf/naming_conventions/).

The parameter `layers=[("quote", "ex")]` sets the quote rule at the exchange level and `strategy="random"` specifies the fallback strategy for unclassified trades.

## Advanced Example
Often it is desirable to classify both on exchange level data and nbbo data. Also, data might only be available as a numpy array. So let's extend the previous example by classifying using the quote rule at exchange level, then at nbbo and all other trades randomly.

```python title="main.py" hl_lines="6  16 17 20"
import numpy as np
from sklearn.metrics import accuracy_score

from tclf.classical_classifier import ClassicalClassifier

X = np.array(
    [
        [1.5, 1, 3, 2, 2.5],
        [2.5, 1, 3, 1, 3],
        [1.5, 3, 1, 1, 3],
        [2.5, 3, 1, 1, 3],
        [1, np.nan, 1, 1, 3],
        [3, np.nan, np.nan, 1, 3],
    ]
)
y_true = np.array([-1, 1, 1, -1, -1, 1])
features = ["trade_price", "bid_ex", "ask_ex", "bid_best", "ask_best"]

clf = ClassicalClassifier(
    layers=[("quote", "ex"), ("quote", "best")], strategy="random", features=features
)
clf.fit(X)
acc = accuracy_score(y_true, clf.predict(X))
```
In this example, input data is available as np.arrays with both exchange (`"ex"`) and nbbo data (`"best"`). We set the layers parameter to `layers=[("quote", "ex"), ("quote", "best")]` to classify trades first on subset `"ex"` and remaining trades on subset `"best"`. Additionally, we have to set `ClassicalClassifier(..., features=features)` to pass column information to the classifier.

Like before, column/feature names must follow our [naming conventions](https://karelze.github.io/tclf/naming_conventions/). For more practical examples, see our [examples section](https://karelze.github.io/tclf/option_trade_classification).

## Citation

```latex
@software{bilz_tclf_2023,
    author = {Bilz, Markus},
    license = {BSD 3},
    month = jan,
    title = {{tclf} -- trade classification with python},
    url = {https://github.com/KarelZe/tclf},
    version = {0.0.4},
    year = {2024}
}
```

## Footnotes
  [^1]: <div class="csl-entry">Chakrabarty, B., Li, B., Nguyen, V., &amp; Van Ness, R. A. (2007). Trade classification algorithms for electronic communications network trades. <i>Journal of Banking &amp; Finance</i>, <i>31</i>(12), 3806–3821. <a href="https://doi.org/10.1016/j.jbankfin.2007.03.003">https://doi.org/10.1016/j.jbankfin.2007.03.003</a></div>
  <span class="Z3988" title="url_ver=Z39.88-2004&amp;ctx_ver=Z39.88-2004&amp;rfr_id=info%3Asid%2Fzotero.org%3A2&amp;rft_id=info%3Adoi%2F10.1016%2Fj.jbankfin.2007.03.003&amp;rft_val_fmt=info%3Aofi%2Ffmt%3Akev%3Amtx%3Ajournal&amp;rft.genre=article&amp;rft.atitle=Trade%20classification%20algorithms%20for%20electronic%20communications%20network%20trades&amp;rft.jtitle=Journal%20of%20Banking%20%26%20Finance&amp;rft.volume=31&amp;rft.issue=12&amp;rft.aufirst=Bidisha&amp;rft.aulast=Chakrabarty&amp;rft.au=Bidisha%20Chakrabarty&amp;rft.au=Bingguang%20Li&amp;rft.au=Vanthuan%20Nguyen&amp;rft.au=Robert%20A.%20Van%20Ness&amp;rft.date=2007&amp;rft.pages=3806%E2%80%933821&amp;rft.spage=3806&amp;rft.epage=3821"></span>
  [^2]: <div class="csl-entry">Ellis, K., Michaely, R., &amp; O’Hara, M. (2000). The accuracy of trade classification rules: Evidence from nasdaq. <i>The Journal of Financial and Quantitative Analysis</i>, <i>35</i>(4), 529–551. <a href="https://doi.org/10.2307/2676254">https://doi.org/10.2307/2676254</a></div>
  <span class="Z3988" title="url_ver=Z39.88-2004&amp;ctx_ver=Z39.88-2004&amp;rfr_id=info%3Asid%2Fzotero.org%3A2&amp;rft_id=info%3Adoi%2F10.2307%2F2676254&amp;rft_val_fmt=info%3Aofi%2Ffmt%3Akev%3Amtx%3Ajournal&amp;rft.genre=article&amp;rft.atitle=The%20accuracy%20of%20trade%20classification%20rules%3A%20evidence%20from%20nasdaq&amp;rft.jtitle=The%20Journal%20of%20Financial%20and%20Quantitative%20Analysis&amp;rft.volume=35&amp;rft.issue=4&amp;rft.aufirst=Katrina&amp;rft.aulast=Ellis&amp;rft.au=Katrina%20Ellis&amp;rft.au=Roni%20Michaely&amp;rft.au=Maureen%20O'Hara&amp;rft.date=2000&amp;rft.pages=529%E2%80%93551&amp;rft.spage=529&amp;rft.epage=551"></span>
  [^3]: <div class="csl-entry">Grauer, C., Schuster, P., &amp; Uhrig-Homburg, M. (2023). <i>Option trade classification</i>. <a href="https://doi.org/10.2139/ssrn.4098475">https://doi.org/10.2139/ssrn.4098475</a></div>
  <span class="Z3988" title="url_ver=Z39.88-2004&amp;ctx_ver=Z39.88-2004&amp;rfr_id=info%3Asid%2Fzotero.org%3A2&amp;rft_val_fmt=info%3Aofi%2Ffmt%3Akev%3Amtx%3Adc&amp;rft.type=document&amp;rft.title=Option%20trade%20classification&amp;rft.aufirst=Caroline&amp;rft.aulast=Grauer&amp;rft.au=Caroline%20Grauer&amp;rft.au=Philipp%20Schuster&amp;rft.au=Marliese%20Uhrig-Homburg&amp;rft.date=2023"></span>
  [^4]: <div class="csl-entry">Harris, L. (1989). A day-end transaction price anomaly. <i>The Journal of Financial and Quantitative Analysis</i>, <i>24</i>(1), 29. <a href="https://doi.org/10.2307/2330746">https://doi.org/10.2307/2330746</a></div>
  <span class="Z3988" title="url_ver=Z39.88-2004&amp;ctx_ver=Z39.88-2004&amp;rfr_id=info%3Asid%2Fzotero.org%3A2&amp;rft_id=info%3Adoi%2F10.2307%2F2330746&amp;rft_val_fmt=info%3Aofi%2Ffmt%3Akev%3Amtx%3Ajournal&amp;rft.genre=article&amp;rft.atitle=A%20day-end%20transaction%20price%20anomaly&amp;rft.jtitle=The%20Journal%20of%20Financial%20and%20Quantitative%20Analysis&amp;rft.volume=24&amp;rft.issue=1&amp;rft.aufirst=Lawrence&amp;rft.aulast=Harris&amp;rft.au=Lawrence%20Harris&amp;rft.date=1989&amp;rft.pages=29"></span>
  [^5]: <div class="csl-entry">Hasbrouck, J. (2009). Trading costs and returns for U.s. Equities: Estimating effective costs from daily data. <i>The Journal of Finance</i>, <i>64</i>(3), 1445–1477. <a href="https://doi.org/10.1111/j.1540-6261.2009.01469.x">https://doi.org/10.1111/j.1540-6261.2009.01469.x</a></div><span class="Z3988" title="url_ver=Z39.88-2004&amp;ctx_ver=Z39.88-2004&amp;rfr_id=info%3Asid%2Fzotero.org%3A2&amp;rft_id=info%3Adoi%2F10.1111%2Fj.1540-6261.2009.01469.x&amp;rft_val_fmt=info%3Aofi%2Ffmt%3Akev%3Amtx%3Ajournal&amp;rft.genre=article&amp;rft.atitle=Trading%20costs%20and%20returns%20for%20U.s.%20Equities%3A%20estimating%20effective%20costs%20from%20daily%20data&amp;rft.jtitle=The%20Journal%20of%20Finance&amp;rft.volume=64&amp;rft.issue=3&amp;rft.aufirst=Joel&amp;rft.aulast=Hasbrouck&amp;rft.au=Joel%20Hasbrouck&amp;rft.date=2009&amp;rft.pages=1445%E2%80%931477&amp;rft.spage=1445&amp;rft.epage=1477"></span>
  [^6]: <div class="csl-entry">Lee, C., &amp; Ready, M. J. (1991). Inferring trade direction from intraday data. <i>The Journal of Finance</i>, <i>46</i>(2), 733–746. <a href="https://doi.org/10.1111/j.1540-6261.1991.tb02683.x">https://doi.org/10.1111/j.1540-6261.1991.tb02683.x</a></div>
  <span class="Z3988" title="url_ver=Z39.88-2004&amp;ctx_ver=Z39.88-2004&amp;rfr_id=info%3Asid%2Fzotero.org%3A2&amp;rft_id=info%3Adoi%2F10.1111%2Fj.1540-6261.1991.tb02683.x&amp;rft_val_fmt=info%3Aofi%2Ffmt%3Akev%3Amtx%3Ajournal&amp;rft.genre=article&amp;rft.atitle=Inferring%20trade%20direction%20from%20intraday%20data&amp;rft.jtitle=The%20Journal%20of%20Finance&amp;rft.volume=46&amp;rft.issue=2&amp;rft.aufirst=Charles&amp;rft.aulast=Lee&amp;rft.au=Charles%20Lee&amp;rft.au=Mark%20J.%20Ready&amp;rft.date=1991&amp;rft.pages=733%E2%80%93746&amp;rft.spage=733&amp;rft.epage=746"></span>
