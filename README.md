![GitHubActions](https://github.com/karelze/tclf//actions/workflows/tests.yaml/badge.svg)
![Codecov](https://codecov.io/gh/karlze/tclf/branch/master/graph/badge.svg)

# tclf 💸

[`scikit-learn`](https://scikit-learn.org/stable/)-compatible implementation of popular trade classification algorithms to classify financial markets transactions into buyer- and seller-initiated trades.

## Algorithms

- Tick test
- Quote rule
- LR algorithm
- EMO rule
- CLNV rule
- Depth rule
- Tradesize rule

## Usage

Documentation is available [here](https://KarelZe.github.io/tclf/).

## References

<div class="csl-bib-body" style="line-height: 2; margin-left: 2em; text-indent:-2em;">
  <div class="csl-entry">Chakrabarty, B., Li, B., Nguyen, V., &amp; Van Ness, R. A. (2007). Trade classification algorithms for electronic communications network trades. <i>Journal of Banking &amp; Finance</i>, <i>31</i>(12), 3806–3821. <a href="https://doi.org/10.1016/j.jbankfin.2007.03.003">https://doi.org/10.1016/j.jbankfin.2007.03.003</a></div>
  <span class="Z3988" title="url_ver=Z39.88-2004&amp;ctx_ver=Z39.88-2004&amp;rfr_id=info%3Asid%2Fzotero.org%3A2&amp;rft_id=info%3Adoi%2F10.1016%2Fj.jbankfin.2007.03.003&amp;rft_val_fmt=info%3Aofi%2Ffmt%3Akev%3Amtx%3Ajournal&amp;rft.genre=article&amp;rft.atitle=Trade%20classification%20algorithms%20for%20electronic%20communications%20network%20trades&amp;rft.jtitle=Journal%20of%20Banking%20%26%20Finance&amp;rft.volume=31&amp;rft.issue=12&amp;rft.aufirst=Bidisha&amp;rft.aulast=Chakrabarty&amp;rft.au=Bidisha%20Chakrabarty&amp;rft.au=Bingguang%20Li&amp;rft.au=Vanthuan%20Nguyen&amp;rft.au=Robert%20A.%20Van%20Ness&amp;rft.date=2007&amp;rft.pages=3806%E2%80%933821&amp;rft.spage=3806&amp;rft.epage=3821"></span>
  <div class="csl-entry">Ellis, K., Michaely, R., &amp; O’Hara, M. (2000). The accuracy of trade classification rules: Evidence from nasdaq. <i>The Journal of Financial and Quantitative Analysis</i>, <i>35</i>(4), 529–551. <a href="https://doi.org/10.2307/2676254">https://doi.org/10.2307/2676254</a></div>
  <span class="Z3988" title="url_ver=Z39.88-2004&amp;ctx_ver=Z39.88-2004&amp;rfr_id=info%3Asid%2Fzotero.org%3A2&amp;rft_id=info%3Adoi%2F10.2307%2F2676254&amp;rft_val_fmt=info%3Aofi%2Ffmt%3Akev%3Amtx%3Ajournal&amp;rft.genre=article&amp;rft.atitle=The%20accuracy%20of%20trade%20classification%20rules%3A%20evidence%20from%20nasdaq&amp;rft.jtitle=The%20Journal%20of%20Financial%20and%20Quantitative%20Analysis&amp;rft.volume=35&amp;rft.issue=4&amp;rft.aufirst=Katrina&amp;rft.aulast=Ellis&amp;rft.au=Katrina%20Ellis&amp;rft.au=Roni%20Michaely&amp;rft.au=Maureen%20O'Hara&amp;rft.date=2000&amp;rft.pages=529%E2%80%93551&amp;rft.spage=529&amp;rft.epage=551"></span>
  <div class="csl-entry">Grauer, C., Schuster, P., &amp; Uhrig-Homburg, M. (2023). <i>Option trade classification</i>. <a href="https://doi.org/10.2139/ssrn.4098475">https://doi.org/10.2139/ssrn.4098475</a></div>
  <span class="Z3988" title="url_ver=Z39.88-2004&amp;ctx_ver=Z39.88-2004&amp;rfr_id=info%3Asid%2Fzotero.org%3A2&amp;rft_val_fmt=info%3Aofi%2Ffmt%3Akev%3Amtx%3Adc&amp;rft.type=document&amp;rft.title=Option%20trade%20classification&amp;rft.aufirst=Caroline&amp;rft.aulast=Grauer&amp;rft.au=Caroline%20Grauer&amp;rft.au=Philipp%20Schuster&amp;rft.au=Marliese%20Uhrig-Homburg&amp;rft.date=2023"></span>
  <div class="csl-entry">Harris, L. (1989). A day-end transaction price anomaly. <i>The Journal of Financial and Quantitative Analysis</i>, <i>24</i>(1), 29. <a href="https://doi.org/10.2307/2330746">https://doi.org/10.2307/2330746</a></div>
  <span class="Z3988" title="url_ver=Z39.88-2004&amp;ctx_ver=Z39.88-2004&amp;rfr_id=info%3Asid%2Fzotero.org%3A2&amp;rft_id=info%3Adoi%2F10.2307%2F2330746&amp;rft_val_fmt=info%3Aofi%2Ffmt%3Akev%3Amtx%3Ajournal&amp;rft.genre=article&amp;rft.atitle=A%20day-end%20transaction%20price%20anomaly&amp;rft.jtitle=The%20Journal%20of%20Financial%20and%20Quantitative%20Analysis&amp;rft.volume=24&amp;rft.issue=1&amp;rft.aufirst=Lawrence&amp;rft.aulast=Harris&amp;rft.au=Lawrence%20Harris&amp;rft.date=1989&amp;rft.pages=29"></span>
  <div class="csl-entry">Hasbrouck, J. (2009). Trading costs and returns for U.s. Equities: Estimating effective costs from daily data. <i>The Journal of Finance</i>, <i>64</i>(3), 1445–1477. <a href="https://doi.org/10.1111/j.1540-6261.2009.01469.x">https://doi.org/10.1111/j.1540-6261.2009.01469.x</a></div>
  <span class="Z3988" title="url_ver=Z39.88-2004&amp;ctx_ver=Z39.88-2004&amp;rfr_id=info%3Asid%2Fzotero.org%3A2&amp;rft_id=info%3Adoi%2F10.1111%2Fj.1540-6261.2009.01469.x&amp;rft_val_fmt=info%3Aofi%2Ffmt%3Akev%3Amtx%3Ajournal&amp;rft.genre=article&amp;rft.atitle=Trading%20costs%20and%20returns%20for%20U.s.%20Equities%3A%20estimating%20effective%20costs%20from%20daily%20data&amp;rft.jtitle=The%20Journal%20of%20Finance&amp;rft.volume=64&amp;rft.issue=3&amp;rft.aufirst=Joel&amp;rft.aulast=Hasbrouck&amp;rft.au=Joel%20Hasbrouck&amp;rft.date=2009&amp;rft.pages=1445%E2%80%931477&amp;rft.spage=1445&amp;rft.epage=1477"></span>
  <div class="csl-entry">Lee, C., &amp; Ready, M. J. (1991). Inferring trade direction from intraday data. <i>The Journal of Finance</i>, <i>46</i>(2), 733–746. <a href="https://doi.org/10.1111/j.1540-6261.1991.tb02683.x">https://doi.org/10.1111/j.1540-6261.1991.tb02683.x</a></div>
  <span class="Z3988" title="url_ver=Z39.88-2004&amp;ctx_ver=Z39.88-2004&amp;rfr_id=info%3Asid%2Fzotero.org%3A2&amp;rft_id=info%3Adoi%2F10.1111%2Fj.1540-6261.1991.tb02683.x&amp;rft_val_fmt=info%3Aofi%2Ffmt%3Akev%3Amtx%3Ajournal&amp;rft.genre=article&amp;rft.atitle=Inferring%20trade%20direction%20from%20intraday%20data&amp;rft.jtitle=The%20Journal%20of%20Finance&amp;rft.volume=46&amp;rft.issue=2&amp;rft.aufirst=Charles&amp;rft.aulast=Lee&amp;rft.au=Charles%20Lee&amp;rft.au=Mark%20J.%20Ready&amp;rft.date=1991&amp;rft.pages=733%E2%80%93746&amp;rft.spage=733&amp;rft.epage=746"></span>
</div>