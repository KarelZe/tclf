---
comments: true
---

# A Primer on Trade Classification Rules

The goal of trade classification algorithms is to identify the *initiator of a trade*. While definitions for the trade initiator differ in literature [cp. @leeInferringInvestorBehavior2000, pp. 94--97; @odders-whiteOccurrenceConsequencesInaccurate2000, p. 262] the trade initiator is binary and either the buyer or the seller.

As the trade initiator is frequently absent in datasets, it must be inferred using trade classification algorithms or other approaches. This article introduces basic rules for trade classification.

## Notation
We denote the predicted class by $y \in \mathcal{Y}$ with $\mathcal{Y}=\{-1,1\}$, whereby $y=-1$ is indicating a seller-initiated and $y=1$ a buyer-initiated trade. We denote the sequence of trade prices of the $i$-th security by $(P_{i,t})_{t=1}^{T}$ and the corresponding ask at $t$ by $A_{i,t}$ and bid by $B_{i,t}$. The midpoint of the bid-ask spread is denoted by $M_{i,t} = \tfrac{1}{2}(B_{i,t} + A_{i,t})$. Moreover, we denote the quoted size at the ask with $\tilde{A}_{i,t}$, $\tilde{B}_{i,t}$ of the bid, and $P_{i,t}$ the trade price at $t$ of the $i$-th security.

For simplicity we assume an ideal data regime, where quote data is complete and spreads are positive.🥵

## Basic Rules

This section presents basic classification rules, that may be used for trade classification independently or integrated into a [hybrid algorithm](#hybrid-rules).

### Quote Rule

The quote rule classifies a trade by comparing the trade price against the corresponding quotes at the time of the trade. If the trade price is above the midpoint of the bid-ask spread, $M_{i,t}$, the trade is classified as a buy and if it is below the midpoint, as a sell [@harrisDayEndTransactionPrice1989, p. 41].

Thus, the classification rule on $\mathcal{A} = \left\{(i, t) \in \mathbb{N}^2: P_{i,t} \neq M_{i,t}\right\}$ is given by:

$$
    \operatorname{quote}\colon \mathcal{A} \to \mathcal{Y},\quad
    \operatorname{quote}(i, t)=
    \begin{cases}
        1,  & \mathrm{if}\ P_{i, t}>M_{i, t}  \\
        -1, & \mathrm{if}\ P_{i, t}<M_{i, t}. \\
    \end{cases}
$$

The quote rule cannot classify trades at the midpoint of the quoted spread. A solution to overcome this limitation is to couple the quote rule with other approaches, as done in [hybrid rules](#hybrid-rules).

**Code**
```python
from tclf.classical_classifier import ClassicalClassifier
clf = ClassicalClassifier(layers=[("quote", "subset")])
clf.fit(X)
```

### Tick Test

A common alternative to the quote rule is the tick test. Based on the rationale that buys increase trade prices and sells lower them, the tick test classifies trades by the change in trade price. It was first applied in [@holthausenEffectLargeBlock1987, p. 244; @hasbrouckTradesQuotesInventories1988, p.240].

The tick test is defined as:

$$
    \operatorname{tick}\colon \mathbb{N}^2 \to \mathcal{Y},\quad
    \operatorname{tick}(i, t)=
    \begin{cases}
        1,                           & \mathrm{if}\ P_{i, t}>P_{i, t-1} \\
        -1,                          & \mathrm{if}\ P_{i, t} < P_{i, t-1}  \\
        Y\sim\mathrm{Uniform}(\mathcal{Y}), & \mathrm{if}\ t=1 \\
        \operatorname{tick}(i, t-1), & \mathrm{else}.
    \end{cases}
$$

Considering the cases the trade price is higher than the previous price (uptick) the trade is classified as a buy. Reversely, if it is below the previous price (downtick), the trade is classified as a sell. If the price change is zero (zero tick), the signing uses the last price different from the current price [@leeInferringTradeDirection1991,p. 735]. To end recursion at $t=1$, we sign the trades randomly as buyer- or seller-initiated to simplify notation 🤓.

The tick rule can sign all trades as long as a last differing trade price exists, but the overall precision can be impacted by infrequent trading.

**Code**
```python
from tclf.classical_classifier import ClassicalClassifier
clf = ClassicalClassifier(layers=[("tick", "subset")], strategy="random")
clf.fit(X)
```

### Reverse Tick Test

The reverse tick test is a variant of the tick test proposed in [@hasbrouckTradesQuotesInventories1988, p.241]. It is similar to the tick rule but classifies based on the next, distinguishable trade price.

$$
    \operatorname{rtick} \colon \mathbb{N}^2 \to \mathcal{Y},\quad
    \operatorname{rtick}(i, t)=
    \begin{cases}
        1,                            & \mathrm{if}\ P_{i, t} > P_{i, t+1} \\
        -1,                           & \mathrm{if}\ P_{i, t} < P_{i, t+1}            \\
        Y\sim\mathrm{Uniform}(\mathcal{Y}), & \mathrm{if}\ t+1=T \\
        \operatorname{rtick}(i, t+1), & \mathrm{else}
    \end{cases}
$$

As denoted in the equation, the trade is classified as seller-initiated, if the next trade is on an uptick or a zero uptick, and classified as buyer-initiated for trades at a downtick or a zero downtick [@leeInferringTradeDirection1991, pp. 735--736].

**Code**
```python
from tclf.classical_classifier import ClassicalClassifier
clf = ClassicalClassifier(layers=[("rev_tick", "subset")], strategy="random")
clf.fit(X)
```


### Depth Rule

The depth rule gauges the trade initiator from the quoted size at the best bid and ask. Based on the observation that an exceeding bid or ask size relates to higher liquidity at one trade side, trades are classified as a buy (sell) for a larger ask (bid) size [@grauerOptionTradeClassification2022, pp. 14--15].
We set the domain as $\mathcal{A} = \left\{(i, t) \in \mathbb{N}^2: P_{i,t} = M_{i,t} \land \tilde{A}_{i,t} \neq \tilde{B}_{i,t} \right\}$. The depth rule is now calculated as:

$$
    \operatorname{depth} \colon \mathcal{A} \to \mathcal{Y},\quad
    \operatorname{depth}(i, t)=
    \begin{cases}
        1,  & \mathrm{if}\ \tilde{A}_{i,t} > \tilde{B}_{i,t}. \\
        -1, & \mathrm{if}\ \tilde{A}_{i,t} < \tilde{B}_{i,t}\\
    \end{cases}
$$

The depth rule classifies midspread trades only, if the ask size is different from the bid size, as the ratio between the ask and bid size is the sole criterion for inferring the trade's initiator. Due to these restrictive conditions in $\mathcal{A}$, the depth rule can sign only a fraction of all trades and must be best followed by other rules.

**Code**
```python
from tclf.classical_classifier import ClassicalClassifier
clf = ClassicalClassifier(layers=[("depth", "subset")])
clf.fit(X)
```

### Trade Size Rule

The trade size rule classifies based on a match between the size of the trade $\tilde{P}_{i, t}$ and the quoted bid and ask sizes. The rationale is, that the market maker tries to fill the limit order of a customer, which results in the trade being executed at the contemporaneous bid or ask, with a trade size equaling the quoted size [@grauerOptionTradeClassification2022]. The trade size rule is defined on $\mathcal{A} = \left\{(i, t) \in \mathbb{N}^2: \tilde{P}_{i,t} = \tilde{A}_{i,t} \neq \tilde{B}_{i,t} \lor \tilde{P}_{i,t} \neq\tilde{A}_{i,t} = \tilde{B}_{i,t} \right\}$ as:

$$
    \operatorname{tsize} \colon \mathcal{A} \to \mathcal{Y},\quad
    \operatorname{tsize}(i, t)=
    \begin{cases}
        1,  & \mathrm{if}\ \tilde{P}_{i, t} = \tilde{B}_{i, t} \neq \tilde{A}_{i, t}  \\
        -1, & \mathrm{if}\ \tilde{P}_{i, t} = \tilde{A}_{i, t} \neq \tilde{B}_{i, t}. \\
    \end{cases}
$$

When both the size of the ask and bid correspond with the trade size or the trade size does not match the quoted sizes, the result is ambiguous.

**Code**
```python
from tclf.classical_classifier import ClassicalClassifier
clf = ClassicalClassifier(layers=[("trade_size", "subset")])
clf.fit(X)
```

## Hybrid Rules

The basic trade classification rules from [basic rules](#basic-rules) can be combined into a hybrid algorithm to enforce universal applicability to all trades and improve the classification performance.

Popular variants include the [LR algorithm](#lee-and-ready-algorithm), the [EMO rule](#ellis-michaely-ohara-rule), and the [CLNV method](#chakrabarty-li-nguyen-van-ness-method). All three algorithms utilize the quote and tick rule to a varying extent. Basic rules are selected based on the proximity of the trade price to the quotes.

As put forth by Grauer et al. [-@grauerOptionTradeClassification2022], basic or hybrid rules can be combined through stacking. This approach generalizes the aforementioned algorithms, as the applied rule is no longer dependent on the proximity to the quotes, but rather on the classifiability of the trade with the primary rules given by the domains and their ordering.

### Lee and Ready Algorithm

The LR algorithm [@leeInferringTradeDirection1991, p. 745] combines the (reverse) tick test and quote rule into a single rule, which is derived from two observations. First, Lee and Ready [-@leeInferringTradeDirection1991, pp. 735-745] observe a higher precision of the quote rule over the tick rule, which makes it their preferred choice. Second, by the means of a simple model, the authors demonstrate that the tick test can correctly classify on average 85.4 % of all midspread trades if the model's assumptions of constant quotes between trades and the arrival of the market and standing orders following a Poisson process are met. Outside the model's tight assumptions, the expected accuracy of the tick test can be unmet.

In combination, the algorithm primarily signs trades according to the quote rule. Trades at the midpoint of the spread, unclassifiable by the quote rule, are classified by the tick test. Overall:

$$
    \operatorname{lr} \colon \mathbb{N}^2 \to \mathcal{Y},\quad\operatorname{lr}(i,t)=
    \begin{cases}
        1,                         & \mathrm{if}\ P_{i, t} > M_{i, t} \\
        -1,                        & \mathrm{if}\ P_{i, t} < M_{i, t} \\
        \operatorname{tick}(i, t), & \mathrm{else}.
    \end{cases}
$$


**Code**
```python
from tclf.classical_classifier import ClassicalClassifier
clf = ClassicalClassifier(layers=[("lr", "subset")])
clf.fit(X)
```

### Ellis-Michaely-O'Hara Rule

Ellis et al. [-@ellisAccuracyTradeClassification2000, pp. 535--536] examine the performance of the previous algorithms for stocks traded at NASDAQ. By analyzing miss-classified trades with regard to the proximity of the trade to the quotes, they observe, that the quote rule and by extension, the [LR algorithm](#lee-and-ready-algorithm), perform particularly well at classifying trades executed at the bid and the ask price but trail the performance of the tick rule for trades inside or outside the spread [@ellisAccuracyTradeClassification2000, pp. 535--536]. The authors combine these observations into a single rule, known as the EMO algorithm.

The EMO algorithm extends the tick rule by classifying trades at the quotes using the quote rule, and all other trades with the tick test. Formally, the classification rule is given by:

$$
    \operatorname{emo} \colon \mathbb{N}^2 \to \mathcal{Y}, \quad
    \operatorname{emo}(i, t)=
    \begin{cases}
        1,                         & \mathrm{if}\ P_{i, t} = A_{i, t} \\
        -1,                        & \mathrm{if}\ P_{i, t} = B_{i, t} \\
        \operatorname{tick}(i, t), & \mathrm{else}.
    \end{cases}
$$

The EMO algorithm embeds both the quote and tick rule. As trades off the quotes are classified by the tick rule, the algorithm's overall success rate is dominated by the tick test assuming most trades are off-the-quotes.

**Code**
```python
from tclf.classical_classifier import ClassicalClassifier
clf = ClassicalClassifier(layers=[("emo", "subset")])
clf.fit(X)
```

### Chakrabarty-Li-Nguyen-Van-Ness Method

Like the previous two algorithms, the CLNV method [@chakrabartyTradeClassificationAlgorithms2007, pp. 3811-3812]  is a hybrid of the quote and tick rule and extends the EMO rule by a differentiated treatment of trades inside the quotes, which are notoriously hard to classify. The authors segment the bid-ask spread into deciles (ten equal-width bins) and classify trades around the midpoint (fourth to seventh decile) by the tick rule and trades close or outside the quotes are categorized by the tick rule.

$$
    \operatorname{clnv} \colon \mathbb{N}^2 \to \mathcal{Y}, \quad
    \operatorname{clnv}(i, t)=
    \begin{cases}
        1,                         & \mathrm{if}\ P_{i, t} \in \left(\frac{3}{10} B_{i,t} + \frac{7}{10} A_{i,t}, A_{i, t}\right] \\
        -1,                        & \mathrm{if}\ P_{i, t} \in \left[ B_{i,t}, \frac{7}{10} B_{i,t} + \frac{3}{10} A_{i,t}\right) \\
        \operatorname{tick}(i, t), & \mathrm{else}
    \end{cases}
$$

It is derived from a performance comparison of the tick rule ([EMO rule](#ellis-michaely-ohara-rule)) against the quote rule ([LR algorithm](#lee-and-ready-algorithm)) on stock data, whereby the accuracy was assessed separately for each decile.

**Code**
```python
from tclf.classical_classifier import ClassicalClassifier
clf = ClassicalClassifier(layers=[("clnv", "subset")])
clf.fit(X)
```

### Stacked Rule

The previous algorithms are static concerning the used base rules and their alignment. Combining arbitrary rules into a single algorithm requires a generic procedure. Grauer et al.[-@grauerOptionTradeClassification2022, p. 15] combine basic and hybrid rules through stacking. In this setting, the trade traverses a stack of pre-defined rules until a rule can classify the trade or the end of the stack is reached. The classification is now dependent on the employed rules but also on their relative ordering.

The most basic application is in the [LR algorithm](#lee-and-ready-algorithm), combining $\operatorname{quote}$ and $\operatorname{tick}$. For a more complex example consider the hybrid rule consisting of $\operatorname{tsize}_{\mathrm{ex}}$, $\operatorname{quote}_{\mathrm{nbbo}}$, $\operatorname{quote}_{\mathrm{ex}}$, $\operatorname{depth}_{\mathrm{nbbo}}$, $\operatorname{depth}_{\mathrm{ex}}$ and  $\operatorname{rtick}_{\mathrm{all}}$ popularized in Grauer et al. [-@grauerOptionTradeClassification2022, p. 15].

In practice, rules may be ordered greedily and new rules added if there are unclassified trades.

**Code**
```python
from tclf.classical_classifier import ClassicalClassifier

layers = [
    ("trade_size", "ex"),
    ("quote", "best"),
    ("quote", "ex"),
    ("depth", "best"),
    ("depth", "ex"),
    ("rev_tick", "all"),
]
clf = ClassicalClassifier(layers=layers, strategy="random")
clf.fit(X)
```
## Footnotes
\bibliography
