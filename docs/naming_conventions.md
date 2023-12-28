For `tclf` to work, we impose constraints on the column names. The following input is required by each rule. Data requirements are additive, if multiple rules are applied.




| Rule                        | Layer Name             | Columns                                                                                   |
|-----------------------------|------------------------|-------------------------------------------------------------------------------------------|
| No classification           | `("nan","sub")`        | None                                                                                      |
| Tick test                   | `("tick","sub")`       | `trade_price`, `price_{sub}_lag`                                                          |
| Reverse tick Test           | `("rev_tick","sub")`   | `trade_price`, `price_{sub}_lead`                                                         |
| Quote Rule                  | `("quote","sub")`      | `trade_price`, `ask_{sub}`, `bid_{sub}`                                                   |
| Lee-Ready Algorithm         | `("lr","sub")`         | `trade_price`, `price_{sub}_lag`, `ask_{sub}`, `bid_{sub}`                                |
| EMO Algorithm               | `("emo","sub")`        | `trade_price`, `price_{sub}_lag`, `ask_{sub}`, `bid_{sub}`                                |
| CLNV Rule                   | `("clnv","sub")`       | `trade_price`, `price_{sub}_lag`, `ask_{sub}`, `bid_{sub}`                                |
| Reverse Lee-Ready Algorithm | `("rev_lr","sub")`     | `trade_price`, `price_{sub}_lead`, `ask_{sub}`, `bid_{sub}`                               |
| Reverse EMO Algorithm       | `("rev_emo","sub")`    | `trade_price`, `price_{sub}_lead`, `ask_{sub}`, `bid_{sub}`                               |
| Reverse CLNV Rule           | `("rev_clnv","sub")`   | `trade_price`, `price_{sub}_lead`, `ask_{sub}`, `bid_{sub}`                               |
| Depth rule                  | `("depth","sub")`      | `trade_price`, `ask_{sub}`, `bid_{sub}`, `ask_size_{sub}`, `bid_size_{sub}` |
| Trade size rule             | `("trade_size","sub")` | `trade_size`, `ask_size_{sub}`, `bid_size_{sub}`                                          |
