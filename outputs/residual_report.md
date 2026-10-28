# Residual diagnostics

- Overall RMSE 7.3768 | MAE 3.0481 | bias -0.0201
- Heteroskedasticity: Spearman(|resid|, pred) = 0.271 (p=2.9e-134); BP-style LM p-value = 0
  (error grows with fare size: consider log-target / sample weighting)

## RMSE by predicted-fare decile
| pred_bin           |     rmse |   n |
|:-------------------|---------:|----:|
| (-116.446, 13.831] |  4.29624 | 800 |
| (13.831, 15.931]   |  2.86448 | 800 |
| (15.931, 17.65]    |  3.21391 | 800 |
| (17.65, 19.263]    |  2.75544 | 799 |
| (19.263, 21.318]   |  3.14423 | 800 |
| (21.318, 24.186]   |  3.57324 | 800 |
| (24.186, 28.584]   |  4.00574 | 799 |
| (28.584, 36.156]   |  5.58963 | 800 |
| (36.156, 65.927]   |  9.14512 | 800 |
| (65.927, 265.183]  | 18.5986  | 800 |

## Edge-case slices
|                      |    n |      mae |       bias |
|:---------------------|-----:|---------:|-----------:|
| long_trip(>p99 dist) |   80 | 31.2203  |  3.77943   |
| very_short(<0.3mi)   |  208 | 11.2551  |  0.454679  |
| slow(<3mph)          | 3488 |  2.61212 | -0.0744579 |
| refund_rows          |   78 |  6.805   | -1.31755   |
| top1%_fares          |   79 | 31.3595  | 14.9682    |

## Worst zone pairs (n>=5)
| n   | mae   | bias   |
|-----|-------|--------|