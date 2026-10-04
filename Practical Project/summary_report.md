# Summary Report — Credit Card Customer Segmentation

## 1. Business problem and dataset
A large Indian private bank wants to stop offering the same card upgrades, rewards and limit revisions to everyone. The goal is to group existing cardholders into behavioural segments so Relationship Managers can offer the right product to the right customer. I used the Kaggle *Credit Card Dataset for Clustering* (`CC_GENERAL.csv`): 8,950 cardholders and 17 behavioural columns (balance, purchases, cash advance, credit limit, payments, tenure and so on) after dropping `CUST_ID`.

## 2. Feature engineering and preprocessing
Only 314 values were missing (`CREDIT_LIMIT` 1, `MINIMUM_PAYMENTS` 313), so I filled them with the median because the columns are heavily skewed. There were no duplicates. I added four behavioural features: `Monthly_Avg_Purchase`, `Monthly_Avg_Cash_Advance`, `Limit_Usage` and `Payment_to_Minpayment_Ratio`, with divide-by-zero guards. Money columns had extreme outliers, so I capped them at Q3 + 3×IQR (winsorising, no rows dropped), then applied `log1p` to fix the right skew (skew fell from about 1.7 to near 0), and finally used `StandardScaler`, because K-Means, Ward linkage and DBSCAN all depend on distance. The final table has 21 features.

## 3. Which algorithm performed best?
| Algorithm | Clusters | Silhouette | Davies-Bouldin | Calinski-Harabasz | Noise |
|---|---|---|---|---|---|
| K-Means (k=4) | 4 | 0.198 | 1.71 | 2,057 | 0% |
| Agglomerative (Ward) | 4 | 0.142 | 2.00 | 1,595 | 0% |
| DBSCAN (eps 2.5, min_samples 10) | 2 | 0.174 | 1.13 | 24 | 3.2% |

**K-Means ranked first overall** and was very stable (Silhouette 0.1974 ± 0.0001 across 5 seeds). DBSCAN's good Davies-Bouldin is misleading because 96.7% of customers fall into one cluster. This matched my business intuition: the four K-Means groups are easy to explain. A Silhouette near 0.2 means the segments overlap, so they are soft groups.

## 4. Cardholder segments (K-Means)
- **Premium Transactors (18.4%)**: highest spend and credit limit, buy almost every month, often pay in full.
- **Revolvers (27.1%)**: large balances and high limit usage, pay only about twice the minimum, so they earn the bank the most interest.
- **Cash-Advance Reliant (26.4%)**: almost no purchases but heavy cash advances; they use the card like a loan, a higher credit-risk group.
- **Low-Activity Light Users (28.1%)**: tiny balances and low spending, nearly dormant and at risk of churn.

DBSCAN added a useful extra view: 284 "noise" customers (3.2%) with about 3.3× the cash advance and 4.4× the payment ratio of normal customers, plus a 15-customer group that buys but never pays. These need manual review by credit-risk and Relationship Managers.

## 5. What I would do next
1. Add merchant-category, repayment-history, bureau-score and income data.
2. Use a monthly time series instead of one 12-month snapshot.
3. Try semi-supervised refinement using real churn and revenue labels.
4. Wrap `predict_segment()` in a real-time scoring API (FastAPI) and run DBSCAN as a separate risk watch-list.
