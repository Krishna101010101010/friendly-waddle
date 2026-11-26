# Online Retail: Preprocessing, Data Warehouse, OLAP & Churn Prediction — Analysis

**Source Notebook:** [`online-retail-preprocessing-data-warehouse-olap.ipynb`](./online-retail-preprocessing-data-warehouse-olap.ipynb)  
**Dataset Archive:** [`data.csv.zip`](./data.csv.zip) (`data.csv`)  
**Notebook Author:** Krishna  

---

## 1. Executive Summary

This notebook implements an end-to-end retail analytics and machine learning pipeline on the **UCI Online Retail** dataset. It progresses from raw transaction cleaning and cancellation netting into a validated **5-table Star Schema**, **OLAP cube operations**, **exploratory sales visuals**, **customer segmentation (RFM, K-Means, Cohort Retention)**, **Market Basket Analysis**, and **leakage-free 90-day Churn Prediction**, concluding with an automated **data quality audit and CSV export** of 8 analytical tables.

---

## 2. Dataset Information (`data.csv.zip`)

| Property | Value |
| :--- | :--- |
| **Archive File** | `data.csv.zip` (7,548,532 bytes compressed) |
| **Extracted File** | `data.csv` (45,580,638 bytes uncompressed, `ISO-8859-1` / `latin1`) |
| **Raw Dimensions** | **541,909 rows × 8 columns** |
| **Date Range** | `2010-12-01 08:26:00` to `2011-12-09 12:50:00` *(Dec 2011 is partial)* |
| **Exact Duplicate Rows** | 5,268 |
| **Cancelled Invoices (`C` prefix)** | 9,288 rows |
| **Negative / Zero Quantities** | 10,624 rows (min `-80,995`, max `80,995`) |
| **Non-Positive Unit Prices** | 2,517 rows (min `-11,062.06`, max `38,970.00`) |

### Column Schema & Raw Quality Profile

| Column | Raw Dtype | Non-Null Count | Missing Count (%) | Unique Values | Notes |
| :--- | :--- | ---: | ---: | ---: | :--- |
| `InvoiceNo` | `object` (`str`) | 541,909 | 0 (0.00%) | 25,900 | 6-digit invoice ID; `C` prefix denotes cancellation |
| `StockCode` | `object` (`str`) | 541,909 | 0 (0.00%) | 4,070 | Product SKU (5 digits or `DCGS*`) plus non-product fee/adjustment codes |
| `Description` | `object` | 540,455 | 1,454 (0.27%) | 4,223 | Item description (some SKUs have multiple text variants) |
| `Quantity` | `int64` | 541,909 | 0 (0.00%) | — | Units per line; negative for returns/cancellations/adjustments |
| `InvoiceDate` | `object` | 541,909 | 0 (0.00%) | — | Format `%m/%d/%Y %H:%M` |
| `UnitPrice` | `float64` | 541,909 | 0 (0.00%) | — | Price per unit in GBP (£) |
| `CustomerID` | `float64` | 406,829 | 135,080 (24.93%) | 4,372 | Numeric customer identifier (`12346` to `18287`) |
| `Country` | `object` | 541,909 | 0 (0.00%) | 38 | Heavily UK-skewed (see below) |

### Geographic Distribution (Top 5 Countries in Raw Data)

1. **United Kingdom:** 495,478 rows (91.43%)
2. **Germany:** 9,495 rows (1.75%)
3. **France:** 8,557 rows (1.58%)
4. **EIRE:** 8,196 rows (1.51%)
5. **Spain:** 2,533 rows (0.47%)

---

## 3. Notebook Pipeline Breakdown

### Section 1 & 2: Setup & Exploration
- Imports `pandas`, `numpy`, `scipy.sparse`, `matplotlib`, `seaborn`, and `scikit-learn` (`SEED = 42`).
- Resolves `data.csv` from `DATA_PATH`, `/kaggle/input/...`, or the current working directory.
  > **Local Execution Note:** The workspace currently contains `data.csv.zip`. Unzip it to `data.csv` (or point `DATA_PATH` to the extracted file) before running the notebook locally.
- Profiles shape, data types, summary statistics, duplicates (`5,268`), and missingness (`CustomerID` 24.93%, `Description` 0.27%).

### Section 3: Cleaning & Cancellation Netting
1. **Deduplication & Typing:** Drops `5,268` exact duplicate rows (`536,641` rows remaining). Parses `InvoiceDate`, casts `CustomerID` to nullable `Int64`, uppercases/strips `StockCode`, and assigns each `StockCode` its single most frequent `Description` (mode).
2. **Non-Product Filtering:** Keeps only SKUs matching `^\d{5}` or `DCGS*` (excluding `ebay` / `unknown product` descriptions). Removes operational/accounting codes such as `POST` (1,256 rows), `DOT` (710), `M` (567), `C2` (144), `D` (77), `S` (62), `BANK CHARGES` (37), `AMAZONFEE` (34), `CRUK` (16), gift vouchers (`GIFT_0001_*`), `PADS`, and `B` (bad debt).
3. **Cancellation Netting (`net_cancellations`):**
   - Rather than blindly deleting `C`-prefixed rows (which leaves the original reversed purchase in revenue), each cancellation is matched to prior purchases with the same `(CustomerID, StockCode, UnitPrice)` **latest-first**, subtracting the cancelled quantity.
   - **Gross product revenue:** `10,247,219.32`
   - **Net product revenue:** `9,804,875.19`
   - **Matched cancelled units:** `246,441` of `267,711` (with `CustomerID`; `162` cancellation rows lacked `CustomerID`)
   - **Final net sales rows (`df_sales`):** `519,815`

### Section 4: Outlier Detection
- Flags `UnitPrice` and `Quantity` outliers via the `1.5 × IQR` rule **without removing them**, preserving legitimate wholesale/bulk orders and high-value items:
  - `UnitPrice`: bounds `[-3.07, 8.45]` → **35,289** flagged rows
  - `Quantity`: bounds `[-14.00, 26.00]` → **26,653** flagged rows
- Visualizes both distributions using log-scale boxplots.

### Section 5: Data Warehouse (Star Schema)
Builds and validates a Star Schema at the **invoice line-item grain**:

| Table | Shape | Role & Key Details |
| :--- | :---: | :--- |
| `fact_sales` | `(519,815, 10)` | Grain: one product line per invoice (`SalesKey`, `InvoiceNo`, foreign keys, `Hour`, `Quantity`, `UnitPrice`, `TotalAmount`) |
| `dim_customer` | `(4,325, 3)` | Unique customers + `CustomerKey = 0` (`"Unknown"`) for guest/missing-ID sales so warehouse totals reconcile |
| `dim_product` | `(3,793, 4)` | `ProductKey`, `StockCode`, standardized `Description`, `MedianUnitPrice` |
| `dim_date` | `(374, 10)` | Continuous daily calendar (`2010-12-01` to `2011-12-09`) with `HasSales` flag |
| `dim_country` | `(38, 2)` | `CountryKey` and `Country` |

- **Automated Assertions:** Verifies surrogate key uniqueness across all dimensions, 100% referential integrity on `fact_sales`, exact row count preservation (`519,815`), and exact revenue conservation (`TotalAmount` sum).

### Section 6: OLAP Operations
Demonstrates core multidimensional cube operations on `fact_sales` joined with `dim_date` and `dim_country`:
- **Roll-up (Year):** `2010` (`760,990.88` across 1,541 invoices) vs. `2011` (`9,043,884.31` across 18,106 invoices).
- **Drill-down (`Year > Quarter > Month`):** Monthly progression peaking in **November 2011** (`1,435,611.94` revenue, `738,435` units, `2,739` invoices).
- **Slice (`Year = 2011`):** Isolates full-year 2011 monthly metrics.
- **Dice (`2011, Q1, invoice total > 100`):** Filters to higher-value Q1 2011 invoices (e.g., Jan `557,437.00` across 958 invoices).
- **Pivot:** Cross-tabulates revenue by `Year × Month`.

### Section 7: Sales Analysis
Generates four exploratory visualizations:
1. **Monthly Net Revenue Trend** (annotating `2011-12` as partial).
2. **Top 10 Products by Revenue.**
3. **Top 10 Countries Outside the UK by Revenue.**
4. **Invoices by Weekday and Hour Heatmap.**

### Section 8: Customer Analysis
1. **RFM Scoring (`1`–`5` quintiles via `rank(pct=True)`):**
   - **Champions:** 1,017 customers | Avg Recency 13.1d | Avg Freq 10.6 | Revenue `5,543,048.0`
   - **Loyal:** 1,050 customers | Avg Recency 35.2d | Avg Freq 3.4 | Revenue `1,357,182.5`
   - **At Risk:** 755 customers | Avg Recency 154.4d | Avg Freq 3.2 | Revenue `855,259.5`
   - **Hibernating or Occasional:** 1,266 customers | Avg Recency 183.4d | Avg Freq 1.0 | Revenue `464,476.7`
   - **New or Promising:** 236 customers | Avg Recency 19.4d | Avg Freq 1.0 | Revenue `74,916.8`
2. **K-Means Clustering:** Applies `log1p` + `StandardScaler` to `(Recency, Frequency, Monetary)`, evaluates silhouette scores for `k = 2..8`, and selects `K = 4` for actionable segmentation (Cluster sizes: `0`: 1,183, `1`: 1,599, `2`: 846, `3`: 696; Cluster 3 represents high-value VIPs with median Monetary `3,610.62`, Frequency `10`, Recency `9` days).
3. **Cohort Retention:** Monthly acquisition cohort matrix showing percentage of returning customers over `MonthsSince` first purchase.

### Section 9: Market Basket Analysis
- Uses a `scipy.sparse.csr_matrix` co-occurrence computation over the **top 200 most frequent products** (`StockCode` by unique invoices).
- Filters for `Support >= 0.01` and `Confidence >= 0.30`, discovering **555 association rules**.
- Highest-lift pair (`27.674` lift, `0.758` confidence, `0.020` support): `WOODEN STAR CHRISTMAS SCANDINAVIAN` → `WOODEN HEART CHRISTMAS SCANDINAVIAN`.

### Section 10: Churn Prediction
- **Target Definition:** `Churn = 1` if a customer makes **no purchase in the 90 days after the cutoff** (`2011-09-11`).
- **Cohort Size:** `3,363` pre-cutoff customers (`1,448` Churned, `1,915` Active).
- **Leakage Prevention:** `churn_dataset(cutoff)` recomputes `net_cancellations` strictly on pre-cutoff data for features (`TotalOrders`, `TotalQuantity`, `TotalRevenue`, `AverageUnitPrice`, `UniqueProducts`, `AverageOrderValue`, `Recency`, `Tenure`, `OrdersPerMonth`, `OrdersLast30Days`, `OrdersLast90Days`, `CancelledOrders`, `IsUK`) and strictly on the `[cutoff, cutoff + 90d)` window for labels.
- **Model Comparison (80/20 stratified split, 5-fold `StratifiedKFold` CV):**

| Model | CV ROC-AUC | CV Std | Accuracy | Precision | Recall | F1 | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (majority)** | 0.5000 | 0.0000 | 0.5691 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| **Logistic Regression** *(selected)* | **0.7524** | 0.0267 | **0.6716** | 0.5966 | **0.7345** | **0.6584** | **0.7383** |
| **Random Forest** | 0.7500 | 0.0280 | 0.6642 | 0.5904 | 0.7207 | 0.6491 | 0.7320 |
| **Gradient Boosting** | 0.7504 | 0.0260 | 0.6597 | **0.5981** | 0.6414 | 0.6190 | 0.7362 |

- **Diagnostics & Validation:**
  - Plots Confusion Matrix, ROC curves, and Permutation Importance (`n_repeats=20` on `roc_auc`).
  - **Out-of-Time Check (Section 10.1):** Trains on an earlier 90-day cutoff (`2,814` customers) and evaluates on the final cutoff (`3,363` customers), achieving **0.7434 ROC-AUC** and **0.6711 Accuracy**.
  - **Out-of-Fold Scoring:** Uses `cross_val_predict` to assign unbiased `ChurnProbability` scores and flags top-25% revenue customers at highest risk.

### Section 11: Data Quality Checks & Export
Audits and exports 8 clean CSV files (to `/kaggle/working` or `output/`):

| Output File | Rows | Columns | Missing Cells | Duplicate Rows |
| :--- | ---: | ---: | ---: | ---: |
| `fact_sales.csv` | 519,815 | 10 | 0 | 0 |
| `dim_customer.csv` | 4,325 | 3 | 1 *(Unknown `CustomerID`)* | 0 |
| `dim_product.csv` | 3,793 | 4 | 0 | 0 |
| `dim_date.csv` | 374 | 10 | 0 | 0 |
| `dim_country.csv` | 38 | 2 | 0 | 0 |
| `customer_rfm.csv` | 4,324 | 12 | 0 | 0 |
| `market_basket_rules.csv` | 555 | 5 | 0 | 0 |
| `churn_scores.csv` | 3,363 | 5 | 0 | 0 |
