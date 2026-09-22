# Annapurna Supermarket Data Platform – Lab Exam Report

## 1. Project Overview
Built a data platform for the Annapurna supermarket case using **MinIO, PostgreSQL, DuckDB and Parquet**. The platform handles sales exports, resend duplicates, historical products/prices, analytical reporting, cross-system queries and finance reconciliation.

## 2. Source Data
Used:
- `sales/` – store sales exports
- `masters.sql` – master data
- `finance_monthly.csv` – signed-off finance revenue
- `billing_notes.md` – vendor/data-quality rules

Source volume: **4,457 files / 1,137,585 raw rows**.

Different source formats were normalized:
- S01–S05: comma CSV
- S06–S09: semicolon files with different column names/date format
- S10–S12: UTF-8 BOM CSV with epoch timestamps

## 3. Platform Setup
### MinIO
Created bucket `annapurna` and uploaded canonical Parquet under:
`annapurna/canonical_sales/`

### PostgreSQL
Loaded `masters.sql` and verified:
- Stores: **12**
- Categories: **14**
- Products: **1,224**
- Price revisions: **4,320**

### DuckDB
Created/used `duckdb/annapurna.duckdb` and configured access to local Parquet, MinIO/S3 and PostgreSQL.

## 4. Data Normalization
Normalized sales into:
`store_id, business_date, bill_no, line_no, product_code, qty, unit_price, line_type, ts, source_file`

The filename date was treated as the **business date**, as required by the vendor notes.

## 5. Duplicate / Resend Handling
Used `(bill_no, line_no)` as the sales-line identity.

Results:
- Raw rows: **1,137,585**
- Canonical rows: **1,120,924**
- Extra duplicate rows removed: **16,661**
- Remaining duplicate identities: **0**

Original source files were preserved.

**Production note:** vendor notes say some resend files are partial. A production loader should validate coverage/conflicts rather than blindly preferring the newest resend.

## 6. Canonical Parquet and Partitioning
Canonical sales were stored as Parquet and partitioned by:
- `store_id`
- `business_month`

Example:
`output/canonical_sales/store_id=S01/business_month=2024-10/data_0.parquet`

## 7. Partition Pruning
For S01 October 2024, `EXPLAIN ANALYZE` showed:
- Scanning Files: **1/146**
- Total Files Read: **1**

Result:
- October rows: **13,516**
- Revenue: **₹6,435,443.95**

This demonstrates that unrelated store/month partitions are not scanned.

## 8. Star Schema
Created:
- `dim_store`
- `dim_product`
- `dim_category`
- `dim_date`
- `fact_sales`

This supports reporting by store, category, product, date, day of week and month.

## 9. Product Identity and Historical Pricing
Product codes can be reissued, so product matching uses:
`product_code + business_date BETWEEN valid_from AND valid_to`

Historical pricing uses:
`product_sk + business_date BETWEEN effective_from AND effective_to`

Thus historical reports use the price valid for the reporting date rather than today's shelf price.

## 10. Revenue Logic
Revenue includes:
`SALE, RETURN, DISCOUNT, VOID`

Calculation:
```sql
CASE
    WHEN line_type IN ('SALE','RETURN','DISCOUNT','VOID')
    THEN qty * unit_price
    ELSE 0
END
```

`TAX` and `TENDER` are excluded from revenue.

## 11. Cross-System Query
DuckDB was connected to PostgreSQL and queried Parquet directly from MinIO.

The cross-system query joined:
- MinIO/S3 sales Parquet
- PostgreSQL `products`

`EXPLAIN ANALYZE` showed the join and that the required Parquet partition was read directly.

## 12. Idempotency Verification
Created `load_audit` with row count and checksum.

Three validation measurements produced:

```text
Run 1 → 1,120,924 → ecdebfb1c6edc5f7de10befbf94b616b
Run 2 → 1,120,924 → ecdebfb1c6edc5f7de10befbf94b616b
Run 3 → 1,120,924 → ecdebfb1c6edc5f7de10befbf94b616b
```

Note: these are three repeatability measurements of the resulting dataset, not three independent full rebuild executions.

## 13. Finance Reconciliation
Monthly analytical revenue was compared with `finance_monthly.csv`.

Matching months: **January, February, April, May, June, August, September, October and November**.

Known differences:
- **March: +₹486,250 finance vs calculated** — institutional order invoiced outside the till.
- **July: +₹232,131.70 finance vs calculated** — S07 missing 2024-07-09, 2024-07-10 and 2024-07-11 exports.
- **December: -₹50.48 finance vs calculated** — finance rounds each bill to rupees.

## 14. Final Architecture
```text
Store Sales Files
       |
       v
Normalization
       |
       v
Canonical Parquet
       |
       +----------> MinIO Object Store
       |
       v
     DuckDB
       |
       +----------> PostgreSQL Master Data
       |
       v
Analytics / Star Schema / Reconciliation
```

## 15. Final Deliverables
- MinIO object store
- PostgreSQL master database
- DuckDB analytical database
- Normalized sales
- Deduplicated canonical sales
- Store/month partitioned Parquet
- Star schema
- Historical product and price logic
- Cross-system MinIO + PostgreSQL query
- EXPLAIN ANALYZE evidence
- Idempotency/checksum audit
- Finance reconciliation
- Data-quality findings

## 16. Conclusion
The Annapurna analytical platform has been implemented with repeatable querying, partition pruning, resend handling, historical pricing, cross-system access and finance reconciliation. The known source-data and revenue-definition differences were identified rather than incorrectly treating them as pipeline errors.
