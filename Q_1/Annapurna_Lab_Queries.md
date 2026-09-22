# Annapurna Lab – Query Sheet

Each section gives the **side topic** followed by the command/query.

## 1. PostgreSQL – Start Server
**Side topic: Start PostgreSQL Docker container**
```cmd
docker ps
docker start postgres
docker ps
```

## 2. PostgreSQL – Connect
**Side topic: Open the sales database**
```cmd
docker exec -it postgres psql -U admin -d salesdb
```

## 3. PostgreSQL – Verify Stores
**Side topic: Check store master count**
```sql
SELECT COUNT(*) FROM stores;
```
Expected: `12`

## 4. PostgreSQL – Verify Products
**Side topic: Check product master count**
```sql
SELECT COUNT(*) FROM products;
```
Expected: `1224`

## 5. PostgreSQL – Verify Categories
**Side topic: Check category master count**
```sql
SELECT COUNT(*) FROM product_categories;
```
Expected: `14`

## 6. PostgreSQL – Verify Price Revisions
**Side topic: Check historical price records**
```sql
SELECT COUNT(*) FROM price_revisions;
```
Expected: `4320`

## 7. Product Reissue
**Side topic: Prove product_code can be reissued**
```sql
SELECT product_sk, product_code, product_name, category_id, valid_from, valid_to
FROM products
WHERE product_code = 'P100621'
ORDER BY valid_from;
```

## 8. DuckDB – Open Database
**Side topic: Open Annapurna analytical database**
```cmd
duckdb
```
```sql
.open duckdb/annapurna.duckdb
SELECT 1;
```

## 9. DuckDB + MinIO
**Side topic: Enable S3/HTTPFS**
```sql
INSTALL httpfs;
LOAD httpfs;
```

## 10. MinIO Credentials
**Side topic: Configure DuckDB for MinIO**
```sql
DROP SECRET IF EXISTS minio_secret;

CREATE SECRET minio_secret (
    TYPE S3,
    KEY_ID 'minioadmin',
    SECRET 'minioadmin123',
    ENDPOINT 'localhost:9000',
    URL_STYLE 'path',
    USE_SSL false,
    REGION 'us-east-1'
);
```

## 11. Read MinIO Parquet
**Side topic: Direct object-store query**
```sql
SELECT *
FROM read_parquet(
    's3://annapurna/canonical_sales/store_id=S01/business_month=2024-10/*.parquet'
)
LIMIT 5;
```

## 12. Store + Month Revenue
**Side topic: October revenue for one store**
```sql
SELECT
    COUNT(*) AS october_rows,
    SUM(
        CASE
            WHEN line_type IN ('SALE','RETURN','DISCOUNT','VOID')
            THEN qty * unit_price
            ELSE 0
        END
    ) AS october_revenue
FROM read_parquet(
    'output/canonical_sales/**/*.parquet',
    hive_partitioning = true
)
WHERE store_id = 'S01'
  AND business_month = '2024-10';
```

## 13. EXPLAIN
**Side topic: Show query plan**
```sql
EXPLAIN
SELECT
    COUNT(*)
FROM read_parquet(
    'output/canonical_sales/**/*.parquet',
    hive_partitioning = true
)
WHERE store_id = 'S01'
  AND business_month = '2024-10';
```

## 14. EXPLAIN ANALYZE
**Side topic: Prove partition pruning / files scanned**
```sql
EXPLAIN ANALYZE
SELECT
    COUNT(*) AS october_rows,
    SUM(
        CASE
            WHEN line_type IN ('SALE','RETURN','DISCOUNT','VOID')
            THEN qty * unit_price
            ELSE 0
        END
    ) AS october_revenue
FROM read_parquet(
    'output/canonical_sales/**/*.parquet',
    hive_partitioning = true
)
WHERE store_id = 'S01'
  AND business_month = '2024-10';
```
Look for:
`Scanning Files: 1/146` and `Total Files Read: 1`.

## 15. Store Dimension
**Side topic: Create dimension table**
```sql
CREATE OR REPLACE TABLE dim_store AS
SELECT store_id, store_name, address_line, city, state
FROM pg.public.stores;
```

## 16. Category Dimension
**Side topic: Create category dimension**
```sql
CREATE OR REPLACE TABLE dim_category AS
SELECT category_id, category_name
FROM pg.public.product_categories;
```

## 17. Product Dimension
**Side topic: Create product dimension**
```sql
CREATE OR REPLACE TABLE dim_product AS
SELECT
    product_sk, product_code, product_name, category_id,
    brand, pack_size, uom, valid_from, valid_to
FROM pg.public.products;
```

## 18. Date Dimension
**Side topic: Create date dimension**
```sql
CREATE OR REPLACE TABLE dim_date AS
SELECT DISTINCT
    business_date AS date_key,
    strftime(business_date, '%Y-%m') AS month,
    strftime(business_date, '%A') AS day_of_week,
    EXTRACT(DAY FROM business_date) AS day_of_month,
    EXTRACT(MONTH FROM business_date) AS month_number,
    EXTRACT(YEAR FROM business_date) AS year
FROM read_parquet(
    'output/canonical_sales/**/*.parquet',
    hive_partitioning = true
);
```

## 19. Fact Table
**Side topic: Create sales fact table**
```sql
CREATE OR REPLACE TABLE fact_sales AS
SELECT
    s.store_id,
    s.business_date AS date_key,
    p.product_sk,
    p.category_id,
    s.bill_no,
    s.line_no,
    s.product_code,
    s.qty,
    s.unit_price,
    s.line_type,
    CASE
        WHEN s.line_type IN ('SALE','RETURN','DISCOUNT','VOID')
        THEN s.qty * s.unit_price
        ELSE 0
    END AS revenue
FROM read_parquet(
    'output/canonical_sales/**/*.parquet',
    hive_partitioning = true
) s
LEFT JOIN pg.public.products p
    ON s.product_code = p.product_code
   AND s.business_date BETWEEN p.valid_from AND p.valid_to;
```

## 20. Revenue by Store
**Side topic: Store-level analysis**
```sql
SELECT store_id, ROUND(SUM(revenue), 2) AS revenue
FROM fact_sales
GROUP BY store_id
ORDER BY revenue DESC;
```

## 21. Revenue by Category
**Side topic: Category-level analysis**
```sql
SELECT
    c.category_name,
    ROUND(SUM(f.revenue), 2) AS revenue
FROM fact_sales f
JOIN dim_category c ON f.category_id = c.category_id
GROUP BY c.category_name
ORDER BY revenue DESC;
```

## 22. Revenue by Day
**Side topic: Day-of-week analysis**
```sql
SELECT
    d.day_of_week,
    ROUND(SUM(f.revenue), 2) AS revenue
FROM fact_sales f
JOIN dim_date d ON f.date_key = d.date_key
GROUP BY d.day_of_week
ORDER BY revenue DESC;
```

## 23. Revenue by Month
**Side topic: Monthly analysis**
```sql
SELECT
    strftime(date_key, '%Y-%m') AS month,
    ROUND(SUM(revenue), 2) AS revenue
FROM fact_sales
GROUP BY strftime(date_key, '%Y-%m')
ORDER BY month;
```

## 24. Historical Pricing
**Side topic: Use price valid on the sale date**
```sql
SELECT
    strftime(s.business_date, '%Y-%m') AS reporting_month,
    ROUND(
        SUM(
            CASE
                WHEN s.line_type IN ('SALE','RETURN','VOID')
                THEN s.qty * pr.selling_price
                ELSE 0
            END
        ), 2
    ) AS revenue_at_historical_price
FROM read_parquet(
    'output/canonical_sales/**/*.parquet',
    hive_partitioning = true
) s
JOIN pg.public.products p
    ON s.product_code = p.product_code
   AND s.business_date BETWEEN p.valid_from AND p.valid_to
JOIN pg.public.price_revisions pr
    ON p.product_sk = pr.product_sk
   AND s.business_date BETWEEN pr.effective_from AND pr.effective_to
WHERE s.business_date >= DATE '2024-12-01'
  AND s.business_date < DATE '2025-01-01'
GROUP BY strftime(s.business_date, '%Y-%m');
```

## 25. PostgreSQL Attachment
**Side topic: Connect DuckDB to PostgreSQL**
```sql
LOAD postgres;

ATTACH 'host=localhost port=5432 dbname=salesdb user=admin password=admin'
AS pg (TYPE POSTGRES);
```

## 26. Cross-System Query
**Side topic: Join MinIO sales with PostgreSQL products**
```sql
SELECT
    s.store_id,
    s.business_month,
    p.product_code,
    p.product_name,
    COUNT(*) AS sales_lines,
    ROUND(
        SUM(
            CASE
                WHEN s.line_type IN ('SALE','RETURN','DISCOUNT','VOID')
                THEN s.qty * s.unit_price
                ELSE 0
            END
        ), 2
    ) AS revenue
FROM read_parquet(
    's3://annapurna/canonical_sales/store_id=S01/business_month=2024-10/*.parquet'
) s
JOIN pg.public.products p
    ON s.product_code = p.product_code
   AND s.business_date BETWEEN p.valid_from AND p.valid_to
WHERE s.store_id = 'S01'
  AND s.business_month = '2024-10'
GROUP BY s.store_id, s.business_month, p.product_code, p.product_name
ORDER BY revenue DESC
LIMIT 10;
```

## 27. Cross-System Query Plan
**Side topic: EXPLAIN the MinIO + PostgreSQL join**
```sql
EXPLAIN ANALYZE
SELECT
    s.store_id,
    s.business_month,
    p.product_code,
    p.product_name,
    COUNT(*) AS sales_lines,
    ROUND(
        SUM(
            CASE
                WHEN s.line_type IN ('SALE','RETURN','DISCOUNT','VOID')
                THEN s.qty * s.unit_price
                ELSE 0
            END
        ), 2
    ) AS revenue
FROM read_parquet(
    's3://annapurna/canonical_sales/store_id=S01/business_month=2024-10/*.parquet'
) s
JOIN pg.public.products p
    ON s.product_code = p.product_code
   AND s.business_date BETWEEN p.valid_from AND p.valid_to
WHERE s.store_id = 'S01'
  AND s.business_month = '2024-10'
GROUP BY s.store_id, s.business_month, p.product_code, p.product_name;
```
Look for `HASH_JOIN`, `READ_PARQUET` and the file-read statistics.

## 28. Idempotency Audit
**Side topic: Create load audit table**
```sql
CREATE OR REPLACE TABLE load_audit (
    run_id INTEGER,
    run_time TIMESTAMP,
    row_count BIGINT,
    checksum VARCHAR
);
```

## 29. Checksum
**Side topic: Verify row count and dataset checksum**
```sql
SELECT
    COUNT(*) AS row_count,
    md5(
        CAST(
            SUM(
                hash(
                    bill_no,
                    line_no,
                    product_code,
                    qty,
                    unit_price,
                    line_type
                )
            ) AS VARCHAR
        )
        || ':' ||
        CAST(COUNT(*) AS VARCHAR)
    ) AS checksum
FROM read_parquet(
    'output/canonical_sales/**/*.parquet',
    hive_partitioning = true
);
```

Expected:
`1,120,924` rows and checksum `ecdebfb1c6edc5f7de10befbf94b616b`.

## 30. Duplicate Validation
**Side topic: Confirm no duplicate line identities remain**
```sql
SELECT
    bill_no,
    line_no,
    COUNT(*) AS duplicate_count
FROM read_parquet(
    'output/canonical_sales/**/*.parquet',
    hive_partitioning = true
)
GROUP BY bill_no, line_no
HAVING COUNT(*) > 1;
```
Expected: **0 rows**.

## 31. Finance Reconciliation
**Side topic: Compare analytical revenue with finance**
```sql
CREATE OR REPLACE TABLE reconciliation AS
SELECT
    month,
    calculated_revenue,
    finance_revenue,
    ROUND(finance_revenue - calculated_revenue, 2) AS difference,
    CASE
        WHEN ABS(finance_revenue - calculated_revenue) < 0.01
            THEN 'MATCH'
        WHEN month = '2024-03'
            THEN 'SOURCE / OUTSIDE-TILL INSTITUTIONAL ORDER'
        WHEN month = '2024-07'
            THEN 'SOURCE DATA GAP / S07 MISSING DAYS'
        WHEN month = '2024-12'
            THEN 'REVENUE DEFINITION / ROUNDING'
        ELSE 'INVESTIGATE PIPELINE'
    END AS classification
FROM (
    SELECT
        strftime(date_key, '%Y-%m') AS month,
        ROUND(SUM(revenue), 2) AS calculated_revenue
    FROM fact_sales
    GROUP BY strftime(date_key, '%Y-%m')
) sales
JOIN (
    SELECT SUBSTR(month, 1, 7) AS month, revenue_inr AS finance_revenue
    FROM read_csv('finance_monthly.csv')
) finance
USING (month)
ORDER BY month;
```

## 32. View Reconciliation
**Side topic: Display monthly reconciliation**
```sql
SELECT *
FROM reconciliation
ORDER BY month;
```

Known differences:
- March: `₹486,250.00`
- July: `₹232,131.70`
- December: `-₹50.48`

---


