-- Telco Churn — SQL Analysis Layer
-- ---------------------------------------------------------------------------
-- These queries reproduce the project's core churn-segment analysis directly
-- in SQL. They run against an in-memory DuckDB view named `customers` that is
-- created from data/raw/telco_customer_churn.csv by scripts/sql_analysis.py.
--
-- The view exposes two derived columns on top of the raw data:
--   churn_flag    INTEGER  -- 1 if the customer churned, else 0
--   total_charges DOUBLE   -- TotalCharges cast to numeric (blank -> NULL)
--
-- Each query below is preceded by a `-- name:` header used by the runner to
-- label and (optionally) export its result set.
-- ---------------------------------------------------------------------------


-- name: overall_churn_rate
-- Baseline churn rate for the whole customer base — the reference point that
-- every segment below is compared against.
SELECT
    COUNT(*)                                    AS customers,
    SUM(churn_flag)                             AS churned,
    ROUND(100.0 * AVG(churn_flag), 2)           AS churn_rate_pct
FROM customers;


-- name: churn_by_contract
-- Churn rate by contract type. Classic GROUP BY with a conditional aggregate
-- (AVG of the 0/1 flag) to compute a rate per segment.
SELECT
    Contract                                    AS contract,
    COUNT(*)                                     AS customers,
    SUM(churn_flag)                              AS churned,
    ROUND(100.0 * AVG(churn_flag), 2)            AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;


-- name: churn_by_payment_method
-- Churn rate by payment method — surfaces the electronic-check risk signal.
SELECT
    PaymentMethod                                AS payment_method,
    COUNT(*)                                     AS customers,
    SUM(churn_flag)                              AS churned,
    ROUND(100.0 * AVG(churn_flag), 2)            AS churn_rate_pct
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;


-- name: churn_by_internet_service
-- Churn rate by internet service — isolates the fiber-optic experience issue.
SELECT
    InternetService                              AS internet_service,
    COUNT(*)                                     AS customers,
    SUM(churn_flag)                              AS churned,
    ROUND(100.0 * AVG(churn_flag), 2)            AS churn_rate_pct
FROM customers
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;


-- name: churn_by_tenure_band
-- Tenure cohort analysis. CASE WHEN buckets a continuous variable (tenure in
-- months) into interpretable, ordered bands before aggregating.
SELECT
    CASE
        WHEN tenure <= 12 THEN '0-12'
        WHEN tenure <= 24 THEN '13-24'
        WHEN tenure <= 48 THEN '25-48'
        ELSE '49-72'
    END                                          AS tenure_band,
    COUNT(*)                                     AS customers,
    SUM(churn_flag)                              AS churned,
    ROUND(100.0 * AVG(churn_flag), 2)            AS churn_rate_pct
FROM customers
GROUP BY tenure_band
ORDER BY tenure_band;


-- name: high_risk_segment
-- Compound high-risk segment: combines the three strongest individual drivers
-- (month-to-month + electronic check + first-year tenure) to show how risk
-- compounds. Demonstrates multi-condition filtering against the baseline.
SELECT
    'month-to-month + e-check + tenure<=12'      AS segment,
    COUNT(*)                                     AS customers,
    SUM(churn_flag)                              AS churned,
    ROUND(100.0 * AVG(churn_flag), 2)            AS churn_rate_pct
FROM customers
WHERE Contract = 'Month-to-month'
  AND PaymentMethod = 'Electronic check'
  AND tenure <= 12;


-- name: contract_churn_ranked
-- Window function: rank contract segments by churn rate and express each
-- segment's rate relative to the overall base rate (a "lift" style metric).
SELECT
    Contract                                                         AS contract,
    COUNT(*)                                                          AS customers,
    ROUND(100.0 * AVG(churn_flag), 2)                                 AS churn_rate_pct,
    RANK() OVER (ORDER BY AVG(churn_flag) DESC)                       AS risk_rank,
    ROUND(AVG(churn_flag) / (SELECT AVG(churn_flag) FROM customers), 2) AS rate_vs_baseline
FROM customers
GROUP BY Contract
ORDER BY risk_rank;


-- name: monthly_revenue_at_risk
-- Business framing: monthly recurring revenue currently attached to churned
-- customers, by contract type — i.e. the revenue actively walking out the door.
SELECT
    Contract                                     AS contract,
    SUM(churn_flag)                              AS churned_customers,
    ROUND(SUM(CASE WHEN churn_flag = 1 THEN MonthlyCharges ELSE 0 END), 2)
                                                 AS monthly_revenue_lost,
    ROUND(12 * SUM(CASE WHEN churn_flag = 1 THEN MonthlyCharges ELSE 0 END), 2)
                                                 AS annualized_revenue_lost
FROM customers
GROUP BY Contract
ORDER BY monthly_revenue_lost DESC;
