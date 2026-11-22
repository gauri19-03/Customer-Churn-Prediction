-- Calculate the total number of customers
SELECT
    COUNT(*) AS total_customers
FROM customers;


-- Count customers who left and customers
-- who continued using the service
SELECT
    Churn,
    COUNT(*) AS customer_count
FROM customers
GROUP BY Churn;

-- Calculate the total customers,
-- churned customers and overall churn percentage
SELECT
    COUNT(*) AS total_customers,

    SUM(
        CASE
            WHEN Churn = 'Yes' THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        SUM(
            CASE
                WHEN Churn = 'Yes' THEN 1
                ELSE 0
            END
        ) * 100.0 / COUNT(*),
        2
    ) AS churn_rate
FROM customers;

-- Analyze which contract type has the
-- highest customer churn percentage

SELECT
    Contract,

    COUNT(*) AS total_customers,

    SUM(
        CASE
            WHEN Churn = 'Yes' THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        AVG(
            CASE
                WHEN Churn = 'Yes' THEN 1.0
                ELSE 0.0
            END
        ) * 100,
        2
    ) AS churn_percentage

FROM customers
GROUP BY Contract
ORDER BY churn_percentage DESC;

-- Analyze customer churn across different
-- payment methods
SELECT
    PaymentMethod,
    COUNT(*) AS total_customers,

    SUM(
        CASE
            WHEN Churn = 'Yes' THEN 1
            ELSE 0
        END
    ) AS churned_customers

FROM customers
GROUP BY PaymentMethod
ORDER BY churned_customers DESC;


-- Compare the average monthly charges
-- of customers who stayed and customers who left
SELECT
    Churn,
    ROUND(AVG(MonthlyCharges), 2)
        AS average_monthly_charges
FROM customers
GROUP BY Churn;


-- Analyze churn patterns for different
-- types of internet services
SELECT
    InternetService,
    COUNT(*) AS total_customers,

    SUM(
        CASE
            WHEN Churn = 'Yes' THEN 1
            ELSE 0
        END
    ) AS churned_customers

FROM customers
GROUP BY InternetService;

--IDENTIFY HIGH-RISK CUSTOMER SEGMENTS
SELECT
    customerID,
    Contract,
    tenure,
    MonthlyCharges,
    Churn

FROM customers

WHERE Contract = 'Month-to-month'
AND MonthlyCharges > 70
AND tenure < 12

ORDER BY MonthlyCharges DESC;