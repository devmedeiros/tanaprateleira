-- ============================================================================
-- SILVER LAYER - ORDERS DOMAIN
-- Arquivo: silver/02_orders.sql
-- Responsabilidade: LIMPEZA APENAS (cada tabela separada, sem joins)
-- ============================================================================

-- ============================================================================
-- 1. SILVER: ORDERS (limpo)
-- ============================================================================

CREATE MATERIALIZED VIEW silver.orders
TBLPROPERTIES (
  "quality" = "silver",
  "domain" = "orders",
  "type" = "entity",
  "grain" = "order_id",
  "description" = "Pedidos limpos e validados"
)
AS
SELECT
  order_id,
  customer_id,
  UPPER(TRIM(order_status)) AS order_status,
  order_purchase_timestamp,
  order_approved_at,
  order_delivered_carrier_date,
  order_delivered_customer_date,
  order_estimated_delivery_date,
  CASE 
    WHEN order_id IS NULL THEN 'invalid_null_id'
    WHEN order_purchase_timestamp IS NULL THEN 'missing_purchase_date'
    WHEN UPPER(TRIM(order_status)) NOT IN ('PENDING_PAYMENT', 'PROCESSING', 'SHIPPED', 'DELIVERED', 'UNAVAILABLE', 'CANCELED', 'APPROVED', 'INVOICED', 'CREATED')
      THEN 'invalid_status'
    ELSE 'valid'
  END AS data_quality_flag,
  _source_file,
  _ingestion_time,
  _processed_at
FROM bronze.orders
WHERE
  order_id IS NOT NULL;

-- ============================================================================
-- 2. SILVER: ORDER ITEMS (limpo)
-- ============================================================================

CREATE MATERIALIZED VIEW silver.order_items
TBLPROPERTIES (
  "quality" = "silver",
  "domain" = "orders",
  "subdomain" = "items",
  "type" = "entity",
  "grain" = "order_id + order_item_id",
  "description" = "Itens de pedido limpos e validados"
)
AS
SELECT
  order_id,
  order_item_id,
  product_id,
  seller_id,
  shipping_limit_date,
  CAST(price AS DECIMAL(10, 2)) AS price,
  CAST(freight_value AS DECIMAL(10, 2)) AS freight_value,
  CASE 
    WHEN order_id IS NULL THEN 'invalid_null_order_id'
    WHEN product_id IS NULL THEN 'invalid_null_product_id'
    WHEN price < 0 THEN 'negative_price'
    WHEN freight_value < 0 THEN 'negative_freight'
    ELSE 'valid'
  END AS data_quality_flag,
  _source_file,
  _ingestion_time,
  _processed_at
FROM bronze.order_items
WHERE
  order_id IS NOT NULL AND product_id IS NOT NULL;

-- ============================================================================
-- 3. SILVER: ORDER PAYMENTS (limpo)
-- ============================================================================

CREATE MATERIALIZED VIEW silver.order_payments
TBLPROPERTIES (
  "quality" = "silver",
  "domain" = "orders",
  "subdomain" = "payments",
  "type" = "entity",
  "grain" = "order_id + payment_sequential",
  "description" = "Pagamentos limpos e validados"
)
AS
SELECT
  order_id,
  payment_sequential,
  LOWER(TRIM(payment_type)) AS payment_type,
  payment_installments,
  CAST(payment_value AS DECIMAL(10, 2)) AS payment_value,
  CASE 
    WHEN order_id IS NULL THEN 'invalid_null_order_id'
    WHEN payment_value < 0 THEN 'invalid_payment_value'
    WHEN payment_type NOT IN ('credit_card', 'boleto', 'debit_card', 'voucher')
      THEN 'invalid_payment_type'
    ELSE 'valid'
  END AS data_quality_flag,
  _source_file,
  _ingestion_time,
  _processed_at
FROM bronze.order_payments
WHERE
  order_id IS NOT NULL;

-- ============================================================================
-- 4. SILVER: ORDER REVIEWS (limpo)
-- ============================================================================

CREATE MATERIALIZED VIEW silver.order_reviews
TBLPROPERTIES (
  "quality" = "silver",
  "domain" = "orders",
  "subdomain" = "reviews",
  "type" = "entity",
  "grain" = "review_id",
  "description" = "Avaliações limpas e validadas"
)
AS
SELECT
  review_id,
  order_id,
  CAST(review_score AS INT) AS review_score,
  TRIM(review_comment_title) AS review_comment_title,
  TRIM(review_comment_message) AS review_comment_message,
  review_creation_date,
  review_answer_timestamp,
  CASE 
    WHEN review_id IS NULL THEN 'invalid_null_review_id'
    WHEN order_id IS NULL THEN 'invalid_null_order_id'
    WHEN review_score NOT IN (1, 2, 3, 4, 5) THEN 'invalid_score'
    ELSE 'valid'
  END AS data_quality_flag,
  _source_file,
  _ingestion_time,
  _processed_at
FROM bronze.order_reviews
WHERE
  review_id IS NOT NULL;