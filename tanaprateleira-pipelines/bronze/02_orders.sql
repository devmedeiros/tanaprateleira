-- ============================================================================
-- BRONZE LAYER - ORDERS DOMAIN
-- Arquivo: bronze/02_orders.sql
-- Tabelas: orders + order_items + order_payments + order_reviews
-- ============================================================================

-- ============================================================================
-- 1. BRONZE: ORDERS
-- ============================================================================

CREATE MATERIALIZED VIEW bronze.orders
TBLPROPERTIES (
  "quality" = "bronze",
  "domain" = "orders",
  "source_system" = "csv",
  "layer" = "raw_ingestion",
  "update_frequency" = "daily"
)
AS
SELECT
  *,
  _metadata.file_path AS _source_file,
  _metadata.file_modification_time AS _ingestion_time,
  current_timestamp() AS _processed_at
FROM read_files(
  '/Volumes/catalog-impacta-capstone/default/capstone/olist_orders_dataset.csv',
  format => 'csv',
  header => true,
  inferSchema => true,
  mode => 'PERMISSIVE',
  charset => 'UTF-8'
);

-- ============================================================================
-- 2. BRONZE: ORDER ITEMS
-- ============================================================================

CREATE MATERIALIZED VIEW bronze.order_items
TBLPROPERTIES (
  "quality" = "bronze",
  "domain" = "orders",
  "subdomain" = "items",
  "source_system" = "csv",
  "layer" = "raw_ingestion",
  "update_frequency" = "daily"
)
AS
SELECT
  *,
  _metadata.file_path AS _source_file,
  _metadata.file_modification_time AS _ingestion_time,
  current_timestamp() AS _processed_at
FROM read_files(
  '/Volumes/catalog-impacta-capstone/default/capstone/olist_order_items_dataset.csv',
  format => 'csv',
  header => true,
  inferSchema => true,
  mode => 'PERMISSIVE',
  charset => 'UTF-8'
);

-- ============================================================================
-- 3. BRONZE: ORDER PAYMENTS
-- ============================================================================

CREATE MATERIALIZED VIEW bronze.order_payments
TBLPROPERTIES (
  "quality" = "bronze",
  "domain" = "orders",
  "subdomain" = "payments",
  "source_system" = "csv",
  "layer" = "raw_ingestion",
  "update_frequency" = "daily"
)
AS
SELECT
  *,
  _metadata.file_path AS _source_file,
  _metadata.file_modification_time AS _ingestion_time,
  current_timestamp() AS _processed_at
FROM read_files(
  '/Volumes/catalog-impacta-capstone/default/capstone/olist_order_payments_dataset.csv',
  format => 'csv',
  header => true,
  inferSchema => true,
  mode => 'PERMISSIVE',
  charset => 'UTF-8'
);

-- ============================================================================
-- 4. BRONZE: ORDER REVIEWS
-- ============================================================================

CREATE MATERIALIZED VIEW bronze.order_reviews
TBLPROPERTIES (
  "quality" = "bronze",
  "domain" = "orders",
  "subdomain" = "reviews",
  "source_system" = "csv",
  "layer" = "raw_ingestion",
  "update_frequency" = "daily"
)
AS
SELECT
  *,
  _metadata.file_path AS _source_file,
  _metadata.file_modification_time AS _ingestion_time,
  current_timestamp() AS _processed_at
FROM read_files(
  '/Volumes/catalog-impacta-capstone/default/capstone/olist_order_reviews_dataset.csv',
  format => 'csv',
  header => true,
  inferSchema => true,
  mode => 'PERMISSIVE',
  charset => 'UTF-8'
);