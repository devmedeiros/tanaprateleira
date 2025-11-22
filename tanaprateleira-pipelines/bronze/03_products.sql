-- ============================================================================
-- BRONZE LAYER - PRODUCTS DOMAIN
-- Arquivo: bronze/03_products.sql
-- Tabelas: products + product_categories
-- ============================================================================

-- ============================================================================
-- 1. BRONZE: PRODUCTS
-- ============================================================================

CREATE MATERIALIZED VIEW bronze.products
TBLPROPERTIES (
  "quality" = "bronze",
  "domain" = "products",
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
  '/Volumes/catalog-impacta-capstone/default/capstone/olist_products_dataset.csv',
  format => 'csv',
  header => true,
  inferSchema => true,
  mode => 'PERMISSIVE',
  charset => 'UTF-8'
);

-- ============================================================================
-- 2. BRONZE: PRODUCT CATEGORIES
-- ============================================================================

CREATE MATERIALIZED VIEW bronze.product_category_name_translation
TBLPROPERTIES (
  "quality" = "bronze",
  "domain" = "products",
  "subdomain" = "categories",
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
  '/Volumes/catalog-impacta-capstone/default/capstone/product_category_name_translation.csv',
  format => 'csv',
  header => true,
  inferSchema => true,
  mode => 'PERMISSIVE',
  charset => 'UTF-8'
);