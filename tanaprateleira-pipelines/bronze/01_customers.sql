-- ============================================================================
-- BRONZE LAYER - CUSTOMERS DOMAIN
-- Arquivo: bronze/01_customers.sql
-- Tabelas: olist_customers + olist_geolocation
-- ============================================================================

-- ============================================================================
-- 1. BRONZE: CUSTOMERS
-- ============================================================================

CREATE MATERIALIZED VIEW bronze.olist_customers
TBLPROPERTIES (
  "quality" = "bronze",
  "domain" = "customers",
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
  '/Volumes/catalog-impacta-capstone/default/capstone/olist_customers_dataset.csv',
  format => 'csv',
  header => true,
  inferSchema => true,
  mode => 'PERMISSIVE',
  charset => 'UTF-8'
);

-- ============================================================================
-- 2. BRONZE: GEOLOCATION
-- ============================================================================

CREATE MATERIALIZED VIEW bronze.olist_geolocation
TBLPROPERTIES (
  "quality" = "bronze",
  "domain" = "customers",
  "source_system" = "csv",
  "subdomain" = "geolocation",
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
  '/Volumes/catalog-impacta-capstone/default/capstone/olist_geolocation_dataset.csv',
  format => 'csv',
  header => true,
  inferSchema => true,
  mode => 'PERMISSIVE',
  charset => 'UTF-8'
);