-- ============================================================================
-- BRONZE LAYER - SELLERS DOMAIN
-- Arquivo: bronze/04_sellers.sql
-- Tabelas: sellers
-- ============================================================================

-- ============================================================================
-- BRONZE: SELLERS
-- ============================================================================

CREATE MATERIALIZED VIEW bronze.sellers
TBLPROPERTIES (
  "quality" = "bronze",
  "domain" = "sellers",
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
  '/Volumes/catalog-impacta-capstone/default/capstone/olist_sellers_dataset.csv',
  format => 'csv',
  header => true,
  inferSchema => true,
  mode => 'PERMISSIVE',
  charset => 'UTF-8'
);