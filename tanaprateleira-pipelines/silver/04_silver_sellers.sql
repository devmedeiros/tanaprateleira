-- ============================================================================
-- SILVER LAYER - SELLERS DOMAIN
-- Arquivo: silver/04_sellers.sql
-- Responsabilidade: LIMPEZA APENAS
-- ============================================================================

CREATE MATERIALIZED VIEW silver.sellers
TBLPROPERTIES (
  "quality" = "silver",
  "domain" = "sellers",
  "type" = "entity",
  "grain" = "seller_id",
  "description" = "Vendedores limpos e validados"
)
AS
SELECT
  seller_id,
  seller_zip_code_prefix,
  LOWER(TRIM(seller_city)) AS seller_city,
  UPPER(TRIM(seller_state)) AS seller_state,
  CASE 
    WHEN seller_id IS NULL THEN 'invalid_null_id'
    WHEN seller_state NOT IN ('AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO') 
      THEN 'invalid_state'
    WHEN LENGTH(TRIM(seller_zip_code_prefix)) < 4 
      THEN 'invalid_zipcode'
    ELSE 'valid'
  END AS data_quality_flag,
  _source_file,
  _ingestion_time,
  _processed_at
FROM bronze.sellers
WHERE
  seller_id IS NOT NULL;