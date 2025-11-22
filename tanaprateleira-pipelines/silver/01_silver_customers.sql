-- ============================================================================
-- SILVER LAYER - CUSTOMERS DOMAIN
-- Arquivo: silver/01_customers.sql
-- Responsabilidade: LIMPEZA APENAS (trim, lowercase, remove nulos, validação)
-- ============================================================================

-- ============================================================================
-- 1. SILVER: CUSTOMERS (limpo)
-- ============================================================================

CREATE MATERIALIZED VIEW silver.customers
TBLPROPERTIES (
  "quality" = "silver",
  "domain" = "customers",
  "type" = "entity",
  "grain" = "customer_id",
  "description" = "Clientes limpos e validados"
)
AS
SELECT
  customer_id,
  customer_unique_id,
  customer_zip_code_prefix,
  LOWER(TRIM(customer_city)) AS customer_city,
  UPPER(TRIM(customer_state)) AS customer_state,
  -- Validação
  CASE 
    WHEN customer_id IS NULL THEN 'invalid_null_id'
    WHEN customer_state NOT IN ('AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO') 
      THEN 'invalid_state'
    WHEN LENGTH(TRIM(customer_zip_code_prefix)) < 4 
      THEN 'invalid_zipcode'
    ELSE 'valid'
  END AS data_quality_flag,
  _source_file,
  _ingestion_time,
  _processed_at
FROM bronze.customers
WHERE
  -- Remove linhas completamente nulas
  customer_id IS NOT NULL;

-- ============================================================================
-- 2. SILVER: GEOLOCATION (limpo)
-- ============================================================================

CREATE MATERIALIZED VIEW silver.geolocation
TBLPROPERTIES (
  "quality" = "silver",
  "domain" = "customers",
  "subdomain" = "geolocation",
  "type" = "entity",
  "grain" = "geolocation_zip_code_prefix",
  "description" = "Geolocalização limpa e validada"
)
AS
SELECT
  geolocation_zip_code_prefix,
  CAST(geolocation_lat AS DECIMAL(10, 6)) AS geolocation_lat,
  CAST(geolocation_lng AS DECIMAL(10, 6)) AS geolocation_lng,
  LOWER(TRIM(geolocation_city)) AS geolocation_city,
  UPPER(TRIM(geolocation_state)) AS geolocation_state,
  CASE 
    WHEN geolocation_zip_code_prefix IS NULL THEN 'invalid_null_zipcode'
    WHEN geolocation_lat IS NULL OR geolocation_lng IS NULL THEN 'missing_coordinates'
    WHEN geolocation_lat < -33.75 OR geolocation_lat > 5.27 THEN 'invalid_latitude'
    WHEN geolocation_lng < -73.99 OR geolocation_lng > -34.79 THEN 'invalid_longitude'
    ELSE 'valid'
  END AS data_quality_flag,
  _source_file,
  _ingestion_time,
  _processed_at
FROM bronze.geolocation
WHERE
  geolocation_zip_code_prefix IS NOT NULL;