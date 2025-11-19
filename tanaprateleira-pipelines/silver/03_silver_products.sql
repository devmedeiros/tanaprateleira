-- ============================================================================
-- SILVER LAYER - PRODUCTS DOMAIN
-- Arquivo: silver/03_products.sql
-- Responsabilidade: LIMPEZA APENAS (sem joins com categorias)
-- ============================================================================

-- ============================================================================
-- 1. SILVER: PRODUCTS (limpo)
-- ============================================================================

CREATE MATERIALIZED VIEW silver.products
TBLPROPERTIES (
  "quality" = "silver",
  "domain" = "products",
  "type" = "entity",
  "grain" = "product_id",
  "description" = "Produtos limpos e validados"
)
AS
SELECT
  product_id,
  LOWER(TRIM(product_category_name)) AS product_category_name,
  COALESCE(product_name_lenght, 0) AS product_name_lenght,
  COALESCE(product_description_lenght, 0) AS product_description_lenght,
  COALESCE(product_photos_qty, 0) AS product_photos_qty,
  COALESCE(product_weight_g, 0) AS product_weight_g,
  COALESCE(product_length_cm, 0) AS product_length_cm,
  COALESCE(product_height_cm, 0) AS product_height_cm,
  COALESCE(product_width_cm, 0) AS product_width_cm,
  CASE 
    WHEN product_id IS NULL THEN 'invalid_null_id'
    WHEN product_category_name IS NULL OR TRIM(product_category_name) = '' THEN 'missing_category'
    WHEN product_weight_g < 0 THEN 'negative_weight'
    WHEN product_length_cm < 0 THEN 'negative_length'
    WHEN product_height_cm < 0 THEN 'negative_height'
    WHEN product_width_cm < 0 THEN 'negative_width'
    ELSE 'valid'
  END AS data_quality_flag,
  _source_file,
  _ingestion_time,
  _processed_at
FROM bronze.products
WHERE
  product_id IS NOT NULL;

-- ============================================================================
-- 2. SILVER: PRODUCT CATEGORIES (limpo)
-- ============================================================================

CREATE MATERIALIZED VIEW silver.product_category_name_translation
TBLPROPERTIES (
  "quality" = "silver",
  "domain" = "products",
  "subdomain" = "categories",
  "type" = "entity",
  "grain" = "product_category_name",
  "description" = "Mapeamento de categorias de produtos"
)
AS
SELECT
  LOWER(TRIM(product_category_name)) AS product_category_name,
  LOWER(TRIM(product_category_name_english)) AS product_category_name_english,
  CASE 
    WHEN TRIM(product_category_name) = '' THEN 'invalid_empty_name'
    WHEN TRIM(product_category_name_english) = '' THEN 'invalid_empty_english'
    ELSE 'valid'
  END AS data_quality_flag,
  _source_file,
  _ingestion_time,
  _processed_at
FROM bronze.product_category_name_translation
WHERE
  product_category_name IS NOT NULL
  AND product_category_name_english IS NOT NULL;