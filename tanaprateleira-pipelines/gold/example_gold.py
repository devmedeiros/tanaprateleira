# Databricks notebook source
# MAGIC %md
# MAGIC > Abaixo deixo exemplos de como pode ser usados as tabelas com python na camada gold.

# COMMAND ----------

display(spark.sql("SHOW TABLES IN `catalog-impacta-capstone`.silver"))

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

orders_path = "`catalog-impacta-capstone`.silver.orders"
orders_df = spark.read.table(orders_path)

# COMMAND ----------

orders_df.groupBy("order_status").count().display()

# COMMAND ----------

orders_df = spark.read.table("`catalog-impacta-capstone`.silver.orders")
order_items_df = spark.read.table("`catalog-impacta-capstone`.silver.order_items")
products_df = spark.read.table("`catalog-impacta-capstone`.silver.products")

filtered_orders_df = orders_df.filter(
    (F.col("order_status").isin(["unavailable", "canceled"]) == False) &
    (F.col("order_approved_at").isNotNull())
)

joined_df = filtered_orders_df.join(
    order_items_df, "order_id", "inner"
).join(
    products_df, "product_id", "inner"
)

agg_df = joined_df.groupBy(
    "order_id",
    "order_approved_at",
    "product_id",
    "product_category_name",
    "product_name_lenght",
    "product_description_lenght",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm"
).agg(
    F.max("order_item_id").alias("quantity"),
    F.sum("price").alias("total_price"),
    F.sum("freight_value").alias("total_freight_value")
)

agg_df.display()

# agg_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("workspace.gold.product_demand_features_daily")
