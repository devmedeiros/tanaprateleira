# Databricks notebook source
# MAGIC %md
# MAGIC > Abaixo deixo exemplos de como pode ser usados as tabelas com python na camada gold.

# COMMAND ----------

display(spark.sql("SHOW TABLES IN `catalog-impacta-capstone`.silver"))

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

orders_path = "`catalog-impacta-capstone`.silver.olist_orders"
orders_df = spark.read.table(orders_path)

# COMMAND ----------

orders_df.display()
