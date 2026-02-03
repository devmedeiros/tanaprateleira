# Databricks notebook source
# MAGIC %md
# MAGIC Teste

# COMMAND ----------

display(spark.sql("SHOW TABLES IN `catalog-impacta-capstone`.gold"))

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

inventory_path = "`catalog-impacta-capstone`.gold.inventory_daily_sku1"
inventory_df = spark.read.table(inventory_path)
inventory_df.display()

# COMMAND ----------

inventory_df.select('event_date', 'inventory_eod', 'approved_qty').display()

# COMMAND ----------

pdf = (
    inventory_df\
      .select(F.col("event_date").cast("timestamp").alias("ds"),
              F.col("approved_qty").cast("double").alias("y"))
      .orderBy("ds")
).toPandas()

# COMMAND ----------

# MAGIC %pip install pmdarima

# COMMAND ----------

from pmdarima import auto_arima

model = auto_arima(
    pdf["y"],
    seasonal=True,
    m=12,                 # ajuste conforme a sazonalidade
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore"
)

# COMMAND ----------

fcst = model.predict(n_periods=28)

# COMMAND ----------

fcst

# COMMAND ----------

import pandas as pd

# COMMAND ----------

last_ds = (
    inventory_df
    .select(F.max(F.col("event_date")).alias("max_event_date"))
    .first()["max_event_date"]
)

# COMMAND ----------

start = pd.to_datetime(last_ds).normalize() + pd.Timedelta(days=1)
ds = pd.date_range(start=start, periods=28, freq="D")

# COMMAND ----------

pred_df = pd.DataFrame({"event_date": ds, "demand_prediction": fcst})

# COMMAND ----------

sdf = spark.createDataFrame(pred_df)
sdf.write.mode("overwrite").saveAsTable("`catalog-impacta-capstone`.gold.demand_prediction_28days")
