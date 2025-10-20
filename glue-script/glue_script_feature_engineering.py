#!/usr/bin/env python3
# ------------------------------------------------------
# File: glue_script_feature_engineering.py
# Purpose: Performs feature engineering for demand forecasting
# using AWS Glue and PySpark. This script extends the preprocessing
# and EDA stages by creating lag and rolling average features.
# ------------------------------------------------------

import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, year, month, dayofweek, lag, avg
from pyspark.sql.window import Window
from datetime import datetime

# ------------------------------------------------------
# AWS Glue Job Setup
# ------------------------------------------------------
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# ------------------------------------------------------
# Step 1: Read Processed Data
# ------------------------------------------------------
input_path = "s3://scdf-project-data/processed/"
df = spark.read.parquet(input_path)
print("Processed dataset loaded successfully.")

# ------------------------------------------------------
# Step 2: Create Time-Based Features
# ------------------------------------------------------
df = df.withColumn("year", year(col("date"))) \
       .withColumn("month", month(col("date"))) \
       .withColumn("day_of_week", dayofweek(col("date")))

print("Temporal features (year, month, day_of_week) created.")

# ------------------------------------------------------
# Step 3: Create Lag and Rolling Average Features
# ------------------------------------------------------
window_spec = Window.partitionBy("store", "item").orderBy("date")

df = df.withColumn("lag_1", lag("sales", 1).over(window_spec))
df = df.withColumn("lag_7", lag("sales", 7).over(window_spec))
df = df.withColumn("rolling_avg_7", avg("sales").over(window_spec.rowsBetween(-6, 0)))

print("Lag and rolling average features created.")

# ------------------------------------------------------
# Step 4: Handle Missing Values from Lag Features
# ------------------------------------------------------
df = df.na.fill(0, subset=["lag_1", "lag_7", "rolling_avg_7"])
print("Missing values in lag features imputed with zeros.")

# ------------------------------------------------------
# Step 5: Write Feature-Engineered Data to S3
# ------------------------------------------------------
output_path = "s3://scdf-project-data/features/"
df.write.mode("overwrite").parquet(output_path)
print("Feature-engineered dataset written to:", output_path)

# ------------------------------------------------------
# Final Step: Commit Job
# ------------------------------------------------------
job.commit()
print("Feature engineering job completed successfully at:", datetime.now())

