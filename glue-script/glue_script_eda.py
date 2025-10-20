#!/usr/bin/env python3
"""
glue_script_eda.py
Performs Exploratory Data Analysis (EDA) on the preprocessed sales dataset stored in Amazon S3.
This job runs as an AWS Glue Job in script mode, producing summary statistics,
aggregations, and correlations, and persisting selected CSV results to S3.
"""

import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, year, month, dayofweek, avg, sum as _sum
from datetime import datetime

# --------------------------------------------
# AWS Glue Job Setup
# --------------------------------------------
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

print("---- Starting Exploratory Data Analysis (EDA) ----")

# --------------------------------------------
# Step 1: Load preprocessed dataset
# --------------------------------------------
input_path = "s3://scdf-project-data/processed/"
print(f"Loading preprocessed dataset from: {input_path}")

df = spark.read.parquet(input_path)

print("---- Schema Verification ----")
df.printSchema()

print("---- Sample Records ----")
df.show(5)

# --------------------------------------------
# Step 2: Descriptive Statistics
# --------------------------------------------
print("---- Descriptive Statistics for 'sales' ----")
df.describe(["sales"]).show()

# --------------------------------------------
# Step 3: Temporal Feature Engineering and Aggregation
# --------------------------------------------
print("---- Adding Temporal Columns (year, month, dayofweek) ----")
df = (
    df.withColumn("year", year(col("date")))
      .withColumn("month", month(col("date")))
      .withColumn("dayofweek", dayofweek(col("date")))
)

print("---- Monthly Sales Aggregation ----")
monthly_sales = (
    df.groupBy("year", "month")
      .agg(_sum("sales").alias("total_sales"))
      .orderBy("year", "month")
)
monthly_sales.show(10)

# --------------------------------------------
# Step 4: Store-Level Analysis
# --------------------------------------------
print("---- Store-Level Average Sales ----")
store_avg = (
    df.groupBy("store")
      .agg(avg("sales").alias("avg_sales"))
      .orderBy(col("avg_sales").desc())
)
store_avg.show(10)

# --------------------------------------------
# Step 5: Item-Level Analysis
# --------------------------------------------
print("---- Item-Level Total Sales ----")
item_sales = (
    df.groupBy("item")
      .agg(_sum("sales").alias("total_sales"))
      .orderBy(col("total_sales").desc())
)
item_sales.show(10)

# --------------------------------------------
# Step 6: Missing Value Check
# --------------------------------------------
print("---- Missing Value Check ----")
missing_info = {c: df.filter(col(c).isNull()).count() for c in df.columns}
print("Missing Values Summary:")
print(missing_info)

# --------------------------------------------
# Step 7: Correlation Analysis
# --------------------------------------------
print("---- Correlation Analysis ----")
corr_store_item = df.stat.corr("store", "item")
corr_store_sales = df.stat.corr("store", "sales")
corr_item_sales = df.stat.corr("item", "sales")

print("Correlation between store and item:", corr_store_item)
print("Correlation between store and sales:", corr_store_sales)
print("Correlation between item and sales:", corr_item_sales)

# --------------------------------------------
# Step 8: Persist Analytical Outputs to S3
# --------------------------------------------
eda_output_path = "s3://scdf-project-data/eda/"
print("---- Writing EDA Outputs to S3 ----")

monthly_sales.write.mode("overwrite").option("header", "true").csv(eda_output_path + "monthly_sales/")
store_avg.write.mode("overwrite").option("header", "true").csv(eda_output_path + "store_avg/")
item_sales.write.mode("overwrite").option("header", "true").csv(eda_output_path + "item_sales/")

print(f"EDA summary outputs written to: {eda_output_path}")

# --------------------------------------------
# Final Step: Commit the Glue Job
# --------------------------------------------
job.commit()
print("EDA job completed successfully at:", datetime.now())

