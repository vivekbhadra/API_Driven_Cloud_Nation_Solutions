import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, mean, udf, year, month, dayofweek, avg, sum as _sum
from pyspark.sql.types import FloatType
from datetime import datetime
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline

# --------------------------------------------
# AWS Glue Job Setup
# --------------------------------------------
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# --------------------------------------------
# Step 1: Read raw data from S3
# --------------------------------------------
input_path = "s3://scdf-project-data/raw/base_sales.csv"
df = spark.read.option("header", "true").csv(input_path)

print("---- Column Data Types ----")
for name, dtype in df.dtypes:
    print(f"{name}: {dtype}")

df = (
    df.withColumn("store", col("store").cast("int"))
      .withColumn("item", col("item").cast("int"))
      .withColumn("sales", col("sales").cast("float"))
      .withColumn("date", col("date").cast("date"))
)

print("---- Summary Statistics ----")
df.describe(["sales", "store", "item"]).show()

# --------------------------------------------
# Optional Data Cleaning (with Imputation)
# --------------------------------------------
print("---- Missing Value Check ----")
missing_info = {colname: df.filter(col(colname).isNull()).count() for colname in df.columns}
print(missing_info)

mean_sales = df.select(mean("sales").alias("mean_sales")).collect()[0]["mean_sales"]
df_cleaned = df.fillna({"sales": mean_sales})

# --------------------------------------------
# Optional Normalisation of Sales
# --------------------------------------------
assembler = VectorAssembler(inputCols=["sales"], outputCol="sales_vector")
scaler = MinMaxScaler(inputCol="sales_vector", outputCol="sales_scaled")
pipeline = Pipeline(stages=[assembler, scaler])
scaler_model = pipeline.fit(df_cleaned)
df_scaled = scaler_model.transform(df_cleaned)

# Convert vector type to scalar float
vector_to_float = udf(lambda vec: float(vec[0]), FloatType())
df_final = (
    df_scaled.withColumn("sales_scaled_value", vector_to_float(col("sales_scaled")))
             .drop("sales_vector", "sales_scaled")
)

print("Row count after cleaning and normalisation:", df_final.count())

# --------------------------------------------
# Step 2: Write cleaned data to 'processed/' as Parquet
# --------------------------------------------
processed_path = "s3://scdf-project-data/processed/"
df_final.write.mode("overwrite").parquet(processed_path)
print("Processed data written to:", processed_path)

# --------------------------------------------
# Step 3: Split data into training and testing subsets
# --------------------------------------------
train_df, test_df = df_final.randomSplit([0.8, 0.2], seed=42)

# --------------------------------------------
# Step 4: Write train and test datasets back to S3 (CSV-safe columns only)
# --------------------------------------------
output_prefix = "s3://scdf-project-data/training/"
train_df.write.mode("overwrite").option("header", "true").csv(output_prefix + "train.csv")
test_df.write.mode("overwrite").option("header", "true").csv(output_prefix + "test.csv")
print("Train/Test split written to:", output_prefix)

# --------------------------------------------
# Step 5: Generate EDA summary outputs for visualisation
# --------------------------------------------
print("---- Generating EDA Summary Outputs ----")

# Add temporal columns
df_final = df_final.withColumn("year", year(col("date"))).withColumn("month", month(col("date"))).withColumn("dayofweek", dayofweek(col("date")))

# Monthly sales summary
monthly_sales = (
    df_final.groupBy("year", "month")
    .agg(_sum("sales").alias("total_sales"))
    .orderBy("year", "month")
)

# Store-level average sales
store_avg = (
    df_final.groupBy("store")
    .agg(avg("sales").alias("avg_sales"))
    .orderBy(col("avg_sales").desc())
)

# Item-level total sales
item_sales = (
    df_final.groupBy("item")
    .agg(_sum("sales").alias("total_sales"))
    .orderBy(col("total_sales").desc())
)

# Write EDA summaries to S3
eda_output_path = "s3://scdf-project-data/eda/"
monthly_sales.write.mode("overwrite").option("header", "true").csv(eda_output_path + "monthly_sales/")
store_avg.write.mode("overwrite").option("header", "true").csv(eda_output_path + "store_avg/")
item_sales.write.mode("overwrite").option("header", "true").csv(eda_output_path + "item_sales/")

print("EDA summary outputs written to:", eda_output_path)

# --------------------------------------------
# Final Step: Commit the Glue Job
# --------------------------------------------
job.commit()
print("Glue job completed successfully at:", datetime.now())

