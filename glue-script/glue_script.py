import sys                                        # Provides access to system-specific parameters and command-line arguments
from awsglue.transforms import *                  # Contains AWS Glue transform classes used for ETL operations (e.g., mapping, resolving choice)
from awsglue.utils import getResolvedOptions      # Parses and retrieves job parameters (like JOB_NAME) passed to the Glue job
from pyspark.context import SparkContext          # Entry point for Spark; manages the connection to the Spark execution cluster
from awsglue.context import GlueContext           # Glue wrapper around SparkContext providing AWS Glue–specific features
from awsglue.job import Job                       # Used to define, initialise, and commit AWS Glue jobs
from pyspark.sql.functions import col             # Provides DataFrame column functions used in transformations and filters
from datetime import datetime                     # Enables handling of date and time (useful for timestamps, logging, or file naming)

# --------------------------------------------
# AWS Glue Job Setup
# --------------------------------------------

# Retrieve the job name passed as a command-line argument from AWS Glue
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

# Initialise a SparkContext — the entry point for Spark execution
sc = SparkContext()

# Create a GlueContext, which extends SparkContext with AWS Glue–specific functionality
glueContext = GlueContext(sc)

# Get the SparkSession from GlueContext to use standard PySpark APIs
spark = glueContext.spark_session

# Define and initialise the Glue job for tracking and logging purposes
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# --------------------------------------------
# Step 1: Read raw data from S3
# --------------------------------------------

# Define the S3 path where the raw input CSV file is stored
input_path = "s3://scdf-project-data/raw/base_sales.csv"

# Read the raw CSV file into a Spark DataFrame with headers as column names
df = spark.read.option("header", "true").csv(input_path)

# This step ingests raw data from Amazon S3 into a Spark DataFrame, which allows distributed
# processing and transformation. Reading with 'header=true' ensures that the first row of the
# CSV file is treated as column names, simplifying subsequent operations.

# --------------------------------------------
# Optional Data Cleaning
# --------------------------------------------

# Drop rows containing null values to ensure data quality
# (You can extend this step with type casting, trimming, or filtering logic as needed)
df_cleaned = df.dropna()

# Data cleaning improves quality and consistency before further processing.
# Dropping rows with missing values prevents errors or skewed results in downstream analytics
# or machine learning tasks. In practice, this step can be extended to include deduplication,
# outlier removal, or data type corrections depending on the dataset.

# --------------------------------------------
# Step 2: Write cleaned data to 'processed/' as Parquet
# --------------------------------------------

# Define the S3 destination path for storing cleaned, processed data
processed_path = "s3://scdf-project-data/processed/"

# Write the cleaned DataFrame in Parquet format — a compressed, columnar storage format
# 'overwrite' mode ensures the existing data is replaced with the latest output
df_cleaned.write.mode("overwrite").parquet(processed_path)

# The Parquet format is chosen because it provides efficient compression and encoding,
# making it faster to query and more cost-effective to store than CSV.
# Writing to the 'processed' zone establishes a clean, structured dataset layer
# that can be reused across multiple analytics or training jobs.

# --------------------------------------------
# Step 3: Split data into training and testing subsets
# --------------------------------------------

# Randomly split the cleaned dataset into 80% training and 20% testing portions
# The fixed random seed (42) ensures consistent, reproducible splits across multiple runs
#
# In any data processing or machine learning workflow, it’s important to evaluate model performance
# on data that was not used during training. Splitting the dataset ensures that:
# - The model learns from one portion (training data) while being tested on unseen data (test data).
# - This separation prevents overfitting, helping measure true generalisation capability.
# - Using a fixed seed allows the same random split every time the job runs, ensuring reproducibility.
#   This is especially critical in automated pipelines where results must remain consistent
#   across development, testing, and production environments.
train_df, test_df = df_cleaned.randomSplit([0.8, 0.2], seed=42)

# --------------------------------------------
# Step 4: Write train and test datasets back to S3
# --------------------------------------------

# Define the S3 base path for saving the training and test data
output_prefix = "s3://scdf-project-data/training/"

# Write the training DataFrame as a CSV file (with header row) to S3
train_df.write.mode("overwrite").option("header", "true").csv(output_prefix + "train.csv")

# Write the testing DataFrame similarly for model validation or evaluation
test_df.write.mode("overwrite").option("header", "true").csv(output_prefix + "test.csv")

# Separating and saving training and test data ensures modularity — allowing machine learning
# pipelines or external analytics tools to directly consume data from distinct S3 paths.
# Writing as CSV (rather than Parquet) is useful when the downstream consumer expects text-based input.

# --------------------------------------------
# Final Step: Commit the Glue Job
# --------------------------------------------

# Commit the job to signal successful completion (required by AWS Glue for job tracking)
job.commit()

# The commit operation formally ends the Glue job, marking it as successful.
# This ensures AWS Glue can manage job states properly and helps trigger dependent jobs
# or notifications within automated ETL pipelines.

