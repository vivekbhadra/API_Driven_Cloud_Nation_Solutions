#!/usr/bin/env python3
# ------------------------------------------------------
# File: glue_script_hyper_params_tuning.py
# Purpose: Performs hyperparameter tuning for demand forecasting
# using AWS Glue and PySpark. This script builds upon the
# feature engineering and modeling stages by optimizing
# hyperparameters for the Random Forest model.
# ------------------------------------------------------

import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
from datetime import datetime
import itertools

# ------------------------------------------------------
# Glue Job Setup
# ------------------------------------------------------
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# ------------------------------------------------------
# Step 1: Load Feature-Engineered Dataset
# ------------------------------------------------------
input_path = "s3://scdf-project-data-mode-1/features/"
df = spark.read.parquet(input_path)
print("Feature dataset loaded successfully. Total rows:", df.count())

# ------------------------------------------------------
# Step 2: Sample Data for Hyperparameter Tuning
# ------------------------------------------------------
sample_fraction = 0.3  # 30% of data
train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

train_df = train_df.sample(fraction=sample_fraction, seed=42)
print(f"Training data sampled: {train_df.count()} rows")

# ------------------------------------------------------
# Step 3: Prepare Features
# ------------------------------------------------------
feature_cols = ["store", "item", "year", "month", "day_of_week", "lag_1", "lag_7", "rolling_avg_7"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
train_df = assembler.transform(train_df).select("features", col("sales").alias("label"))
test_df = assembler.transform(test_df).select("features", col("sales").alias("label"))

print("Feature vectors created.")

# ------------------------------------------------------
# Step 4: Define Evaluator
# ------------------------------------------------------
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")

# ------------------------------------------------------
# Step 5: Linear Regression Baseline
# ------------------------------------------------------
lr = LinearRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train_df)
pred_lr = lr_model.transform(test_df)
rmse_lr = evaluator.evaluate(pred_lr)
print(f"Linear Regression RMSE (sampled data): {rmse_lr:.4f}")

# ------------------------------------------------------
# Step 6: Random Forest with Optimized Grid + Incremental Saving
# ------------------------------------------------------
rf_param_grid = {
    "numTrees": [50, 100],   # small grid for speed
    "maxDepth": [8, 12],
    "maxBins": [32]
}

output_base_path = "s3://scdf-project-data-mode-1/models/incremental_rf/"

# Iterate over all hyperparameter combinations
for numTrees, maxDepth, maxBins in itertools.product(
        rf_param_grid["numTrees"],
        rf_param_grid["maxDepth"],
        rf_param_grid["maxBins"]):

    print(f"Training RF: numTrees={numTrees}, maxDepth={maxDepth}, maxBins={maxBins}")

    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="label",
        numTrees=numTrees,
        maxDepth=maxDepth,
        maxBins=maxBins,
        seed=42
    )

    try:
        rf_model = rf.fit(train_df)
        predictions = rf_model.transform(test_df)
        rmse = evaluator.evaluate(predictions)

        print(f"Completed RF model => RMSE: {rmse:.4f}")

        # Save model immediately
        model_path = f"{output_base_path}rf_trees{numTrees}_depth{maxDepth}_bins{maxBins}"
        rf_model.write().overwrite().save(model_path)
        print(f"Model saved to: {model_path}")

    except Exception as e:
        print(f"Error training RF with numTrees={numTrees}, maxDepth={maxDepth}, maxBins={maxBins}: {e}")

# ------------------------------------------------------
# Step 7: Commit Job
# ------------------------------------------------------
job.commit()
print("Glue ML job completed successfully at:", datetime.now())