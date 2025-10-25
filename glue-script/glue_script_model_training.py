import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
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
# Step 1: Load Feature-Engineered Dataset
# ------------------------------------------------------
input_path = "s3://scdf-project-data/features/"
df = spark.read.parquet(input_path)
print("Feature dataset loaded successfully.")

# ------------------------------------------------------
# Step 2: Prepare Data for Model Training
# ------------------------------------------------------
feature_cols = ["store", "item", "year", "month", "day_of_week", "lag_1", "lag_7", "rolling_avg_7"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(df).select("features", col("sales").alias("label"))

train_df, test_df = data.randomSplit([0.7, 0.3], seed=42)
print("Data split into training and testing sets.")

# ------------------------------------------------------
# Step 3: Train Linear Regression and Random Forest Models
# ------------------------------------------------------
lr = LinearRegression(featuresCol="features", labelCol="label")
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="label",
    numTrees=100,
    maxDepth=12,
    maxBins=32,
    seed=42
)

lr_model = lr.fit(train_df)
rf_model = rf.fit(train_df)
print("Both models trained successfully.")

# ------------------------------------------------------
# Step 4: Evaluate Model Performance (RMSE)
# ------------------------------------------------------
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")

for name, model in [("Linear Regression", lr_model), ("Random Forest", rf_model)]:
    predictions = model.transform(test_df)
    rmse = evaluator.evaluate(predictions)
    print(f"{name} RMSE: {rmse}")

# ------------------------------------------------------
# Step 5: Generate Predictions using the Random Forest Model
# ------------------------------------------------------
rf_predictions = rf_model.transform(test_df) \
    .withColumnRenamed("label", "actual_sales") \
    .withColumnRenamed("prediction", "predicted_sales")

# Save predictions to S3
predictions_output_path = "s3://scdf-project-data/predictions/"
rf_predictions.write.mode("overwrite").parquet(predictions_output_path)
print("Predictions written to:", predictions_output_path)

# Display a few sample predictions in CloudWatch
print("Sample predictions (top 5 rows):")
for row in rf_predictions.limit(5).collect():
    print(row)

# ------------------------------------------------------
# Step 6: Save Models to S3
# ------------------------------------------------------
output_path = "s3://scdf-project-data/models/"
lr_model.write().overwrite().save(output_path + "linear_regression_model")
rf_model.write().overwrite().save(output_path + "random_forest_model")

print("Models saved to:", output_path)

# ------------------------------------------------------
# Final Step: Commit Job
# ------------------------------------------------------
job.commit()
print("Machine Learning training and prediction job completed successfully at:", datetime.now())

