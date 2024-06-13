from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import confusion_matrix, classification_report

# Initialize Spark session
spark = SparkSession.builder.appName('Iris').getOrCreate()

# Load Iris dataset using scikit-learn
iris = load_iris()
cols = [i.replace('(cm)', '').strip().replace(' ', '_') for i in iris.feature_names] + ['label']  # Column name cleanup

# Create a Pandas DataFrame
pdf = pd.DataFrame(np.c_[iris.data, iris.target], columns=cols)

# Define the schema
schema = StructType([
    StructField('sepal_length', DoubleType(), True),
    StructField('sepal_width', DoubleType(), True),
    StructField('petal_length', DoubleType(), True),
    StructField('petal_width', DoubleType(), True),
    StructField('label', DoubleType(), True)
])

# Convert the Pandas DataFrame to a Spark DataFrame with the defined schema
df = spark.createDataFrame(pdf, schema=schema)

# Define the feature columns
feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Vectorize all numerical columns into a single feature column
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
data = assembler.transform(df)

# Scale the features
scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures')
scaler_model = scaler.fit(data)
data = scaler_model.transform(data)

# Split the data into training and test sets with an 80/20 split
train, test = data.randomSplit([0.8, 0.2], seed=1234)

# Define the RandomForestClassifier with the specified hyperparameters
rf = RandomForestClassifier(featuresCol='scaledFeatures', labelCol='label', numTrees=20, maxDepth=5)

# Train the model
rf_model = rf.fit(train)

# Make predictions on the test set
predictions = rf_model.transform(test)

# Convert predictions to Pandas DataFrame for comparison
predictions_pd = predictions.select('label', 'prediction').toPandas()

# Generate confusion matrix and classification report
target_names = iris.target_names
conf_matrix = confusion_matrix(predictions_pd['label'], predictions_pd['prediction'])
class_report = classification_report(predictions_pd['label'], predictions_pd['prediction'], target_names=target_names)

print('Confusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(class_report)

# Evaluate the model with various metrics
evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction')

accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

print('Test Accuracy: {:.4f}'.format(accuracy))
print('Test Precision: {:.4f}'.format(precision))
print('Test Recall: {:.4f}'.format(recall))
print('Test F1 Score: {:.4f}'.format(f1))

# Show some predictions
predictions.select('features', 'scaledFeatures', 'label', 'prediction').show(10)

# Stop the session
spark.stop()

