from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import confusion_matrix, classification_report
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

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

# Train the RandomForest model
rf_model = rf.fit(train)

# Make predictions on the test set with RandomForest
rf_predictions = rf_model.transform(test)

# Make predictions on the test set with LogisticRegression
lr_predictions = lr_model.transform(test)

# Create a parameter grid for hyperparameter tuning (fixed parameters)
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [20]) \
    .addGrid(rf.maxDepth, [5]) \
    .build()

# Define the cross-validator
crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy'),
                          numFolds=5)
                          
# Train the model using cross-validation
cvModel = crossval.fit(train)

# Make predictions on the test set
predictions = cvModel.transform(test)


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

# Define LogisticRegression classifier
lr = LogisticRegression(featuresCol='scaledFeatures', labelCol='label')

# Train the LogisticRegression model
lr_model = lr.fit(train)

# Make predictions on the test set with LogisticRegression
lr_predictions = lr_model.transform(test)

# Convert predictions to Pandas DataFrame for comparison
lr_predictions_pd = lr_predictions.select('label', 'prediction').toPandas()

# Generate confusion matrix and classification report for LogisticRegression
lr_conf_matrix = confusion_matrix(lr_predictions_pd['label'], lr_predictions_pd['prediction'])
lr_class_report = classification_report(lr_predictions_pd['label'], lr_predictions_pd['prediction'], target_names=iris.target_names)

print('LogisticRegression Confusion Matrix:')
print(lr_conf_matrix)
print('\nLogisticRegression Classification Report:')
print(lr_class_report)


# Evaluate the RandomForest model with various metrics
evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction')
rf_accuracy = evaluator.evaluate(rf_predictions, {evaluator.metricName: "accuracy"})
rf_precision = evaluator.evaluate(rf_predictions, {evaluator.metricName: "weightedPrecision"})
rf_recall = evaluator.evaluate(rf_predictions, {evaluator.metricName: "weightedRecall"})
rf_f1 = evaluator.evaluate(rf_predictions, {evaluator.metricName: "f1"})

print('RandomForest Test Accuracy: {:.4f}'.format(rf_accuracy))
print('RandomForest Test Precision: {:.4f}'.format(rf_precision))
print('RandomForest Test Recall: {:.4f}'.format(rf_recall))
print('RandomForest Test F1 Score: {:.4f}'.format(rf_f1))

# Evaluate the LogisticRegression model with various metrics
lr_accuracy = evaluator.evaluate(lr_predictions, {evaluator.metricName: "accuracy"})
lr_precision = evaluator.evaluate(lr_predictions, {evaluator.metricName: "weightedPrecision"})
lr_recall = evaluator.evaluate(lr_predictions, {evaluator.metricName: "weightedRecall"})
lr_f1 = evaluator.evaluate(lr_predictions, {evaluator.metricName: "f1"})

print('LogisticRegression Test Accuracy: {:.4f}'.format(lr_accuracy))
print('LogisticRegression Test Precision: {:.4f}'.format(lr_precision))
print('LogisticRegression Test Recall: {:.4f}'.format(lr_recall))
print('LogisticRegression Test F1 Score: {:.4f}'.format(lr_f1))


# Stop the session
spark.stop()

