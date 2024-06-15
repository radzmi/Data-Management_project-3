from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Initialize Spark session
spark = SparkSession.builder.appName('Iris').getOrCreate()

# Load Iris dataset using scikit-learn
data = load_iris()
cols = [i.replace('(cm)','').strip().replace(' ','_') for i in data.feature_names] + ['label']

# Create a Pandas DataFrame
pdf = pd.DataFrame(np.c_[data.data, data.target], columns=cols)

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

assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
data = assembler.transform(df)

scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures')
scaler_model = scaler.fit(data)
data = scaler_model.transform(data)

# Split the data into training and test sets with an 80/20 split
train, test = data.randomSplit([0.80, 0.20], seed=1234)

# Define the RandomForestClassifier with a parameter grid
rf = RandomForestClassifier(featuresCol='scaledFeatures', labelCol='label')

paramGrid_rf = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20, 50]) \
    .addGrid(rf.maxDepth, [5, 10, 20]) \
    .build()

crossval_rf = CrossValidator(estimator=rf,
                             estimatorParamMaps=paramGrid_rf,
                             evaluator=MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy'),
                             numFolds=5)

# Train the RandomForest model using cross-validation
cvModel_rf = crossval_rf.fit(train)

# Make predictions on the test set
rf_predictions = cvModel_rf.transform(test)

# Evaluate the RandomForest model
evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
rf_accuracy = evaluator.evaluate(rf_predictions)
print('RandomForest Test Accuracy: {:.4f}'.format(rf_accuracy))

rf_predictions.select('features', 'scaledFeatures', 'label', 'prediction').show(10)

# Show the best RandomForest model's hyperparameters
bestModel_rf = cvModel_rf.bestModel
print('Best RandomForest Model Parameters:')
print(' - numTrees: {}'.format(bestModel_rf.getNumTrees))
print(' - maxDepth: {}'.format(bestModel_rf.getOrDefault('maxDepth')))

# Stop the session
spark.stop()