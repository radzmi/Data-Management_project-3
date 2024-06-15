from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
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

# Define the LogisticRegression with a parameter grid
lr = LogisticRegression(featuresCol='scaledFeatures', labelCol='label')

paramGrid_lr = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

crossval_lr = CrossValidator(estimator=lr,
                             estimatorParamMaps=paramGrid_lr,
                             evaluator=MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy'),
                             numFolds=5)

# Train the LogisticRegression model using cross-validation
cvModel_lr = crossval_lr.fit(train)

# Make predictions on the test set
lr_predictions = cvModel_lr.transform(test)

# Evaluate the LogisticRegression model
evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
lr_accuracy = evaluator.evaluate(lr_predictions)
print('LogisticRegression Test Accuracy: {:.4f}'.format(lr_accuracy))

lr_predictions.select('features', 'scaledFeatures', 'label', 'prediction').show(10)

# Show the best LogisticRegression model's hyperparameters
bestModel_lr = cvModel_lr.bestModel
print('Best LogisticRegression Model Parameters:')
print(' - regParam: {}'.format(bestModel_lr.getOrDefault('regParam')))
print(' - elasticNetParam: {}'.format(bestModel_lr.getOrDefault('elasticNetParam')))

# Stop the session
spark.stop()







