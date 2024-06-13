from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# Initialize Spark session
spark = SparkSession.builder.appName('Iris').getOrCreate()

# Load Iris dataset using scikit-learn
data = load_iris()
cols = [i.replace('(cm)','').strip().replace(' ','_') for i in data.feature_names] + ['label'] # Column name cleanup

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

# splitting train and test set
train, test = data.randomSplit([0.80, 0.20], seed=1234)

rf = RandomForestClassifier(featuresCol='scaledFeatures', labelCol='label')
rf_model = rf.fit(train)
predictions = rf_model.transform(test)

evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
print('Test Accuracy: {:.4f}'.format(accuracy))

predictions.select('features', 'scaledFeatures', 'label', 'prediction').show(10)

# Stop the session
spark.stop()
:quiei