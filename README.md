# Data-Management_project-3

# Importing all package required
import numpy as np #Numerical computing
import pandas as pd # Data manipulation and analysis

import os # Interacting with the operating system
import urllib #  Fetching data from the web
import sys # Interacting with the Python runtime environment.

import pyspark # Distributed data processing with Spark
from pyspark.sql.functions import *  # DataFrame manipulation functions in Spark
from pyspark.ml.classification import * # Machine learning classification models in Spark
from pyspark.ml.evaluation import * # Evaluation metrics for machine learning models in Spark
from pyspark.ml.feature import * # Feature extraction, transformation, and selection tools in Spark


# start Spark session
spark = pyspark.sql.SparkSession.builder.appName('Iris').getOrCreate()

# print runtime versions
print ('****************')
print ('Python version: {}'.format(sys.version))
print ('Spark version: {}'.format(spark.version))
print ('****************')
