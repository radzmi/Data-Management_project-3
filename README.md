# Data Management: Project 3 (Classification of Iris dataset using Spark)

# About
The dataset was introduced by the British statistician and biologist Ronald Fisher in his 1936 paper. He collect the data to use multiple measurements in his taxonomy studies. The dataset are widely used in learning classifications technique in machine learning as it provides three different flower class. The dataset contains a set of 150 records under 5 attributes - Petal Length, Petal Width, Sepal Length, Sepal width and Class(Species).

# Steps
The classifications are seperated by three different steps and recorded in three different .py file

## Load the dataset
The dataset are downloaded from kaggle.com in .csv format. Then the dataset were uploaded to putty using pscp method. Then the first script called "iris_rf.py" were created.

## Random forest
In the script "iris_rf.py" the dataset were loaded using scikit-learn package. Then, we created pandas dataframe and add scheme. Scheme were created as Spark requires a defined schema to understand the structure of the data. By creating a schema explicitly, you ensure that Spark handles the data correctly. Scaling the features ensures that each feature contributes equally to the distance calculations in algorithms. Next, the dataset set were split into 80% training set and 20% testing set. After splitting the data, random forest model are performed on the training data. k-fold cross-validation were perfomed to evaluate the model performance.

<img width="371" alt="spark1" src="https://github.com/radzmi/Data-Management_project-3/assets/152348714/a4456f62-ec60-4f12-8436-799086257651">

The results shows that the best parameters for randomforest is 20 for number of tress and 5 as depth. 




























iris_rf.py
iris_lr.py
iris_spark.py
