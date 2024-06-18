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

## Logistic regression
The second script named "iris_lr.py" is a script were logistic regression were performed on the dataset. Logistic regression were performed in order to compare the model performance with random forest model. Both model have their own strenght and weaknesses. Thus, performing both model and comparing their performances will produce a better classifications. The proceduer for logistic regression are the same as random forest, just using logistic regression function instead of random forest.

<img width="353" alt="spark_lr" src="https://github.com/radzmi/Data-Management_project-3/assets/152348714/1b68e233-c090-42e5-abe9-2204400beaeb">


From the output, the logistic regression will perform better when the parameters are set to 'regParam = 0.01' and 'elasticNetParam = 0.5'.

## Comparison

After performing both models and tuning the parameters, the model are performed once again with the parameters seleted before.

<img width="411" alt="spark_compare" src="https://github.com/radzmi/Data-Management_project-3/assets/152348714/1d0d3ccd-ea48-42ae-ab71-0bbd1dba3553">

### Confusion Matirx
The confusion matrix for both model is identical, indicating that both models made the same classifications.

Setosa: 12 instances correctly classified (no misclassifications).
Versicolor: 4 instances correctly classified (no misclassifications).
Virginica: 11 instances correctly classified, 2 misclassified as Versicolor.

### Classification Reports
Both classification reports is also identical, indicating similar performance on this dataset.
   
Precision: Measures the accuracy of the positive predictions.
Recall: Measures the ability to find all the relevant cases.
F1-Score: Harmonic mean of precision and recall, providing a single metric that balances both concerns.

# Conclusion

Identical Performance: Both Random Forest and Logistic Regression models performed identically on this dataset in terms of accuracy, precision, recall, and F1 score.
Detailed Performance: Both models perfectly classified the 'setosa' class. They also performed well on the 'virginica' class, with a slight misclassification for 'virginica' instances as 'versicolor'.
Robustness: The identical performance suggests that both models are robust and well-tuned for this particular dataset.

To identify the best technique for this dataset is depends on a few factors. Logistic Regression are better than Random Forest if you prefered for its simplicity and interpretability, especially in scenarios where a linear relationship is expected and the model needs to be easily explained. Random forest may be preferred for its ability to model complex, non-linear relationships and its robustness to overfitting due to the ensemble approach.
