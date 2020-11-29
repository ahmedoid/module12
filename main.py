import sys
import traceback

import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main():
    try:
        # 1 -  Data Loading
        iris = pd.read_csv('iris.csv')
        # 2 - Exploratory Data Analysis
        # Preview of Data
        print(f"First 6 Records\n {iris.head()}")
        # get Information about data
        print(f"info {iris.info()}")
        print(f"Value count for each type :\n{iris['variety'].value_counts()}")
        # visualization data
        sns.pairplot(iris, hue='variety', markers='+')
        plt.show()
        # 3 - Data Cleaning(Preprocessing)
        # label_encoder object knows how to understand word labels.
        label_encoder = preprocessing.LabelEncoder()
        #  Encode labels in column 'variety'.
        iris['variety'] = label_encoder.fit_transform(iris['variety'])
        # result after encoding
        print(f"Result after encoding:\n{iris['variety'].unique()}")
        # Modeling with scikit-learn
        # Model Selection and Creation
        X = iris.drop(['variety'], axis=1)
        y = iris['variety']
        # Data Splitting [Test,Training]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)
        print(f"Shape of X-Training : {x_train.shape}")
        print(f"Shape of Y-Training : {y_train.shape}")
        print(f"Shape of X-Test : {x_test.shape}")
        print(f"Shape of Y-Test : {y_test.shape}")
        # range of k we want to try
        k_range = list(range(1, 26))
        # empty list to store scores
        scores = []
        # Model Selection and Creation
        # loop through reasonable values of k
        for k in k_range:
            # run KNeighborsClassifier with k neighbours
            knn = KNeighborsClassifier(n_neighbors=k)
            # Train the model
            knn.fit(x_train, y_train)
            y_pred = knn.predict(x_test)
            # append scores for k neighbors to scores list
            scores.append(metrics.accuracy_score(y_test, y_pred))
        # we should have 26 scores here
        print(f'Length of list {len(scores)}')
        print(f'Max of list {max(scores)}')
        # plot how accuracy changes as we vary k
        plt.plot(k_range, scores)
        plt.xlabel('Value of k for KNN')
        plt.ylabel('Accuracy Score')
        plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
        plt.show()
        # Model Calculation Metrics and Results Explanation
        log_reg = LogisticRegression()
        log_reg.fit(x_train, y_train)
        y_pred = log_reg.predict(x_test)
        # Summary of the predictions made by the classifier
        print_report(y_pred, y_test)
        #####
        #
        # Random Forests
        #
        ####
        # Create a Gaussian Classifier
        clf = RandomForestClassifier(n_estimators=100)
        # Train the model
        clf.fit(x_train, y_train)
        rf_pred = clf.predict(x_test)
        # Summary of the predictions made by the classifier
        print_report(rf_pred, y_test)
    except:
        exception_type, exception_value, exception_traceback = sys.exc_info()
        print("Exception Type: {}\nException Value: {}".format(exception_type, exception_value))
        file_name, line_number, procedure_name, line_code = traceback.extract_tb(exception_traceback)[-1]
        print("File Name: {}\nLine Number: {}\nProcedure Name: {}\nLine Code: {}".format(file_name, line_number,
                                                                                         procedure_name, line_code))
    finally:
        pass


def print_report(y_pred, y_test):
    report = classification_report(y_test, y_pred)
    print(f"Classification Report\n {report}")
    confusion = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix\n {confusion}")
    accuracy_score = metrics.accuracy_score(y_test, y_pred)
    print(f"Accuracy Score{accuracy_score}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

