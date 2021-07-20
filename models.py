import logging

import numpy as np
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score, )
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


def print_result(classifier, expected, predicted, results):
    accuracy = accuracy_score(expected, predicted)
    recall = recall_score(expected, predicted, average="binary")
    precision = precision_score(expected, predicted, average="binary")
    f1 = f1_score(expected, predicted, average="binary")
    cm = metrics.confusion_matrix(expected, predicted)
    tpr = float(cm[1][1]) / np.sum(cm[1])
    fpr = float(cm[0][0]) / np.sum(cm[0])
    logging.debug("\n")
    logging.debug("-------" + classifier + "-------")
    logging.debug("Confusion matrix:\n" + str(cm))
    logging.debug(f"TPR: {tpr:.3f}, FPR: {fpr:.3f}")
    logging.debug(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F-Score: {f1:.3f}")

    results[classifier] = {
        # 'TPR': tpr,
        'ACC': accuracy,
        'PRE': precision,
        'REC': recall,
        'FPR': fpr,
        'F1': f1
    }


def train_and_test(x_train, y_train, x_test, y_test, model_list):
    results = {}

    if "Naive Bayes" in model_list:
        # fit a Naive Bayes model to the data
        model = GaussianNB()
        model.fit(x_train, y_train)
        print_result("Naive Bayes", y_test, model.predict(x_test), results)

    if "Logistic Regression" in model_list:
        # fit a logistic regression model to the data
        model = LogisticRegression(max_iter=2000)
        model.fit(x_train, y_train)
        print_result("Logistic Regression", y_test, model.predict(x_test), results)

    if "Decision Tree" in model_list:
        # fit a decision tree model to the data
        model = DecisionTreeClassifier()
        model.fit(x_train, y_train)
        logging.debug(model)
        print_result("Decision Tree", y_test, model.predict(x_test), results)

    if "AdaBoost" in model_list:
        # fit an ada boost classifier model to the data
        model = AdaBoostClassifier(n_estimators=100)
        model.fit(x_train, y_train)
        print_result("AdaBoost", y_test, model.predict(x_test), results)

    if "Random Forest" in model_list:
        # fit a random forest model to the data
        model = RandomForestClassifier(n_estimators=100)
        model = model.fit(x_train, y_train)
        # noinspection PyUnresolvedReferences
        print_result("Random Forest", y_test, model.predict(x_test), results)

    return results
