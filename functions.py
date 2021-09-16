import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix, classification_report, precision_score, recall_score, accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

def metrics(model, x_train, y_train, x_test, y_test):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    plot_confusion_matrix(model, x_train, y_train, ax=ax[0], cmap=plt.cm.Blues, xticks_rotation='vertical')
    ax[0].set_title('Train Confusion Matrix')
    plot_confusion_matrix(model, x_test, y_test, ax=ax[1], cmap=plt.cm.Blues, xticks_rotation='vertical')
    ax[1].set_title('Test Confusion Matrix')
    plt.show()
    print(classification_report(y_test, model.predict(x_test)))
    print('\n')

def make_model(model, x_train, y_train):
    return model.fit(x_train, y_train)

def all_models(x_train, x_test, y_train, y_test, 
               objects = [LogisticRegression(fit_intercept=False, C=1e12), 
                          Pipeline([('ss', StandardScaler()), ('knn', KNeighborsClassifier())]), 
                          GaussianNB(),
                          DecisionTreeClassifier(),
                          RandomForestClassifier(), 
                          XGBClassifier(), 
                          Pipeline([('ss', StandardScaler()), ('svm', SVC())])],
               index = ['Logistic Regression', 'K-Nearest Neighbors', 'Naive Bayes', 'Decision Tree', 
                        'Random Forest', 'XGBoost', 'Support Vector Machine']
               
              ):
    models = []
    precision = []
    recall = []
    accuracy = []
    f1 = []

    for i, o in enumerate(objects):
        print(f'{index[i]} Results:')
        models.append(make_model(o, x_train, y_train))
        metrics(models[-1], x_train, y_train, x_test, y_test)
        
    for i in models:
        prediction = i.predict(x_test)
        precision.append(precision_score(y_test, prediction, average='macro'))
        recall.append(recall_score(y_test, prediction, average='macro'))
        accuracy.append(accuracy_score(y_test, prediction))
        f1.append(f1_score(y_test, prediction, average='macro'))
    df = pd.DataFrame(np.array([precision, recall, accuracy, f1]).T, 
                      index = index, columns = ['Precision Score', 'Recall Score', 'Accuracy Score', 'F1 Score']).style.format('{:.2%}')
    display(df)
    print(f'The model with the highest precision score is {df.data.idxmax()[0]}.')
    print(f'The model with the highest recall score is {df.data.idxmax()[1]}.')
    print(f'The model with the highest accuracy score is {df.data.idxmax()[2]}.')
    print(f'The model with the highest F1 score is {df.data.idxmax()[3]}.')

    return models

def plot_importances(model, index, title='Feature Importances'):
    f_import = pd.Series(model.feature_importances_, index=index)
    plot = f_import.nlargest(20).to_frame()
    
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.barplot(data=plot, y=plot.index, x=0)
    ax.set_title(title)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    plt.show();