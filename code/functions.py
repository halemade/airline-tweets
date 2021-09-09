#import the appropriate tools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
import plotly.graph_objects as go
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, recall_score, precision_score,\
accuracy_score,f1_score,confusion_matrix,plot_confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
import plotly.express as px
import warnings
import time
warnings.filterwarnings('ignore')

def vanilla_models(X,y,test_size=.3):
    """ This function takes in predictors, a target variable and an optional test
    size parameter and returns results for 9 baseline classifiers"""
    
    names = ["Logistic Regression","Nearest Neighbors","Naive Bayes", "Linear SVM", "RBF SVM","Decision Tree",
             "Random Forest", "Gradient Boost", "AdaBoost","XGBoost"]
    
    req_scaling = ["Nearest Neighbors"]

    classifiers = [
        LogisticRegression(),
        KNeighborsClassifier(3),
        GaussianNB(),
        SVC(kernel="linear", C=.5),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        GradientBoostingClassifier(),
        AdaBoostClassifier(),
        XGBClassifier()
        ]  
    
    #init df to hold report info for all classifiers
    df = pd.DataFrame(columns = ['classifier','train accuracy','train precision',
                                 'train recall','train f1 score','test accuracy',
                                 'test precision','test recall','test f1 score',
                                 'test time'])
    
    #train test splitsies
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .3,random_state=42)
    
    #iterate over classifiers
    for count,clf in enumerate(classifiers):
        start = time.time()
        scaler = StandardScaler()
        if names[count] in req_scaling:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled  = scaler.transform(X_test)
 
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        clf.fit(X_train_scaled,y_train)
        train_preds = clf.predict(X_train_scaled)
        test_preds = clf.predict(X_test_scaled)
        
        #training stats
        train_recall = round(recall_score(y_train,train_preds,average = 'weighted'),3)
        train_precision = round(precision_score(y_train,train_preds,average='weighted'),3)
        train_acc = round(accuracy_score(y_train,train_preds),3)
        train_f1 = round(f1_score(y_train,train_preds,average='weighted'),3)
        
        #testing stats
        recall = round(recall_score(y_test,test_preds,average='weighted'),3)
        precision = round(precision_score(y_test,test_preds,average='weighted'),3)
        f1 = round(f1_score(y_test,test_preds,average='weighted'),3)
        cm = confusion_matrix(y_test,test_preds)
        acc = round(accuracy_score(y_test,test_preds),3)
        end = time.time()
        elapsed = round((end-start),2)
        
        #append results to dataframe
        df = df.append({'classifier':names[count],'train accuracy':train_acc,
                        'train precision':train_precision,'train recall':train_recall,
                        'train f1 score':train_f1,'test accuracy':acc,
                        'test precision':precision,'test recall':recall,
                        'test f1 score':f1,'test time':elapsed},ignore_index=True)
    return df

def run_model(clf,X,y):
    #train test splitsies
    """takes in an instantiated classifier and the predictive and target data. 
    use only for models on that do not require data scaling"""
    
    start = time.time()
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=42)
    X_train, y_train = SMOTE().fit_resample(X_train,y_train)
    clf.fit(X_train,y_train)
    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)
    model_report = classification_report(y_test, test_preds,target_names = labels.keys(),output_dict = True)

    #training stats
    train_recall = round(recall_score(y_train,train_preds,average = 'weighted'),3)
    train_precision = round(precision_score(y_train,train_preds,average='weighted'),3)
    train_acc = round(accuracy_score(y_train,train_preds),3)
    train_f1 = round(f1_score(y_train,train_preds,average='weighted'),3)

    #testing stats
    recall = round(recall_score(y_test,test_preds,average='weighted'),3)
    precision = round(precision_score(y_test,test_preds,average='weighted'),3)
    f1 = round(f1_score(y_test,test_preds,average='weighted'),3)
    cm = confusion_matrix(y_test,test_preds)
    acc = round(accuracy_score(y_test,test_preds),3)
    end = time.time()
    elapsed = round((end-start),2)
    #append results to dataframe
    report = dict({'classifier':clf,'train accuracy':train_acc,
                    'train precision':train_precision,'train recall':train_recall,
                    'train f1 score':train_f1,'test accuracy':acc,
                    'test precision':precision,'test recall':recall,
                    'test f1 score':f1,'test time':elapsed})
    #plot confusion matrix
    train_plot = plot_confusion_matrix(clf,X_train,y_train)
    test_plot = plot_confusion_matrix(clf,X_test,y_test)
    return report, "Top plot: Training Data", "Bottom Plot: Testing Data"


def run_scaled_model(clf,X,y):
    #order: TTS,scale,resample, use only the scaled for predictions
    #train test splitsies
    """takes in an instantiated classifier and the predictive and target data. 
    use only for models on that require data scaling"""
    start = time.time()
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=42)
    #!!!scale before resampling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train_scaled,y_train)
    clf.fit(X_train_resampled,y_train_resampled)
    train_preds = clf.predict(X_train_scaled) #don't predict on resampled data, predict on scaled X_train
    test_preds = clf.predict(X_test_scaled)
   

    #training stats
    train_recall = round(recall_score(y_train,train_preds,average = 'weighted'),3)
    train_precision = round(precision_score(y_train,train_preds,average='weighted'),3)
    train_acc = round(accuracy_score(y_train,train_preds),3)
    train_f1 = round(f1_score(y_train,train_preds,average='weighted'),3)

    #testing stats
    recall = round(recall_score(y_test,test_preds,average='weighted'),3)
    precision = round(precision_score(y_test,test_preds,average='weighted'),3)
    f1 = round(f1_score(y_test,test_preds,average='weighted'),3)
    cm = confusion_matrix(y_test,test_preds)
    acc = round(accuracy_score(y_test,test_preds),3)
    end = time.time()
    elapsed = round((end-start),2)
    #append results to dataframe
    report = dict({'classifier':clf,'train accuracy':train_acc,
                    'train precision':train_precision,'train recall':train_recall,
                    'train f1 score':train_f1,'test accuracy':acc,
                    'test precision':precision,'test recall':recall,
                    'test f1 score':f1,'test time':elapsed})
    #plot confusion matrix
    train_plot = plot_confusion_matrix(clf,X_train,y_train)
    test_plot = plot_confusion_matrix(clf,X_test,y_test)
    return report, "Top plot: Training Data", "Bottom Plot: Testing Data"

def plot_importances(model_dict,X):
    features = dict(zip(X.columns,model_dict[0]['classifier'].feature_importances_))
    fi = pd.DataFrame({
    "features": list(X.columns),
    "importances": model_dict[0]['classifier'].feature_importances_ ,
    })
    sort = fi.sort_values(by=['importances'],ascending=False)
    fig = px.bar(sort, x="features", y="importances", barmode="group")
    fig.update_layout(title = 'XGBoost Feature Importances')    
    return fig.show()

def plot_roc(results):
    #plot ROC curve
    kind = 'val'
    c_fill      = 'rgba(52, 152, 219, 0.2)'
    c_line      = 'rgba(52, 152, 219, 0.5)'
    c_line_main = 'rgba(41, 128, 185, 1.0)'
    c_grid      = 'rgba(189, 195, 199, 0.5)'
    c_annot     = 'rgba(149, 165, 166, 0.5)'
    c_highlight = 'rgba(192, 57, 43, 1.0)'
    fpr_mean    = np.linspace(0, 1, 100)
    interp_tprs = []
    for i in range(100):
        fpr           = results[kind]['fpr'][i]
        tpr           = results[kind]['tpr'][i]
        interp_tpr    = np.interp(fpr_mean, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)
    tpr_mean     = np.mean(interp_tprs, axis=0)
    tpr_mean[-1] = 1.0
    tpr_std      = 2*np.std(interp_tprs, axis=0)
    tpr_upper    = np.clip(tpr_mean+tpr_std, 0, 1)
    tpr_lower    = tpr_mean-tpr_std
    auc          = np.mean(results[kind]['auc'])
    fig = go.Figure([
        go.Scatter(
            x          = fpr_mean,
            y          = tpr_upper,
            line       = dict(color=c_line, width=1),
            hoverinfo  = "skip",
            showlegend = False,
            name       = 'upper'),
        go.Scatter(
            x          = fpr_mean,
            y          = tpr_lower,
            fill       = 'tonexty',
            fillcolor  = c_fill,
            line       = dict(color=c_line, width=1),
            hoverinfo  = "skip",
            showlegend = False,
            name       = 'lower'),
        go.Scatter(
            x          = fpr_mean,
            y          = tpr_mean,
            line       = dict(color=c_line_main, width=2),
            hoverinfo  = "skip",
            showlegend = True,
            name       = f'AUC: {auc:.3f}')
    ])
    fig.add_shape(
        type ='line', 
        line =dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_layout(
        template    = 'plotly_white', 
        title_x     = 0.5,
        xaxis_title = "1 - Specificity",
        yaxis_title = "Sensitivity",
        width       = 800,
        height      = 800,
        legend      = dict(
            yanchor="bottom", 
            xanchor="right", 
            x=0.95,
            y=0.01,
        )
    )
    fig.update_yaxes(
        range       = [0, 1],
        gridcolor   = c_grid,
        scaleanchor = "x", 
        scaleratio  = 1,
        linecolor   = 'black')
    fig.update_xaxes(
        range       = [0, 1],
        gridcolor   = c_grid,
        constrain   = 'domain',
        linecolor   = 'black')
    return fig.show()
