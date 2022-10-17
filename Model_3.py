#Import necessary libraries
print('Import necessary libraries')
import pandas as pd
import numpy as np
import scipy.io
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import os
import math
import shutil
from PIL import Image
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")

#Data reading and data preprocessing
print('Data reading...')
im = Image.open('./figs/fig.1.jpg')
print(im.format, im.size, im.mode)  
im.resize((30, 30))
# What is read from fig is the order of the corresponding IDs of the pictures
shape = 48  
data_fig = np.empty((600, shape, shape))
for i in range(600):
    img = Image.open('./figs/fig.'+str(i)+'.jpg')
    img_ = img.resize((shape, shape))
    array = np.asarray(img_, dtype="float32")
    data_fig[i, :, :] = array
data = pd.read_excel('./Tabular data/merge3.xlsx')
ID = ['ID', 'SIDE']
feature_con = ['V00AGE', 'P01BMI']
feature_dis = ['P02RACE', 'P02SEX',
               'V00NSAIDS(C+D)', 'V00COMORB', 'P02ACTRISK']
targets = ['V00WOMKP']
data = data[targets]#data = data[feature_dis+feature_con+targets]
feature_dis = ['P02RACE', 'P02SEX',
               'V00NSAIDS(C+D)', 'V00COMORB', 'P02ACTRISK']
# Normalize and stitch image data
data_fig = data_fig/255.0
data_fig = pd.DataFrame(data_fig.reshape(-1, shape**2))
data = pd.concat((data, data_fig), axis=1)
# Labels are rounded and placed last two lines
data[targets] = data[targets].astype(int)
data_targets = data[targets]
data = data.drop(columns=targets)
data = pd.concat([data, data_targets], axis=1)
# mess up the order
index = np.load('./index.npy')
data_ = data.iloc[index,:]
# remove abnormal data
index_to_drop=[ 57, 528, 383,  65, 386, 260,   9, 355, 409, 333, 242, 250, 346,
       258,  10, 147, 567, 286, 511, 493, 576,  35, 595, 557, 313, 109,
       282, 254,  21, 560, 359,  33, 539]
for i in index_to_drop:
    data_ = data_.drop(index=i)
# Divide the dataset into 10 parts
print('Divide the dataset into 10 parts')
data_dict = dict()
size = math.floor(len(data_)/10)
remaind = len(data_)-size*10
sample_sizes = []
for i in range(10):
    if remaind > 0:
        sample_sizes.append(size+1)
        remaind -= 1
    else:
        sample_sizes.append(size)
start = 0
for i, s in enumerate(sample_sizes):
    end = start+s
    data_dict[i] = data_.iloc[start:end]
    start = end
# Regenerate the dataset
print('Regenerate the dataset')
dataset_dict = dict()
for i in range(10):
    ith_dataset = data_dict[i]
    for j in range(9):
        dataset_key = (i+j+1) % 10
        ith_dataset = pd.concat([ith_dataset, data_dict[dataset_key]], axis=0)
    dataset_dict[i]=ith_dataset
#random forest
def CrossValidationForRF(x_train, y_train, x_test, y_test, d_test, d_train):
    Hyperparameter = {"n_estimators": [1, 10, 50, 100, 500],
                      "criterion": ["gini", "entropy"],
                      "max_depth": [5, 10, 15, 25],
                      "min_samples_split": [2, 5, 10]}
    grid = GridSearchCV(estimator=RandomForestClassifier(),
                        param_grid=Hyperparameter)
    grid_result = grid.fit(x_train, y_train)
    rf_parameters = grid_result.best_params_
    model = OneVsRestClassifier(RandomForestClassifier(**rf_parameters))
    model.fit(x_train, y_train) 
    trainPredict_RF = model.predict_proba(x_train)
    testPredict_RF = model.predict_proba(x_test)   
    best_train_auc= roc_auc_score(d_train, trainPredict_RF, average='micro')
    best_test_auc=roc_auc_score(d_test, testPredict_RF, average='micro')   
    return rf_parameters, best_train_auc, best_test_auc,testPredict_RF
#Support Vector Machines
def CrossValidationForSVM(x_train, y_train, x_test, y_test, d_test, d_train):
    Hyperparameter = [{'kernel': ['rbf'], 'C': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 1, 10, 100, 1000]},
                      {'kernel': ['linear'], 'C': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 1, 10, 100, 1000]}]
    grid = GridSearchCV(estimator=SVC(), param_grid=Hyperparameter)
    grid_result = grid.fit(x_train, y_train)
    svm_parameter = grid_result.best_params_
    model = OneVsRestClassifier(SVC(**svm_parameter, probability=True))
    model.fit(x_train, y_train) 
    train_predict_SVM = model.predict_proba(x_train)
    test_predict_SVM = model.predict_proba(x_test)
    train_auc = roc_auc_score(d_train, train_predict_SVM, average='micro')
    test_auc = roc_auc_score(d_test, test_predict_SVM, average='micro')
    return svm_parameter, train_auc, test_auc,test_predict_SVM
#logistic regression
def CrossValidationForLR(x_train, y_train, x_test, y_test, d_test, d_train):
    C = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 1, 10, 100, 1000]
    penaltys = ['l1', 'l2']
    # Pack hyperparameter ranges into a dictionary
    Hyperparameter = dict(C=C, penalty=penaltys)
    # SVC model in support vector machine
    grid = GridSearchCV(estimator=LogisticRegression(),param_grid=Hyperparameter)
    # Fit the model to the training dataset
    grid_result = grid.fit(x_train, y_train)
    # Return the best parameter combination
    lr_parameter = grid_result.best_params_
    # LogisticRegression
    model = OneVsRestClassifier(LogisticRegression(**lr_parameter))
    model.fit(x_train, y_train)
    train_predict_LR = model.predict_proba(x_train)
    test_predict_LR = model.predict_proba(x_test)
    train_auc = roc_auc_score(d_train, train_predict_LR, average='micro')
    test_auc = roc_auc_score(d_test, test_predict_LR, average='micro')
    return lr_parameter, train_auc, test_auc,test_predict_LR
#decision tree
def CrossValidationForTRE(x_train, y_train, x_test, y_test, d_test, d_train):
    from sklearn.model_selection import GridSearchCV
    from sklearn.tree import DecisionTreeClassifier
    entropy_thresholds = np.linspace(0, 1, 100)
    gini_thresholds = np.linspace(0, 0.2, 100)
    #set parameter matrix
    param_grid = [{'criterion': ['entropy'], 'min_impurity_decrease': entropy_thresholds},
            {'criterion': ['gini'], 'min_impurity_decrease': gini_thresholds},
            {'max_depth': np.arange(2,10)},
            {'min_samples_split': np.arange(2,30,2)}]
    grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
    grid.fit(x_train,y_train)
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestRegressor   
    tre_parameter = grid.best_params_
    trainPredict_tre = grid.predict_proba(x_train)
    testPredict_tre = grid.predict_proba(x_test)
    train_auc = roc_auc_score(d_train, trainPredict_tre, average='micro')
    test_auc = roc_auc_score(d_test, testPredict_tre, average='micro')
    return tre_parameter, train_auc, test_auc,testPredict_tre
#Bayesian
def CrossValidationForBYS(x_train, y_train, x_test, y_test, d_test, d_train):
    from skopt import BayesSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    import warnings
    from skopt.space import Real, Categorical, Integer
    warnings.filterwarnings("ignore")
    # parameter ranges are specified by one of below 
    knn = KNeighborsClassifier()
    #defining hyper-parameter grid
    grid_param = { 'n_neighbors' : list(range(2,11)) , 
                'algorithm' : ['auto','ball_tree','kd_tree','brute'] }
    #initializing Bayesian Search
    Bayes = BayesSearchCV(knn , grid_param , n_iter=30 , random_state=14)
    Bayes.fit(x_train,y_train)
    bys_parameter = Bayes.best_params_
    train_predict_BYS = Bayes.predict_proba(x_train)
    test_predict_BYS = Bayes.predict_proba(x_test)   
    train_auc = roc_auc_score(d_train, train_predict_BYS, average='micro')
    test_auc = roc_auc_score(d_test, test_predict_BYS, average='micro')
    return bys_parameter, train_auc, test_auc,test_predict_BYS

#Ten cross-validation
print('Ten cross-validation')
train_aucs = [[] for i in range(5)]
test_aucs = [[] for i in range(5)]
best_parameters = [[] for i in range(5)]
tprs_RF=[]
tprs_SVM=[]
tprs_LR=[]
tprs_BYS=[]
tprs_TRE=[]
mean_fpr=np.linspace(0,1,100)
for i in tqdm(range(10)):
    kth_data=dataset_dict[i].copy()
    x_train = kth_data.iloc[:int(0.9*len(kth_data)), :-2]
    y_train = kth_data.iloc[:int(0.9*len(kth_data)), -1]
  
    x_test = kth_data.iloc[int(0.9*len(kth_data)):, :-2]
    y_test = kth_data.iloc[int(0.9*len(kth_data)):, -1]
    
    # Use one-hot form when calculating AUC and 0, 1, 2, 3 when training the model
    d_test = pd.get_dummies(y_test)
    d_train = pd.get_dummies(y_train)

    rf_parameter, rf_train_auc, rf_test_auc,testPredict_RF = CrossValidationForRF(x_train, y_train, x_test, y_test, d_test, d_train)
    train_aucs[0].append(rf_train_auc)
    test_aucs[0].append(rf_test_auc)
    best_parameters[0].append(rf_parameter)
    
    svm_parameter, svm_train_auc, svm_test_auc,testPredict_SVM=CrossValidationForSVM(x_train, y_train, x_test, y_test, d_test, d_train)
    train_aucs[1].append(svm_train_auc)
    test_aucs[1].append(svm_test_auc)
    best_parameters[1].append(svm_parameter)
    
    lr_parameter, lr_train_auc, lr_test_auc,testPredict_LR=CrossValidationForLR(x_train, y_train, x_test, y_test, d_test, d_train)
    train_aucs[2].append(lr_train_auc)
    test_aucs[2].append(lr_test_auc)
    best_parameters[2].append(lr_parameter)

    tre_parameter, tre_train_auc, tre_test_auc,testPredict_TRE=CrossValidationForTRE(x_train, y_train, x_test, y_test, d_test, d_train)
    train_aucs[3].append(tre_train_auc)
    test_aucs[3].append(tre_test_auc)
    best_parameters[3].append(tre_parameter)
    
    bys_parameter, bys_train_auc, bys_test_auc,testPredict_BYS=CrossValidationForBYS(x_train, y_train, x_test, y_test, d_test, d_train)
    train_aucs[4].append(bys_train_auc)
    test_aucs[4].append(bys_test_auc)
    best_parameters[4].append(bys_parameter)

    # First expand the matrices y_one_hot and y_score, then calculate the false positive rate FPR and true rate TPR
    fpr_RF, tpr_RF, thresholds_RF = roc_curve(
        d_test.values.ravel(), testPredict_RF.ravel())
    auc_RF = auc(fpr_RF, tpr_RF)
    tprs_RF.append(interp(mean_fpr,fpr_RF,tpr_RF))
    tprs_RF[-1][0]=0.0

    fpr_SVM, tpr_SVM, thresholds_SVM = roc_curve(
        d_test.values.ravel(), testPredict_SVM.ravel())
    auc_SVM = auc(fpr_SVM, tpr_SVM)
    tprs_SVM.append(interp(mean_fpr,fpr_SVM,tpr_SVM))
    tprs_SVM[-1][0]=0.0

    fpr_LR, tpr_LR, thresholds_LR = roc_curve(
        d_test.values.ravel(), testPredict_LR.ravel())
    auc_LR = auc(fpr_LR, tpr_LR)
    tprs_LR.append(interp(mean_fpr,fpr_LR,tpr_LR))
    tprs_LR[-1][0]=0.0

    fpr_TRE, tpr_TRE, thresholds_TRE = roc_curve(
        d_test.values.ravel(), testPredict_TRE.ravel())
    auc_TRE = auc(fpr_TRE, tpr_TRE)
    tprs_TRE.append(interp(mean_fpr,fpr_TRE,tpr_TRE))
    tprs_TRE[-1][0]=0.0

    fpr_BYS, tpr_BYS, thresholds_BYS = roc_curve(
        d_test.values.ravel(), testPredict_BYS.ravel())
    auc_BYS = auc(fpr_BYS, tpr_BYS)
    tprs_BYS.append(interp(mean_fpr,fpr_BYS,tpr_BYS))
    tprs_BYS[-1][0]=0.0
#output result
print('AUC value of RF (10-fold cross-validation):',test_aucs[0])
print('AUC value of SVM (10-fold cross-validation):',test_aucs[1])
print('AUC value of LR (10-fold cross-validation):',test_aucs[2])
print('AUC value of TRE (10-fold cross-validation):',test_aucs[3])
print('AUC value of BYS (10-fold cross-validation):',test_aucs[4])

print('RF AUC value (10-fold cross-validation)')
print("AUC of the training set：", sum(train_aucs[0])/10)
print("AUC of the testing set：", sum(test_aucs[0])/10)

print('SVM AUC value (10-fold cross-validation)')
print("AUC of the training set：", sum(train_aucs[1])/10)
print("AUC of the testing set：", sum(test_aucs[1])/10)

print('LR AUC  value (10-fold cross-validation)')
print("AUC of the training set：", sum(train_aucs[2])/10)
print("AUC of the testing set：", sum(test_aucs[2])/10)

print('TRE AUC  value (10-fold cross-validation)')
print("AUC of the training set：", sum(train_aucs[3])/10)
print("AUC of the testing set：", sum(test_aucs[3])/10)

print('BYS AUC  value (10-fold cross-validation)')
print("AUC of the training set：", sum(train_aucs[4])/10)
print("AUC of the testing set：", sum(test_aucs[4])/10)

a=input('Press any key to exit')