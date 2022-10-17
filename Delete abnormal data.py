#Import necessary libraries
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
import os
import math
import shutil
from PIL import Image
from sklearn.model_selection import GridSearchCV 
import warnings
warnings.filterwarnings("ignore")

# Data reading and data preprocessing
# Reading pictures For color pictures, no matter the picture format is PNG, BMP or JPG, in PIL, 
# after using the open() function of the Image module to open, the mode of the returned picture object is RGB.
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
data = pd.read_excel('./资料v2.0/资料v2.0/merge3.xlsx')
ID = ['ID', 'SIDE']
feature_con = ['V00AGE', 'P01BMI']
feature_dis = ['P02RACE', 'P02SEX',
               'V00NSAIDS(C+D)', 'V00COMORB', 'P02ACTRISK']
targets = ['V00WOMKP']
data = data[targets]
# Handling Missing Values
# Whether the result has missing values, remove the rows with missing values in the result
from sklearn.impute import KNNImputer
index_nan = data.index[np.where(np.isnan(data[targets]))[0]]
print('Index of row with missing label：', index_nan)
print('Each row of missing data is at most：', np.max(data.isnull().sum(axis=1)))
print('The number of rows with missing values is：', len(data)-data.dropna(axis=0, how='any').shape[0])
# Fill missing values using KNN method
imputer = KNNImputer(n_neighbors=3)
imputed = imputer.fit_transform(data)
data = pd.DataFrame(imputed, columns=data.columns)
print('After filling, the number of rows with missing values is：', len(data)-data.dropna(axis=0, how='any').shape[0])
# Normalized
data_fig = data_fig/255.0
data_fig = pd.DataFrame(data_fig.reshape(-1, shape**2))
data = pd.concat((data, data_fig), axis=1)
data[targets] = data[targets].astype(int)
data_targets = data[targets]
data = data.drop(columns=targets)
data = pd.concat([data, data_targets], axis=1)
# mess up the order
index = np.load('./index.npy')
data_ = data.iloc[index, :]
# Use the quartile method to remove abnormal data
img_feature_columns = [i for i in range(shape*shape)]
img_feature = data_[img_feature_columns].to_numpy()
means = []
for feature in img_feature:
    means.append(feature.mean())

def detect_outliers(to_drop):
    drop_index = []
    Q1 = np.percentile(to_drop, 25)
    Q3 = np.percentile(to_drop, 75)
    IQR = Q3 - Q1
    outlier_step = 1.5 * IQR
    for i, nu in enumerate(to_drop):
        if (nu < Q1 - outlier_step) | (nu > Q3 + outlier_step):
            drop_index.append(index[i])
    return np.array(drop_index)
index_to_drop = detect_outliers(means)
print('The number of the abnormal picture that needs to be deleted:', index_to_drop)
print('There are {} pictures in total'.format(len(index_to_drop)))
index_to_drop
# Remove abnormal data in data_
for i in index_to_drop:
    data_ = data_.drop(index=i)

a=input('Press any key to exit')