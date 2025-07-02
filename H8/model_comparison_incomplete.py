"""
Fill in the missing code. The lines with missing code have the string "#####" or '*'
"INSTRUCTIONS" comments explain how to fill in the mising code.
the outputfile.txt has the printouts from the program.
Your results should be similar within reason, if not, re-run the program,
since we are using RandomizedSearchCV, meaning there is some randomness involved.
Actually, we added np.random.seed() to fix the results, so you can check them.

You will be filling in code in two types of models:
1. a regression model and
2. a classification model.

Most of the time, because of similarities,
you can cut and paste from one model to the other.
But in a few instances, you cannot do this, so
you need to pay attention.
Also, in some cases,
you will find a "hint" for a solution 
in one of the two scripts (regression or classification)
that you can use as inspiration for the other.

This double task gives you the opportunity to look at the results
in both regression and classification approaches.

At the bottom, you will find some questions that we pose.
You do not need to write and turn in the answer to these questions,
but we strongly recommend you find out the answers to them.
"""

"""
In this script you will learn how to use pipelines to compare models
so as to select a better  model from a list of candidates.

There are a number of open source libraries in python that
allow you to compare models and are very easy to use:

1. PyCaret's compare_models() function allows you to compare Scikit-Learn models.
You can read about here:
https://archive.ph/vSeYy
https://archive.ph/h5HI3
https://pycaret.readthedocs.io/en/latest/index.html

2. H2O also has an AutoML function that allows you to compare models.
https://archive.ph/8DdJ4

3. Microsoft also has an AutoML function:
https://azure.microsoft.com/en-us/services/machine-learning/automatedml/
https://archive.ph/GuH96


However, it is important for you to know what these libraries are doing under wraps,
and it is always good to have a customized way to compare models that you can modify to your liking.
This is why you will compare models using the scikit-learn pipeline in this homework.

"""

import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import fAux
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn import preprocessing

np.random.seed(1) #to fix the results
 
#file_path = 'outputfile.txt'
#sys.stdout = open(file_path, "w")

#df = pd.read_csv('EURUSD_H3_200001030000_202107201800.csv', sep='\t')
#df = pd.read_csv('GBPUSD_H3_200001030000_202107201800.csv', sep='\t')
#df = pd.read_csv('NZDUSD_H3_200001030000_202107201800.csv', sep='\t')
df = pd.read_csv('USDCAD_H3_200001030000_202107201800.csv', sep='\t')
#df = pd.read_csv('USDCHF_H3_200001030000_202107201800.csv', sep='\t')

df['<DATETIME>'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
df = df.set_index('<DATETIME>')
df.drop(['<TIME>'], axis=1, inplace=True)
df.drop(['<DATE>'], axis=1, inplace=True)

#buld the best window features after the exploratory data analysis:
for n in [1,2,3,4,11,14]:
    name = 'ret' + str(n)
    df[name] = df["<OPEN>"].pct_change(periods=n) #for trading with open
    #df[name] = df["<CLOSE>"].pct_change(periods=n) #for trading with close

#build date-time features
df["hour"] = df.index.hour.values
df["day"] = df.index.dayofweek.values

#build target assuming we know today's open
df['retFut1'] = df['<OPEN>'].pct_change(1).shift(-1).fillna(0) #if you enter the trade immediately after the open
#df['retFut1'] = df['<CLOSE>'].pct_change(1).shift(-1) #if you wait until the close to enter the trade
#df = np.log(df+1)

#transform the target
df['retFut1_categ'] = np.where((df['retFut1'] > 0), 1, 0)

#Since we are trading right after the open, 
#we only know yesterday's  high low close volume spread etc.
df['<HIGH>'] = df['<HIGH>'].shift(1)
df['<LOW>'] = df['<LOW>'].shift(1)
df['<CLOSE>'] = df['<CLOSE>'].shift(1)
df['<VOL>'] = df['<VOL>'].shift(1)
df['<SPREAD>'] = df['<SPREAD>'].shift(1)

#select the features (by dropping)
cols_to_drop = ["<OPEN>","<HIGH>","<LOW>","<CLOSE>","<TICKVOL>","<VOL>","<SPREAD>"]  #optional
df_filtered = df.drop(cols_to_drop, axis=1)

#distribute the df data into X inputs and y target
X = df_filtered.drop(['retFut1', 'retFut1_categ'], axis=1) 
y = df_filtered[['retFut1_categ']]


#MixedNB expects categorical features to be label encoded
#as per  https://archive.ph/Ki1DS#selection-5521.0-5521.12
le = preprocessing.LabelEncoder()
X.hour = le.fit_transform(X.hour)
X.day = le.fit_transform(X.day)

#select the samples
x_train = X.iloc[0:10000]
x_test = X.iloc[10000:12000]

y_train = y.iloc[0:10000]
y_test = y.iloc[10000:12000]

df_train = df_filtered.iloc[0:10000]
df_test = df_filtered.iloc[10000:12000]


##########################################################################################################################

#set up the grid search and fit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer 
from sklearn import preprocessing
import phik
from phik.report import plot_correlation_matrix
from scipy.special import ndtr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


def phi_k(y_true, y_pred):
    dfc = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    try:
        phi_k_corr = dfc.phik_matrix(interval_cols=[]).iloc[1,0]
        phi_k_sig  = dfc.significance_matrix(interval_cols=[]).iloc[1,0]
        phi_k_p_val = 1 - ndtr(phi_k_sig) 
    except:
        phi_k_corr = 0
        phi_k_p_val = 0
    #print(phi_k_corr)
    print(phi_k_p_val)
    return phi_k_corr


#myscorer = None #use default accuracy score
myscorer = make_scorer(phi_k, greater_is_better=True)

numeric_sub_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler())])
categorical_sub_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
print(x_train.dtypes)
numeric_features_ix = x_train.select_dtypes(include=['float64']).columns
categorical_features_ix = x_train.select_dtypes(include=['int64']).columns

#Note: transformer 3-element tuples can be: ('name', function or pipeline, column_number_list or column_index)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_sub_pipeline, numeric_features_ix),
        ('cat', categorical_sub_pipeline, categorical_features_ix)], remainder='passthrough')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from mixed_naive_bayes import MixedNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

rfc_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])
rfc_pipeline.fit(x_train, y_train.values.squeeze())
y_pred = rfc_pipeline.predict(x_test)

"""
INSTRUCTIONS
Add a few more classifiers to this list
You may want to compare especially random forest with mixed naive bayes
however remember that MixedNB needs to know the categorical features column indexes, so
you need to use inspect_me to find out which catagorical features column indexes to include
after the preprocessor has finished preprocessing the inputs.
If you add MixedNB, 
Add PCA() to the numeric_sub_pipeline since the inputs of MixedNB are assumed to be non-correlated (="naive")
LabelEncode the categorical features as per the tutorial here: https://archive.ph/Ki1DS#selection-5521.0-5521.12
This involves some thinking, because unlike PCA(), LabelEncode() does not take as input a matrix X but an array y.
see: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
"""

inspect_me = preprocessor.fit_transform(x_train) #columns 6 to 18 are categorical

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    RandomForestClassifier(n_estimators=100, random_state=42),
    LogisticRegression(max_iter=1000, random_state=42),
    # MixedNB(categorical_features=np.arange(5,18))  # columns 6 to 18 are categorical
    ]
for classifier in classifiers:
    classifier_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', classifier)])
    classifier_pipe.fit(x_train, y_train.values.squeeze())   
    print(classifier)
    print("model score: %.3f" % classifier_pipe.score(x_test, y_test))


"""

INSTRUCTION
The above pipeline compares models with no grid-searching of parameters but
the comparison and the parameter grid-search can be done simultaneously.
Optionally extend the above code as follows:
Create a dictionary with mappings of classifier names with their information i.e. objects and parameter grids:
models_list = {'Logistic Regression': (classifier_lr, param_grid_lr),
               'K Nearest Neighbours': (classifier_knn, param_grid_knn)}
Iterate through every key-value pair in the dictionary and build your pipelines:
model_cvs = {}
for model_name, model_info in models_list.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                       ('classifier', model_info[0])])
    model_cvs[model_name] = train_and_score_model(model_name, pipeline, model_info[1])
"""