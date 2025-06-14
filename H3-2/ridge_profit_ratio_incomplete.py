"""
Fill in the missing code. The lines with missing code have the string "#####" or '*'
"INSTRUCTIONS" comments explain how to fill in the mising code.
the outputfile.txt has the printouts from the program.
"RESULTS" comments explain what results to expect from the program.
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
In this homework you will program your own custom scoring function, the profit factor (ratio)
"""

import warnings
warnings.simplefilter('ignore')
 

import numpy as np
import pandas as pd
import fAux
import matplotlib.pyplot as plt
import seaborn as sns
import sys

np.random.seed(1) #to fix the results
rs = 2
 
file_path = 'outputfile.txt'
sys.stdout = open(file_path, "w")


#df = pd.read_csv('EURUSD_H3_200001030000_202107201800.csv', sep='\t')
#df = pd.read_csv('GBPUSD_H3_200001030000_202107201800.csv', sep='\t')
#df = pd.read_csv('NZDUSD_H3_200001030000_202107201800.csv', sep='\t')
df = pd.read_csv('USDCAD_H3_200001030000_202107201800.csv', sep='\t')
#df = pd.read_csv('USDCHF_H3_200001030000_202107201800.csv', sep='\t')

df['<DATETIME>'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
df = df.set_index('<DATETIME>')
df.drop(['<TIME>'], axis=1, inplace=True)
df.drop(['<DATE>'], axis=1, inplace=True)

#save the open for white reality check
#openp = df['<OPEN>'].copy() #for the case we want to enter trades at the open


##build window momentum features
for n in list(range(1,21)):
    name = 'ret' + str(n)
    df[name] = df["<OPEN>"].pct_change(periods=n)#for trading with open
    
##build window momentum features (best features, but result is similar than w/ all 21)
#for n in list(range(1,6)):
#    name = 'ret' + str(n)
#    df[name] = df["<OPEN>"].pct_change(periods=n) #for trading with open

#build date time features
df["hour"] = df.index.hour.values
df["day"] = df.index.dayofweek.values
df_dummies_hour = pd.get_dummies(df["hour"], prefix='hour')
df_dummies_day = pd.get_dummies(df["day"], prefix='day')
df =df.join(df_dummies_hour)
df=df.join(df_dummies_day)
df.drop(["hour","day"], axis=1, inplace=True)


#build target assuming we know today's open
df['retFut1'] = df['<OPEN>'].pct_change(1).shift(-1).fillna(0) #if you enter the trade immediately after the open
#df = np.log(df+1)

#Since we are trading right after the open, 
#we only know yesterday's  high low close volume spread etc.
df['<HIGH>'] = df['<HIGH>'].shift(1)
df['<LOW>'] = df['<LOW>'].shift(1)
df['<CLOSE>'] = df['<CLOSE>'].shift(1)
df['<VOL>'] = df['<VOL>'].shift(1)
df['<SPREAD>'] = df['<SPREAD>'].shift(1)

#select the features (by dropping)
cols_to_drop = ["<OPEN>","<HIGH>","<LOW>","<CLOSE>","<TICKVOL>","<VOL>","<SPREAD>"]  #optional
df.drop(cols_to_drop, axis=1, inplace=True)

#distribute the df data into X inputs and y target
X = df.drop(['retFut1'], axis=1)
y = df[['retFut1']]

#select the samples
x_train = X.iloc[0:10000]
x_test = X.iloc[10000:12000]

y_train = y.iloc[0:10000]
y_test = y.iloc[10000:12000]

##########################################################################################################################
#Exploratory Data Analysis
from scipy.stats import pearsonr
from scipy.stats import spearmanr

def plot_corr(corr, size=5, title="Pearson correlation"):
    """Function plots a graphical correlation matrix dataframe 
    Input:
        df: pandas corr DataFrame from e.g. corr = df.corr(method='pierson')
        size: vertical and horizontal size of the plot
        title: title of the plot
    """
    fig, ax = plt.subplots(figsize=(size, size))
    #optionally, substitute the regular ax with an sns.heatmap (the regular ax has horrible colors)
    ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    annot=True,
    square=True
    )
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title(title)
    #plt.tight_layout() #can use instead of rotation='vertical'
    
#get the non-categorical data  by excluding the uint8 dtype
#we do this because neither the spearman nor the pearson correlations can deal with categorical variables    
df_filtered = df.select_dtypes(exclude=['uint8'])

#Plot the matrix containing the pair-wise spearman coefficients
#ideally we want to pay special attention to the correlation between each period return and retFut1    
corr = df_filtered.corr(method='spearman').round(2)
plot_corr(corr, size=20, title="Spearman correlation")
#plt.show()
plt.savefig(r'Results\%s.png' %("Spearman Correlation Matrix"))

#Plot the matrix containing the pair-wise pearson coefficients
##ideally we want to pay special attention to the correlation between each period return and retFut1    
corr = df_filtered.corr(method='pearson').round(2)
plot_corr(corr, size=20, title="Pearson correlation")
#plt.show()
plt.savefig(r'Results\%s.png' %("Pearson Correlation Matrix"))

#You want to plot the significance of the correlations
#This is optional in our case because we have tons of data.
def calculate_pvalues(df, method='pearson'):
    # df = df.dropna()._get_numeric_data()
    df = df.dropna().select_dtypes(include=['number']).loc[:, df.dtypes != 'bool']

    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        if method=='pearson':
            for c in df.columns:
                pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
        else:
            for c in df.columns:
                pvalues[r][c] = round(spearmanr(df[r], df[c])[1], 4)                
    return pvalues

#plot the significance values
print('pearson significance values')
print(calculate_pvalues(df_filtered, method='pearson'))
print('spearman significance values')
print(calculate_pvalues(df_filtered, method='spearman'))

plt.close("all") 
##########################################################################################################################
#set up the grid search and fit

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr
from sklearn.metrics import make_scorer 
from sklearn.impute import SimpleImputer


def information_coefficient(y_true, y_pred):
    rho, pval = spearmanr(y_true,y_pred) #spearman's rank correlation
    print (rho)
    return rho


def sharpe(y_true, y_pred):
    positions_arr = np.where(y_pred> 0,1,-1 )
    positions_arr = np.nan_to_num(positions_arr, nan=0.0)
    dailyRet_arr = positions_arr * y_true
    dailyRet_arr = np.nan_to_num(dailyRet_arr, nan=0.0)
    ratio = (252.0 ** (1.0/2.0)) * np.mean(dailyRet_arr) / np.std(dailyRet_arr)
    return ratio

"""
INSTRUCTIONS

In a previous homework, we gave the code for the profit ratio criterion.
Adapt this code to the function profit_ratio placed just below.
The function header takes two parameters, y_true and y_pred which are both arrays.
The function returns the profit_ratio, which is a float.
The function is called further below.

"""

def profit_ratio(y_true, y_pred):
    positions = np.where(y_pred > 0, 1, -1)
    positions = np.nan_to_num(positions, nan=0.0)

    daily_ret = positions * y_true
    daily_ret = np.nan_to_num(daily_ret, nan=0.0)

    profit = np.sum(daily_ret[daily_ret > 0])
    loss = -np.sum(daily_ret[daily_ret < 0])
    if loss == 0:
        profit_ratio = np.inf 
    else:
        profit_ratio = profit / loss
    
    return profit_ratio
    

"""
INSTRUCTIONS

Use the make_scorer sciki-learn utility to define a scorer called myscorer, 
similar to the one that is already defined for the sharpe ratio,
but substituting the call to the sharpe function by the call to profit_ratio that
you defined above.
Note that myscorer is used as a parameter in 
RandomizedSearchCV(pipe, param_grid, cv=5, scoring=myscorer, return_train_score=True)
further below.
Once you have defined this extra myscorer,
you need to do two things:
complete the code for in-sample and out-of-sample profit ratio below and
complete the file called 
ridge_profit_ratio_results_incomplete.txt

"""
#myscorer = None #uses the default r2 score, not recommended
#myscorer = "neg_mean_absolute_error"
#myscorer = make_scorer(information_coefficient, greater_is_better=True)
#myscorer = make_scorer(sharpe, greater_is_better=True)
myscorer = make_scorer(information_coefficient, greater_is_better=True)

imputer = SimpleImputer(strategy="constant", fill_value=0)
#we turn off scaling because we should not scale dummies (and returns are already mostly scaled)
scaler = StandardScaler(with_mean=False, with_std=False)
ridge = Ridge(max_iter=1000) 

pipe = Pipeline([("imputer", imputer), ("scaler", scaler), ("ridge", ridge)])
a_rs = np.logspace(-7, 0, num=20, endpoint = True)

param_grid =  [{'ridge__alpha': a_rs}]

grid_search = RandomizedSearchCV(pipe, param_grid, cv=5, scoring=myscorer, return_train_score=True, random_state=rs)
#grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring=myscorer, return_train_score=True)

grid_search.fit(x_train.values, y_train.values.ravel())

best_parameters = grid_search.best_params_
best_model = grid_search.best_estimator_


print("Best parameters : {}".format(best_parameters))
#print('Best estimator {}'.format(best_model))
print("Best cross-validation score : {:.2f}".format(grid_search.best_score_*100))
results = pd.DataFrame(grid_search.cv_results_)

#print(results.T)
results.to_csv("results_ridgereg.csv")


#########################################################################################################################

# Train set
# Make "predictions" on training set (in-sample)
#positions = np.where(best_model.predict(x_train.values)> 0,1,-1 )
positions = np.where(grid_search.predict(x_train.values)> 0,1,-1 ) #POSITIONS

dailyRet = pd.Series(positions).fillna(0).values * y_train.retFut1 #for trading right after the open

dailyRet = dailyRet.fillna(0)

cumret = np.cumprod(dailyRet + 1) - 1

plt.figure(1)
plt.plot(cumret.index, cumret)
plt.title('Cross-validated RidgeRegression on currency: train set')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
#plt.show()
plt.savefig(r'Results\%s.png' %("TrainCumulative"))

"""
INSTRUCTIONS
complete the profit_ratio (p_ratio) code below

"""

cagr = (1 + cumret[-1]) ** (252 / len(cumret)) - 1
maxDD, maxDDD = fAux.calculateMaxDD(cumret)
s_ratio = (252.0 ** (1.0/2.0)) * np.mean(dailyRet) / np.std(dailyRet)
profits = np.sum(dailyRet[dailyRet > 0])
losses = -np.sum(dailyRet[dailyRet < 0])
p_ratio = profits / losses if losses != 0 else np.inf
print (('In-sample: CAGR={:0.6} Sharpe ratio={:0.6} Profit ratio={:0.6} maxDD={:0.6} maxDDD={:d} Calmar ratio={:0.6}\n'\
).format(cagr, s_ratio, p_ratio, maxDD, maxDDD.astype(int), -cagr/maxDD))

# Test set
# Make "predictions" on test set (out-of-sample)
#positions2 = np.where(best_model.predict(x_test.values)> 0,1,-1 ) 
positions2 = np.where(grid_search.predict(x_test.values)> 0,1,-1 ) #POSITIONS

dailyRet2 = pd.Series(positions2).fillna(0).values * y_test.retFut1 #for trading right after the open
dailyRet2 = dailyRet2.fillna(0)

cumret2 = np.cumprod(dailyRet2 + 1) - 1

plt.figure(2)
plt.plot(cumret2.index, cumret2)
plt.title('Cross-validated RidgeRegression on currency: test set')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
#plt.show()
plt.savefig(r'Results\%s.png' %("TestCumulative"))

rho, pval = spearmanr(y_test,grid_search.predict(x_test.values)) #spearman's rank correlation: very small but significant


"""
INSTRUCTIONS
complete the profit_ratio (p_ratio) code below
make sure you use dailyRet2, not dailyRet

"""
cagr = (1 + cumret2[-1]) ** (252*8 / len(cumret2)) - 1
maxDD, maxDDD = fAux.calculateMaxDD(cumret2)
s_ratio = (252.0 ** (1.0/2.0)) * np.mean(dailyRet2) / np.std(dailyRet2)
profits = np.sum(dailyRet2[dailyRet2 > 0])
losses = -np.sum(dailyRet2[dailyRet2 < 0])
p_ratio = profits / losses if losses != 0 else np.inf
print (('Out-of-sample: CAGR={:0.6} Sharpe ratio={:0.6} Profit ratio={:0.6} maxDD={:0.6} maxDDD={:d} Calmar ratio={:0.6}\n'\
).format(cagr, s_ratio, p_ratio, maxDD, maxDDD.astype(int), -cagr/maxDD))


#plot the residuals
true_y = y_test.values.ravel()
pred_y = grid_search.predict(x_test.values)
residuals = np.subtract(true_y, pred_y)

from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf
fig, axes = plt.subplots(ncols=2, figsize=(14,4))
# sns.distplot(residuals, fit=norm, ax=axes[0], axlabel='Residuals', label='Residuals')
# use histplot instead of distplot
sns.histplot(residuals, kde=True, stat='density', ax=axes[0], label='Residuals')
axes[0].set_title('Residual Distribution')
axes[0].legend()
plot_acf(residuals, lags=10, zero=False, ax=axes[1], title='Residual Autocorrelation')
axes[1].set_xlabel('Lags')
sns.despine()
fig.tight_layout();
#plt.show()
plt.savefig(r'Results\%s.png' %("Residuals"))
plt.close("all") 

#Residual autocorrelation
#If the p-value of the test is greater than the required significance (>0.05), residuals are independent
import statsmodels.api as sm
lb = sm.stats.acorr_ljungbox(residuals, lags=[10], boxpierce=False)
print("Ljung-Box test p-value", lb.iloc[0, 1])


#plot the coefficients
importance = pd.DataFrame(zip(best_model[2].coef_.ravel().tolist(), x_train.columns.values.tolist()))
importance.columns = ['slope','feature_name']
importance_plot = sns.barplot(x=importance['feature_name'], y=importance['slope'], data=importance,orient='v',dodge=False,order=importance.sort_values('slope',ascending=False).feature_name)
for item in importance_plot.get_xticklabels(): #rotate the x labels by 90 degrees to avoid text overlapping
    item.set_rotation(90)
#plt.show()
plt.savefig(r'Results\%s.png' %("Coefficients"))


