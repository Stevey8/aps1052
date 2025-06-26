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

"""

"""
In this script you will learn how to apply Monte Carlo Permutation to input data
and use the permuted data in a loop to test a trained model.
The permutation function is already programmed for you.
"""
import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import fAux
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import mc_permutation

global global_returns
global global_labels


np.random.seed(2) #to fix the results
rs = 2
 
#file_path = 'outputfile.txt'
#saveMe = sys.stdout
#sys.stdout = open(file_path, "w")

#df = pd.read_csv('EURUSD_H3_200001030000_202107201800.csv', sep='\t')
#df = pd.read_csv('GBPUSD_H3_200001030000_202107201800.csv', sep='\t')
#df = pd.read_csv('NZDUSD_H3_200001030000_202107201800.csv', sep='\t')
df = pd.read_csv('USDCAD_H3_200001030000_202107201800.csv', sep='\t')
#df = pd.read_csv('USDCHF_H3_200001030000_202107201800.csv', sep='\t')

##########################################################################################################################
#Data preparation with original daa
df['<DATETIME>'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
df.drop(['<TIME>'], axis=1, inplace=True)
df.drop(['<DATE>'], axis=1, inplace=True)
df_hold_for_permute = df.iloc[0:10000]

df = df.set_index('<DATETIME>')

#save the close and open for white reality check
openp = df['<OPEN>'].copy() #for the case we want to enter trades at the open
#close = df['<CLOSE>'].copy() #for the case we want to enter trades at the close

#Feature engineering with original data
#buld the best window features after the exploratory data analysis:
for n in list(range(1,21)):
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

#select the samples
x_train = X.iloc[0:10000]

y_train = y.iloc[0:10000]

df_train = df_filtered.iloc[0:10000]

global_returns = df['<OPEN>'].pct_change(1).shift(-1).fillna(0)
global_labels = y


##########################################################################################################################

#set up the grid search and fit
from sklearn.model_selection import GridSearchCV
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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import FunctionTransformer
# import talib as ta
import ta
from ta.trend import EMAIndicator 
import detrendPrice 


def search_sequence_numpy(arr,seq):
    """ Find sequence in an array using NumPy only.

    Parameters
    ----------    
    arr    : input 1D array
    seq    : input 1D array

    Output
    ------    
    Output : 1D Array of indices in the input array that satisfy the 
    matching of input sequence in the input array.
    In case of no match, an empty list is returned.
    https://stackoverflow.com/questions/36522220/searching-a-sequence-in-a-numpy-array
    https://archive.is/GBCln
    """
    
    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() >0:
        return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
    else:
        return []         # No match found



def mean_return(y_true, y_pred):
    #this function has access to two global variables (global_labels and global_returns)
    #to help in the calculation of the model returns (multiplying mkt returns by positions derived from y_pred)
    #search_sequence_numpy searches for the index ix of the global_labels that correspond to the sequence of true lables (y_true)
    #those ix are then used to obtain the mkt returns that need to be multiplied by the positions to obtain the model returns

    ix = search_sequence_numpy(global_labels.values.flatten(), y_true) 
    mkt_returns = global_returns.values[ix] # alternative to global_returns.values[ix[0]:ix[0]+y_true.size]
    mkt_returns = mkt_returns.flatten()
    
    positions = np.where(y_pred> 0,1,-1 )
    positions = np.nan_to_num(positions, nan=0.0)
    dailyRet = positions * mkt_returns
    dailyRet = np.nan_to_num(dailyRet, nan=0.0)
    
    #val_model_returns.append(dailyRet)
    #mean_return = gmean(dailyRet+1)-1 #will use log returns instead beause...
    dailyRet = np.log(dailyRet+1)
    mean_return = np.mean(dailyRet) #GridSearchCV and RandomSearchCV use arithmetic mean
    return mean_return

def profit_ratio(y_true, y_pred):
    
    ix = search_sequence_numpy(global_labels.values.flatten(), y_true) 
    mkt_returns = global_returns.values[ix] # alternative to global_returns.values[ix[0]:ix[0]+y_true.size]
    mkt_returns = mkt_returns.flatten()
    
    positions_arr = np.where(y_pred> 0,1,-1 )
    positions_arr = np.nan_to_num(positions_arr, nan=0.0)
    dailyRet_arr = positions_arr * mkt_returns #calculate the daily returns of the system
    dailyRet_arr = np.nan_to_num(dailyRet_arr, nan=0.0)
    profits = np.where((dailyRet_arr >= 0), dailyRet_arr, 0)
    losses = np.where((dailyRet_arr < 0), dailyRet_arr, 0)
    profit_ratio = np.sum(profits)/np.sum(np.abs(losses))
    return profit_ratio


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

#PANDAS exponential smoothing:
def pandas_ewm_smoother(x_train, span=None):
    x_train = pd.DataFrame(x_train)
    x_train_smooth = x_train.ewm(span=span, adjust=True).mean()
    return  x_train_smooth.values

#Ta-lib exponential smoothing:
def talib_ewm_smoother(x_train, span=None):
    w = np.arange(x_train.shape[0])
    for i in range(0,x_train.shape[1]):
        # a = ta.EMA(x_train[:,i], timeperiod=span)
        a = EMAIndicator(close=x_train[:,i], window=span).ema_indicator().values
        w = np.c_[w,a]
    return w[:,1:]

myscorer = None #use default accuracy score
#myscorer = make_scorer(mean_return, greater_is_better=True)
#myscorer = make_scorer(profit_ratio, greater_is_better=True)
#myscorer = make_scorer(phi_k, greater_is_better=True)

#when using smoother, use TimesSeriesSplit
n_splits = 5
#split = 5 
#split = TimeSeriesSplit(n_splits=5, max_train_size=2000) #fixed size window
split = TimeSeriesSplit(n_splits=n_splits)


#smoother = FunctionTransformer(talib_ewm_smoother)
smoother = FunctionTransformer(pandas_ewm_smoother)

numeric_sub_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler()),
    ('smoother', smoother),
    ('imputer2', SimpleImputer(strategy='constant', fill_value=0))])
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


logistic = LogisticRegression(max_iter=1000, solver='liblinear') 

pipe = Pipeline(steps=[('preprocessor', preprocessor),('logistic', logistic)])

c_rs = np.logspace(3, -4, num=20, endpoint = True)
#penalty type=L2 like ridge regression (small coefficients preferred), L1 like lasso  (coefficients can become zero)
p_rs= ["l1", "l2"]
spans_rs = [{'span': 2},{'span': 3},{'span': 4},{'span': 5},{'span': 6},{'span': 7},{'span': 8},{'span': 9},{'span': 10},{'span': 11}, {'span': 12},{'span': 13},{'span': 14},{'span': 15},{'span': 16},{'span': 17},{'span': 18},{'span': 19},{'span': 20}]

param_grid =  [{'preprocessor__num__smoother__kw_args':  spans_rs, 'logistic__C': c_rs, 'logistic__penalty': p_rs}]

grid_search = RandomizedSearchCV(pipe, param_grid, cv=split, scoring=myscorer, return_train_score=True, random_state=rs)
#grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring=myscorer, return_train_score=True)

grid_search.fit(x_train, y_train.values.ravel())

best_parameters = grid_search.best_params_
best_model = grid_search.best_estimator_


print("Best parameters : {}".format(best_parameters))
#print('Best estimator {}'.format(best_model))
print("Best cross-validation score : {:.2f}".format(grid_search.best_score_*100))
results = pd.DataFrame(grid_search.cv_results_)

#print(results.T)
results.to_csv("results_logisticreg.csv")


#########################################################################################################################

# Train set
# Make "predictions" on training set (in-sample)
#positions = np.where(best_model.predict(x_train)> 0,1,-1 )
positions = np.where(grid_search.predict(x_train)> 0,1,-1 ) #POSITIONS

#dailyRet = pd.Series(positions).shift(1).fillna(0).values * x_train.ret1 #for trading at the close
dailyRet = pd.Series(positions).fillna(0).values * df_train.retFut1 #for trading right after the open

dailyRet = dailyRet.fillna(0)

cumret = np.cumprod(dailyRet + 1) - 1


plt.figure(1)
plt.plot(cumret.index, cumret)
plt.title('Cross-validated LogisticRegression on currency: train set')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
#plt.show()
plt.savefig(r'Results\%s.png' %("TrainCumulative"))

    
cagr = (1 + cumret[-1]) ** (252 / len(cumret)) - 1
maxDD, maxDDD = fAux.calculateMaxDD(cumret)
s_ratio = (252.0 ** (1.0/2.0)) * np.mean(dailyRet) / np.std(dailyRet)
profits = np.where((dailyRet >= 0), dailyRet, 0)
losses = np.where((dailyRet < 0), dailyRet, 0)
p_ratio = np.sum(profits)/np.sum(np.abs(losses))
print (('In-sample: CAGR={:0.6} Sharpe ratio={:0.6} Profit ratio={:0.6} maxDD={:0.6} maxDDD={:d} Calmar ratio={:0.6}\n'\
).format(cagr, s_ratio, p_ratio, maxDD, maxDDD.astype(int), -cagr/maxDD))


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

#monte-carlo relications
    
    
count = 0
loop_max = 3

for i in range(1,loop_max):
    
    """
    INSTRUCTIONS
    apply the permutation function to the newdf
    Note that the input data is permuted BEFORE any feature engineering takes place.
    THIS ORDER IS IMPORTANT.
    You cannot permute certain engineered features 
    (like overlapping n-period returns) because they are autocorrelated (serially correlated).
    PERMUTING HIGHLY SERIALLY CORRELATED FEATURES VIOLATES THE ASSUMPTIONS OF THE MONTE CARLO TEST.
    See p. 13 of Permutation and Randomization Tests for Trading System Development by Timothy Masters
    Note that during feature engineering, the target is never permuted
    """
    
    #permute the bar data, change the names of the columns
    newdf = pd.DataFrame(df_hold_for_permute.values, columns=['Open','High','Low','Close','Tickvol','Volume','Spread','Date'])
    # permute the data
    # o = ######
    o = mc_permutation.bar_permutation(newdf)    
    perm_df = pd.DataFrame(o.values, columns=['<OPEN>','<HIGH>','<LOW>','<CLOSE>','<TICKVOL>','<VOL>','<SPREAD>','<DATETIME>'])
    
    #data preparation with permuted data
    perm_df = perm_df.set_index('<DATETIME>')
    
    #feature engineering with permuted data
    #buld the best window features:
    for n in list(range(1,21)):
        name = 'ret' + str(n)
        perm_df[name] = perm_df["<OPEN>"].pct_change(periods=n) #for trading with open
    
    
    #build date-time features
    perm_df["hour"] = perm_df.index.hour.values
    perm_df["day"] = perm_df.index.dayofweek.values
    
    #build target assuming we know today's open (use df['<OPEN>'] for target, not perm_df['<OPEN>']; do not permute target)
    perm_df['retFut1'] = df['<OPEN>'].pct_change(1).shift(-1).fillna(0) #if you enter the trade immediately after the open
    
    #transform the target (retFut1 was not permuted)
    perm_df['retFut1_categ'] = np.where((perm_df['retFut1'] > 0), 1, 0)
    
    #Since we are trading right after the open, 
    #we only know yesterday's  high low close volume spread etc.
    perm_df['<HIGH>'] = perm_df['<HIGH>'].shift(1)
    perm_df['<LOW>'] = perm_df['<LOW>'].shift(1)
    perm_df['<CLOSE>'] = perm_df['<CLOSE>'].shift(1)
    perm_df['<VOL>'] = perm_df['<VOL>'].shift(1)
    perm_df['<SPREAD>'] = perm_df['<SPREAD>'].shift(1)
    
    #select the features (by dropping)
    cols_to_drop = ["<OPEN>","<HIGH>","<LOW>","<CLOSE>","<TICKVOL>","<VOL>","<SPREAD>"]  #optional
    perm_df_filtered = perm_df.drop(cols_to_drop, axis=1)
    
    #distribute the df data into perm_X inputs and y target (no permutation for y)
    perm_X = perm_df_filtered.drop(['retFut1', 'retFut1_categ'], axis=1) 
    y = df_filtered[['retFut1_categ']]
    
    #select the samples 
    perm_x_train = perm_X.iloc[0:10000] #perm_x_train has permuted data
    
    y_train = y.iloc[0:10000] #y_train has original y_train
    
    perm_df_train = perm_df_filtered.iloc[0:10000] #perm_df_train.retFut1 was not permuted, but everything else was permuted
    
    ########################################################################################################################
    
    # Train set with permuted data
    # Make "predictions" on permuted training set (in-sample)
    
    """
    INSTRUCTIONS
    Use the permuted x_train data within the trained model to predict and calculate the positions
    Fill in the missing #### with loop_max equal to 3.
    Make sure everything works.
    Then go back and change loop_max to 100
    Run the script with loop_max = 100 and make sure that 
    "probability of selection bias" < .11
    """
    # perm_positions = #####
    perm_positions = np.where(grid_search.predict(perm_x_train)> 0,1,-1 ) 
    
    # perm_dailyRet = ##### #perm_df_train.retFut1 was not permuted
    perm_dailyRet = pd.Series(perm_positions).fillna(0).values * perm_df_train.retFut1
    
    perm_dailyRet = perm_dailyRet.fillna(0)
    
    perm_cumret = np.cumprod(perm_dailyRet + 1) - 1
    
    
    plt.figure(i+1)
    plt.plot(perm_cumret.index, cumret)
    plt.title('Cross-validated LogisticRegression on currency: perm train set')
    plt.ylabel('Cumulative Returns')
    plt.xlabel('Date')
    plt.show()
    #plt.savefig(r'Results\%s.png' %("TrainCumulative"))
    
        
    perm_cagr = (1 + perm_cumret[-1]) ** (252 / len(perm_cumret)) - 1
    perm_maxDD, perm_maxDDD = fAux.calculateMaxDD(perm_cumret)
    perm_s_ratio = (252.0 ** (1.0/2.0)) * np.mean(perm_dailyRet) / np.std(perm_dailyRet)
    perm_profits = np.where((perm_dailyRet >= 0), perm_dailyRet, 0)
    perm_losses = np.where((perm_dailyRet < 0), perm_dailyRet, 0)
    perm_p_ratio = np.sum(perm_profits)/np.sum(np.abs(perm_losses))
    print (('In-sample_perm: CAGR={:0.6} Sharpe ratio={:0.6} Profit ratio={:0.6} maxDD={:0.6} maxDDD={:d} Calmar ratio={:0.6}\n'\
    ).format(perm_cagr, perm_s_ratio, perm_p_ratio, perm_maxDD, perm_maxDDD.astype(int), -perm_cagr/perm_maxDD))
        
    if perm_p_ratio > p_ratio:
        count = count+1
        
    print("iteration:", i)    
    print("p_ratio", p_ratio)
    print("perm_p_ratio", perm_p_ratio)
    print("num times criterion on permuted data > original criterion:", count)
    print("probability of selection bias",((count+1)/(loop_max+1)))