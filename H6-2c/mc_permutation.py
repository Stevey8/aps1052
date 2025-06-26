#Bar and volume permutation program
#this program is based on Timothy Master's 
#Permutation and Randomization Tests pp. 31-37
#This permutation preserves the overall trend and
#the intra-day relationships of the bar's HLC to the O.
#It does NOT preserve the persistence in time of return volatility levels. 
#To use this permutation, see pseudo code at the bottom of the file 

import numpy as np
import pandas as pd

"""
the input df should have this format:

Index Open      High    Low     Close   Tickvol  Volume   Spread Date
0     4.998	4.998	4.968	4.979	etc.     etc.     etc.   etc.
1     4.965	4.970	4.939	4.939	etc.
2     4.941	4.952	4.921	4.941	etc.


where the Index above is the dataframe index (not a column)

The output df has the same format (but the predictors are permuted)
    
"""

def bar_permutation(df):

    df = df.fillna(0)
    df = df.replace(0, 1)
    
    df["Open"] = np.log(df.Open)
    df["High"] = np.log(df.High)
    df["Low"] = np.log(df.Low)
    df["Close"] = np.log(df.Close)
    df["Tickvol"] = np.abs(np.log(df.Tickvol))
    df["Volume"] = np.abs(np.log(df.Volume))
    df["Spread"] = np.abs(np.log(df.Spread))
    
    #get a subset (avoiding the first bar), starting from the second bar...
    df2 = df.iloc[1:,:].reset_index(drop=True)
    
    #relative inter-bar differences
    rel_open =  pd.DataFrame(df2.Open - df2.Close.shift(1))
    rel_tickvol = pd.DataFrame(df2.Tickvol - df2.Tickvol.shift(1))
    rel_volume = pd.DataFrame(df2.Volume - df2.Volume.shift(1))
    rel_spread = pd.DataFrame(df2.Spread - df2.Spread.shift(1))
    
    #relative intra-bar differences
    rel_high = pd.DataFrame(df2.High - df2.Open)
    rel_low =  pd.DataFrame(df2.Low - df2.Open)
    rel_close = pd.DataFrame(df2.Close - df2.Open)
    
    #fill in the missing Nan values using the first bar 
    rel_open.iloc[0:1] = df.Open.iloc[1:2].values[0]-df.Close.iloc[0:1].values[0]  #fill in the Nan
    rel_tickvol.iloc[0:1] = df.Tickvol.iloc[1:2].values[0]-df.Tickvol.iloc[0:1].values[0] #fill in the Nan
    rel_volume.iloc[0:1] = df.Volume.iloc[1:2].values[0]-df.Volume.iloc[0:1].values[0] #fill in the Nan
    rel_spread.iloc[0:1] = df.Spread.iloc[1:2].values[0]-df.Spread.iloc[0:1].values[0] #fill in the Nan
    
    #permute the index for intra-bar differences (high, low, close)
    permuted_idx = np.random.permutation(rel_high.index)
    
    #use the same index for the permutations of rel_high, rel_low and rel_close
    p_rel_high= rel_high.iloc[permuted_idx].reset_index(drop=True)
    p_rel_low = rel_low.iloc[permuted_idx].reset_index(drop=True)
    p_rel_close = rel_close.iloc[permuted_idx].reset_index(drop=True)
    
    #permute the index for the inter-bar differences (open and tickvolume, volume, spread)
    permuted_idx = np.random.permutation(rel_open.index)
    
    #use the same index for the permutations of rel_open, tickvol, rel_volume, rel_spread
    p_rel_open = rel_open.iloc[permuted_idx].reset_index(drop=True)
    p_rel_tickvol = rel_tickvol.iloc[permuted_idx].reset_index(drop=True)
    p_rel_volume = rel_volume.iloc[permuted_idx].reset_index(drop=True)
    p_rel_spread = rel_spread.iloc[permuted_idx].reset_index(drop=True)
    
    p_open = rel_open.copy()
    p_high = rel_open.copy()
    p_low = rel_open.copy()
    p_close = rel_open.copy()
    p_tickvol = rel_tickvol.copy()
    p_volume = rel_volume.copy()
    p_spread = rel_spread.copy()
    
    
    #reconstruct the new open, high, low, close levels, starting from the second bar
    for index, row in p_open.iterrows():
        if index == 0:
            prev = df.Close.iloc[0:1].values[0]
        else:
            prev = p_close[0].iloc[index-1]
        p_open[0].iloc[index] = prev + p_rel_open[0].iloc[index]
        p_high[0].iloc[index] = p_open[0].iloc[index] + p_rel_high[0].iloc[index]
        p_low[0].iloc[index] = p_open[0].iloc[index] + p_rel_low[0].iloc[index]
        p_close[0].iloc[index] = p_open[0].iloc[index] + p_rel_close[0].iloc[index]
    
    #reconstruct the new tickvolume levels, starting from the second bar
    for index, row in p_tickvol.iterrows():
        if index == 0:
            prev = df.Tickvol.iloc[0:1].values[0]
        else:
            prev = p_tickvol['Tickvol'].iloc[index-1]
        p_tickvol['Tickvol'].iloc[index] = prev + p_rel_tickvol['Tickvol'].iloc[index]
    
    #reconstruct the new volume levels, starting from the second bar
    for index, row in p_volume.iterrows():
        if index == 0:
            prev = df.Volume.iloc[0:1].values[0]
        else:
            prev = p_volume['Volume'].iloc[index-1]
        p_volume['Volume'].iloc[index] = prev + p_rel_volume['Volume'].iloc[index]
        
    #reconstruct the new spread levels, starting from the second bar
    for index, row in p_spread.iterrows():
        if index == 0:
            prev = df.Spread.iloc[0:1].values[0]
        else:
            prev = p_spread['Spread'].iloc[index-1]
        p_spread['Spread'].iloc[index] = prev + p_rel_spread['Spread'].iloc[index]
    
    
    #insert the reconstructed values at index 1 so we can later insert the first (original) bar at index 0
    p_df = df2.copy()
    p_df.index = np.arange(1, df2.shape[0]+1)
    p_df['Open'] = p_open.values.flatten()
    p_df['High'] = p_high.values.flatten()
    p_df['Low'] = p_low.values.flatten()
    p_df['Close'] = p_close.values.flatten()
    p_df['Tickvol'] = p_tickvol.values.flatten()
    p_df['Volume'] = p_volume.values.flatten()
    p_df['Spread'] = p_spread.values.flatten()
    
    #add the first bar at index 0 and resort the index
    new_row = [df.Open.iloc[0:1].values[0], df.High.iloc[0:1].values[0], df.Low.iloc[0:1].values[0], df.Close.iloc[0:1].values[0], df.Tickvol.iloc[0:1].values[0], df.Volume.iloc[0:1].values[0], df.Spread.iloc[0:1].values[0], df.Date.iloc[0:1].values[0], ]
    p_df.loc[0,:] = new_row
    p_df = p_df.sort_index()
    
    #reverse the log operation
    p_df['Open'] = np.abs(np.exp(p_df.Open))
    p_df['High'] =  np.abs(np.exp(p_df.High))
    p_df['Low']  =  np.abs(np.exp(p_df.Low))
    p_df['Close'] =  np.abs(np.exp(p_df.Close))
    p_df['Tickvol'] = np.abs(np.exp(p_df.Tickvol))
    p_df['Volume'] = np.abs(np.exp(p_df.Volume))
    p_df['Spread'] = np.abs(np.exp(p_df.Spread))
    
    return p_df
    


