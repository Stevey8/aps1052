

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#seaborn.mpl.rcParams['figure.figsize'] = (10.0, 6.0)
#seaborn.mpl.rcParams['savefig.dpi'] = 90
'''
https://people.duke.edu/~ccc14/sta-663/ResamplingAndMonteCarloSimulations.html
This subroutine will calculate White's Reality Check for a single trading rule
in accordance with Aronson's Evidence Based Technical Analysis p.237ff

'''


#returns must be detrended by subtracting the average daily return of the benchmark
def bootstrap(ser):
    ser.dropna(inplace=True)
    ser = np.log(ser+1)
    arr = np.array(ser.values)
    alpha = .05*100 #significance alpha
    reps = 5000 #how many bootstrapings, 50000 limit if you have 8GB RAM
    
    percentile = 100-alpha
    ave = np.average(arr) #arithmetic mean
    
    print("average return %f" %ave)
    
    centered_arr = arr-ave
    n = len(centered_arr)
    #constructs 50000 alternative return histories and calculates their theoretical averages
    xb = np.random.choice(centered_arr, (n, reps), replace=True)
    mb = xb.mean(axis=0) #arithmetic mean
    
    #sorts the 50000 averages
    mb.sort()
    #calculates the 95% conficence interval (two tails) threshold for the theoretical averages
    print("uncorrected CI (log-returns, daily): ", np.percentile(mb, [2.5, 97.5])) 
    threshold = np.percentile(mb, [percentile])[0] #un corrected confidence interval, rough estimate
    
    
    if ave > threshold:
        print("Reject Ho = The population distribution of rule returns has an expected value of zero or less (because p_value is small enough)")
    else:
        print("Do not reject Ho = The population distribution of rule returns has an expected value of zero or less (because p_value is not small enough)")
    
    #count_vals will be the items i that are smaller than ave
    count_vals = 0
    for i in mb:
        count_vals += 1
        if i > ave:
            break
     
    larger = len(mb) - count_vals #larger will be items i that are larger than ave
    p = larger/len(mb)
    #p = 1-count_vals/len(mb)
    
    print("p_value:")
    print(p)
    
    
     
    #histogram
    sr = pd.Series(mb)
    desc = sr.describe()
    count = desc[0]
    std = desc[2]
    minim = desc[3]
    maxim = desc[7]
    R = maxim-minim
    c = count
    s = std
    bins = int(round(R*(c**(1/3))/(3.49*std),0))
    fig = sr.hist(bins=bins)
    plt.axvline(x = ave, color = 'b', label = 'axvline - full height')
    # plt.show()
    
     #about the histogram
     #https://stackoverflow.com/questions/33458566/how-to-choose-bins-in-matplotlib-histogram
     #R(c^(1/3))/(3.49σ)
     #R is the range of data (in your case R = 3-(-3)= 6),
     #c is the number of samples,
     #σ is your standard deviation.
    
    
    #this CI_bias_corrected_accelerated_bootstrap is taken from TimothyMaster's Permutation and Randomization Tests pp.104-114
    #this bias correction compensates for the fact that the parameter estimates from the bootstrap
    #under- and over- estimate the true parameter (the mean return)
    
    def CI_bias_corrected_accelerated_bootstrap(accel, count_vals, mb, low_conf, high_conf):
        from scipy.stats import norm
        zo = norm.ppf(count_vals/len(mb))
        zlo = norm.ppf(low_conf) #ppf = percent point function = quantile function = inverse normal cdf
        zhi = norm.ppf(high_conf)
        alo = norm.cdf(zo + (zo + zlo)/(1-accel*(zo+zlo)))
        ahi = norm.cdf(zo + (zo + zhi)/(1-accel*(zo+zhi)))
        k = int(np.trunc(((alo*(len(mb)+1))-1)))
        if k < 0:
            k = 0
        
        annual = 252
        annual = 1
        low_bound = mb[k]*annual #annualized low_bound, 252 annualization factor
        
        k = int(np.trunc(((1-ahi)*(len(mb)+1)) -1))
        if k < 0:
            k = 0
        high_bound = mb[len(mb)-1-k]*annual #annualized high_bound, 252 annualization factor
        return low_bound, high_bound
    
    
    def jacknife_mean(returns):
        # Leave one observation out from the returns to get the jackknife sample and store the jk_sample_means
        jk_sample_means= []
        index = np.arange(len(returns))
        for i in range(n):
            jk_sample = returns[index != i]
            jk_sample_means.append(np.mean(jk_sample))
        
        # The jackknife estimate is the mean of the jk_sample_means from each sample
        jk_sample_means = np.array(jk_sample_means)
        jk_sample_mean_of_means = np.mean(jk_sample_means)
        #print("Jackknife estimate of the mean = {}".format(jk_sample_mean_of_means))
        return jk_sample_means, jk_sample_mean_of_means
    
    def calculate_acceleration(jk_sample_means, jk_sample_mean_of_means):
        jk_diffs = jk_sample_mean_of_means - jk_sample_means
        numerator = np.sum(np.power(jk_diffs,3))
        denominator = 6*np.power(np.sum(np.power(jk_diffs,2)),3/2)
        acceleration = numerator/denominator
        return acceleration
    
    
    returns = arr.copy()
    jk_sample_means, jk_sample_mean_of_means = jacknife_mean(returns)
    accel = calculate_acceleration(jk_sample_means,jk_sample_mean_of_means)
    low_conf = 0.1 #should be lower than 0.5
    high_conf = 0.9 #should be higher than 0.5
    low_bound, high_bound =  CI_bias_corrected_accelerated_bootstrap(accel, count_vals, mb, low_conf, high_conf)
    print("What follows is the bias corrected accelerated bootstrap confidence interval (log-returns, daily).")
    print("This bias correction compensates for the fact that ")
    print("the parameter estimates from the bootstrap under- and over- estimate the true parameter (the mean return). ")
    print("Low bound, there is a 90% chance that our return is at least: ", low_bound)
    print("High bound, there is a 90% chance that our return is at most: ", high_bound)








