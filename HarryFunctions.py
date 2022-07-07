# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 15:48:08 2022

@author: htm516

A function library to house functions I am using over many scripts

Funciton list

hf_find,cmvir,hf_ecg,clean_peaks,fill_peaks,remove_peaks,find_local_peaks,calc_all_hrv,viz_all_HRV,hrSQI,cp_plot
"""
#%% Load libraries

import csv
from   datetime import datetime
import fnmatch
import heartpy as hp
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
from   matplotlib import gridspec
import math
import neurokit2 as nk 
import numpy as np
import os
import pandas as pd
import pingouin as pg 
#from   PyEMD import EMD
import pylab
import re
import ruptures as rpt
import scipy
from   scipy.interpolate import interp1d,UnivariateSpline, interp2d
import scipy.io as sio
from   scipy.ndimage import median_filter
from   scipy.signal import (welch, find_peaks)
import scipy.stats as spyst
import seaborn as sns
from   sklearn.decomposition import PCA
import statsmodels.api as sm
from   statsmodels.formula.api import glm
from   statsmodels.multivariate.manova import MANOVA
from   statsmodels.stats.diagnostic import acorr_ljungbox
from   statsmodels.stats.multicomp import pairwise_tukeyhsd
import sys

# rpy2
os.environ['R_HOME'] = "C://Users//htm516//Anaconda3//envs//rstudio//lib//R"

# Load packages
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri

# Function to convert an R object to a Python dictionary
def robj_to_dict(robj):
    return dict(zip(robj.names, map(list, robj)))

def robj_from_df(df):
    with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
        r_from_df = ro.conversion.py2rpy(df)
    return r_from_df

cpt    = importr('changepoint')
cptnp  = importr('changepoint.np')
mgcv   = importr('mgcv')
statsr = importr('stats')

# Order of functions
#
# hf_find(pattern, path)
# cmvir(N)
# viz_ECG_mini(ax,ecg_sig, peaks_new,time,sr=500, NZoom=[], Title = "Filtered Signal",ylab="",col="plum")
# hf_ecg(ecg,sr=500,freqfilt=[0.5,20])
# clean_peaks(r_peaks, ecg_sig, sr=500):
# fill_peaks(r_peaks,ecg_sig,sr=500):
# remove_peaks(r_peaks,sr=500):
# find_local_peaks(r_peaks,ecg_sig,width=5, f_twidth = False, sr=500):
# hf_cp(signal, cp_type='cptnp',time=[]):
# calc_all_hrv(r_pks, ecg_sig,  sr=500, f_freq=True, f_nonlin=False,t_width=5):
# viz_all_HRV(stats, Title="",NZoom=[]):
# hrSQI(hr,sr=sr):
# cp_plot(signal,bkps,time=[],NZoom=[],title='',ylab=''):
# hf_goodness(true_peaks, test_peaks, sig, margin=0, f_roc = False):
# interp_vec(data,time_data,time_out,S_kind='nearest',S_fill='extrapolate'):

#%% dummy variables
sr=500

#%% Search for files in a folder
def hf_find(pattern, path):
    # A function to search folders for tiles which match a certain string
    
    result = []
    for root,dirs,  files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

#%% Viridis function
def cmvir(N):
    # A function to generate a colourblind-friendly set of colours
    
    out = [mplcm.viridis(i) for i in np.linspace(0, 0.9, N)] 
    out = out[::3]+out[1::3]+out[2::3] #give higher contrast order
    return out

#%% Mini Plot Function
    
def viz_ECG_mini(ax,ecg_sig, peaks_new,time,sr=500, NZoom=[], Title = "Filtered Signal",xlab="",ylab="",col="plum"):
    # A simple plotting function, that can plot peaks and signal over custom width, plot color, labels, etc.
    
    if len(NZoom)==0:
        NZoom = [0,max(time)]
    plt.plot(time,ecg_sig, label = 'ECG signal', color=col,linewidth=2)
    plt.scatter(time[peaks_new],ecg_sig[peaks_new], color='black', marker='x',label = 'identified peaks',zorder=10)
#plt.legend()
    try:
        plt.ylim([min(ecg_sig[(time>NZoom[0]) & (time<NZoom[1])])-abs(min(ecg_sig[(time>NZoom[0]) & (time<NZoom[1])]))*0.1,max(ecg_sig[(time>NZoom[0]) & (time<NZoom[1])])+abs(max(ecg_sig[(time>NZoom[0]) & (time<NZoom[1])]))*0.1])
    except:
        print("Warning: Limits of signal not found")
    ax.set_xlim(NZoom)
    ax.set_title(Title)
    ax.set_xlabel(xlab,fontsize=22)
    ax.set_ylabel(ylab,fontsize=22)

#%% Current ECG processing pipeline

def hf_ecg(ecg,sr=500,freqfilt=[0.5,20]):
    # Default - carry out a bandpass filter of 0.5 - 20 Hz on the signal
    
    ecg_out = nk.ecg_clean(ecg, sampling_rate=sr, method="neurokit")    
    if len(freqfilt)==1:  #assume removing low freq noise
        sos = scipy.signal.butter(5, freqfilt, btype='highpass', output="sos", fs=sr)
    elif len(freqfilt)==2:
        sos = scipy.signal.butter(5, freqfilt, btype='bandpass', output="sos", fs=sr)
    ecg_out = scipy.signal.sosfiltfilt(sos, ecg_out)
    #b = np.ones(max([int(sr / 50),2])) # filter with moving average kernel width of 50Hz (needs at least 100Hz sr)
    #a = [len(b)]
    #print(a,b)
    #ecg_out = scipy.signal.filtfilt(b, a, ecg_out, method="pad")
    #add this commented out bit if it's v. necessary to remove 50Hz for zome reason
    return ecg_out


#%% Clean peaks function

def clean_peaks(r_peaks, ecg_sig, sr=500):
    # Overall function to clean peaks, including adding missed peaks, removing false peaks, and local correction to account for preprocessing
    
    r_out=[int(x) for x in r_peaks]
    
    # peak list and sampling rate as inputs
    print('Started with %d peaks' % len(r_out))
    
    # clean all current peaks                
    r_out=find_local_peaks(r_out,ecg_sig, width=0.02, f_twidth = True, sr=500)
    
    # find bonus peaks (cleans peaks while placing them)
    r_out = fill_peaks(r_out,ecg_sig)
    
    # then remove excess peaks
    r_out = remove_peaks(r_out)
    
    r_out = [int(x) for x in r_out]
    
    return(r_out)


#%% Clean peaks subfunction 1 - fill in missing peaks
def fill_peaks(r_peaks,ecg_sig,sr=500):
    # Function to fill in peaks that may have been missed
    N_in = len(r_peaks)
    r_out=np.copy(r_peaks)      # Make a copy that can be changed
    r_old=np.asarray([])
    x=0
    while len(r_out) > len(r_old) and x<10:
        r_old = np.copy(r_out)
        ibi = np.diff(r_out)/sr     # Work out the timing difference between peaks
        ibi_mid = median_filter(ibi,size=21, mode="nearest") # creates an array the length of ibi, where each entrant is the median over the specified window size (was 11)
        fill = ibi/ibi_mid          # ratio of gap to local median gap
        peaks_to_add = []
        for ind in range(len(ibi)):
            # Npeaks = len(r_out)
            if fill[ind]>1.6:       # if the gap is 1.3x large than local median
                #print(r_out[ind],r_out[ind+1],np.median(ecg_sig[r_out[ind]:r_out[ind+1]]))
                if np.median(ecg_sig[r_out[ind]:r_out[ind+1]])!=0:    # if ecg is 0 in the gap, don't fill in peaks
                    if ind < len(ibi) and ind > 0 and (fill[ind]+fill[ind+1]<2):  # if this gap and the next are big together, just shift the current peak
                        r_out[ind] = find_local_peaks([int(np.mean([r_out[ind-1],r_out[ind+1]]))],ecg_sig,width=0.01, f_twidth = True, sr=sr)
                    elif ind < len(ibi) and ind > 0 and (fill[ind]+fill[ind-1]<2):
                        r_out[ind] = find_local_peaks([int(np.mean([r_out[ind-1],r_out[ind+1]]))],ecg_sig,width=0.01, f_twidth = True, sr=sr)
                    else:
                        peaks_to_add.extend(np.linspace(r_out[ind],r_out[ind+1],int(fill[ind]+2))[1:-1])  # add peaks! 1.3-1.7 = 1 peak, 1.7-2.7 = 2 peaks, etc
        peaks_to_add=[int(x) for x in np.round(peaks_to_add)]
        peaks_to_add=find_local_peaks(peaks_to_add,ecg_sig,width=0.01, f_twidth = True, sr=sr) #Look for best peak 0.1s around centre
        try:    r_out.extend(peaks_to_add)
        except: r_out = np.append(r_out, peaks_to_add)
        r_out = [int(x) for x in r_out]
        r_out = np.unique(r_out)
        x+=1
    r_out = np.unique(r_out)
    r_out.sort()
    print("Added %d peaks over %d cycles" % (len(r_out)-N_in,x))
    return(r_out)
    
#%% Clean peaks subfunction 2 - remove excess peaks  
def remove_peaks(r_peaks,sr=500):
    # Function to remove peaks that have been incorrectly added
    N_in = len(r_peaks)
    r_out=np.copy(r_peaks)      # Make a copy that can be changed
    ibi = np.diff(r_out)/sr     # Work out the timing difference between peaks
    ibi_mid = median_filter(ibi,size=11, mode="nearest") # creates an array the length of ibi, where each entrant is the median over the specified window size# was np.median(ibi)
    gap_2 = (np.asarray(r_out[2:])-np.asarray(r_out[:-2]))/(ibi_mid[:-1]*sr) # works out the gap between point n and point n+2
    while(np.min(gap_2)<1.5):   # while a gap exists petween points n and n+2 that is less than 1.3 * the local median
          for ind in reversed(range(len(gap_2))):
              if gap_2[ind]<1.5:
                  r_out=np.delete(r_out,ind) # remove offending index
                  ibi = np.diff(r_out)/sr
                  ibi_mid = median_filter(ibi,size=11, mode="nearest") 
                  gap_2 = (np.asarray(r_out[2:])-np.asarray(r_out[:-2]))/(ibi_mid[:-1]*sr)
    ibi = np.diff(r_out)/sr
    ibi_mid = median_filter(ibi,size=41, mode="nearest")
    gap_1 = ibi/ibi_mid
    while(np.min(gap_1)<0.5):
        for ind in reversed(range(len(gap_1))):
              if gap_1[ind]<0.5:
                  r_out=np.delete(r_out,ind)
                  ibi = np.diff(r_out)/sr
                  ibi_mid = median_filter(ibi,size=41, mode="nearest")
                  gap_1 = ibi/ibi_mid
    print("Removed %d peaks" % (N_in-len(r_out)))
    return(r_out)
    
#%% Clean peaks subfunction 3 - fill the local peak within a given region
def find_local_peaks(r_peaks,ecg_sig,width=5, f_twidth = False, sr=500):
    # Function to correct peak identified in preprocessing by looking for a local peak on the raw signal
    # f_twidth indicates that the width indicated is the time range, rather than the number of indicies
    
    if f_twidth:
        width = int(np.ceil(width*sr)) #Round up, ensure a width of at least 1
    
    r_out = np.copy(r_peaks)
    for x in np.arange(0,len(r_peaks)):  #len(Peaks)
        r_new=r_peaks[x]
        r_old=r_new+1   #just choosing some other value
        while r_old!=r_new:
            r_old=r_new
            min_i = np.max([r_old-width,0])
            max_i = np.min([r_old+width+1,len(ecg_sig)])
            miniSearch = ecg_sig[min_i:max_i]
            #print(min_i,max_i)
            maxLoc = np.where(miniSearch==np.amax(miniSearch))
            
            r_new += maxLoc[0][0]-width
            if r_new>len(ecg_sig):  r_new=len(ecg_sig)
            elif r_new<0:           r_new=0
        r_out[x]=r_new
    r_out = np.unique(r_out)
    return(r_out)

#%% Changepoint Function
def hf_cp(signal, cp_type='cptnp',time=[]):
    # Function to calculate changepoint, providing both Python and R libraries options
    # cp_type = cptnp, cptmean, cptvar, cptmeanvar, linear, l2, and l1.
    
    # R packages 
    if cp_type=='cptnp': #non-parametric
        bkps = cptnp.cpt_np(FloatVector(signal),penalty="MBIC" ,nquantiles =33,pen_value=0,minseglen=1) 
        bkps = list(cpt.cpts(bkps))
    elif cp_type=='cptmean':
        bkps = cpt.cpt_mean(FloatVector(signal), method = "PELT")
        bkps = list(cpt.cpts(bkps))
    elif cp_type=='cptvar':
        bkps = cpt.cpt_var(FloatVector(signal), method = "PELT")
        bkps = list(cpt.cpts(bkps))
    elif cp_type=='cptmeanvar':
        bkps = cpt.cpt_meanvar(FloatVector(signal), method = "PELT")
        bkps = list(cpt.cpts(bkps))
        
        # Rupture Python packages
    elif cp_type=='linear':
        X = np.vstack((time,np.ones(len(time)))).T 
        Y = np.column_stack((signal.reshape(-1, 1), X))
        algo = rpt.Pelt(model="linear",jump=1).fit(Y)
        bkps = algo.predict(pen=np.log(len(signal)) * np.std(signal) ** 2)
    elif cp_type=='mean' or cp_type=='l2':
        algo = rpt.Pelt(model="l2",jump=1).fit(signal)
        bkps = algo.predict(pen=np.log(len(signal)) * np.std(signal) ** 2)
    elif cp_type=='median' or cp_type=='l1':
        algo = rpt.Pelt(model="l1",jump=1).fit(signal)
        bkps = algo.predict(pen=np.log(len(signal)) * np.std(signal) ** 2)
    else:
        print('Please input a valid choice of changepoint type for cp_type.\n Current options are: cptnp, cptmean, cptvar, cptmeanvar, linear, l2, and l1.')
        bkps=[]
    return bkps



#%% Calculate statistics function

def calc_all_hrv(r_pks, ecg_sig,  sr=500, f_freq=True, f_nonlin=False,t_width=5):
    # Function to calculate various local HRV statistics, and corresponding time indices
    
    # Make peaks a numpy array
    r_pks = np.asarray(r_pks)
    
    # Calculate overall values
    Nsample = len(ecg_sig)
    Ntime = Nsample/sr
    
    #Calculate looping parameters for time/frequency loops
    t_jump   = 1
    f_width = 300
    f_jump   = 5
    
    # Calculate instantaneous values
    out={'ecg_r_pks':r_pks, 'sampling_rate':sr,'ecg':ecg_sig}
    out['hr']=60/np.diff(r_pks/sr)
    out['ibi']=np.diff(r_pks/sr)
    out['time_0']=(r_pks[:-1]+r_pks[1:])/sr/2
    out['time_5']=np.arange(0,np.ceil((Ntime-t_width)/t_jump))*t_jump+t_width/2
    out['time_300']=np.arange(0,(np.ceil(Ntime-f_width+f_jump)/f_jump))*f_jump+f_width/2
    
    

    out_5       = pd.DataFrame()
    #print(Ntime,window_time, loop_time, np.ceil((Ntime-window_time+loop_time)/loop_time)-1)
    for x in np.arange(0,np.ceil((Ntime-t_width+t_jump)/t_jump)-1):
        start_i   = x*t_jump*sr
        end_i     = np.min([(x*t_jump+t_width), Ntime])*sr
        loop_pks  = [y for y in r_pks if y>=start_i and y<end_i] 
        try:
            results   = nk.hrv_time(loop_pks, sampling_rate=sr) 
        except:
            try:
                for col in results.columns:
                    results[col].values[:] = results[col].values[:]*0
                print('error in time HRV at %d' % (x))
            except:
                out['time_5'] = out['time_5'][1:]
                continue

        # If participant is not the first, then append the data 
        if out_5.empty: out_5 = results
        else:           out_5 = pd.concat([out_5,results],ignore_index=True)
        
        if not np.mod(x,100):
            print("Time HRV, %d loops calculated" % (x))
    
    out.update(out_5.to_dict(orient="list"))
    
    if f_freq:
        #Calculate freq and non-linear time values
        out_300     = pd.DataFrame()
        for x in np.arange(0,np.ceil((Ntime-f_width+f_jump)/f_jump)):
            start_i   = x*f_jump*sr
            end_i     = np.min([(x*f_jump+f_width), Ntime])*sr
            loop_pks  = [y for y in r_pks if y>=start_i and y<end_i] 
            try:
                results_f   = nk.hrv_frequency(loop_pks, sampling_rate=sr) 
            except:
                try:
                    for col in results_f.columns:
                        results_f[col].values[:] = results_f[col].values[:]*0
                except:
                    out['time_300'] = out['time_300'][1:] #shrink by one
                    continue
            if f_nonlin:
                results   = results.join(nk.hrv_nonlinear(loop_pks, sampling_rate=sr))
    
            # If out_300 isn't empty, then append the data 
            if out_300.empty:   out_300 = results_f
            else:               out_300 = pd.concat([out_300,results_f],ignore_index=True)
            
            if not np.mod(x,100):
                print("Freq HRV, %d loops calculated" % (x))
        
        out.update(out_300.to_dict(orient="list"))
        
    return out



#%% Visualise HRV statistics

def viz_all_HRV(stats, Title="",NZoom=[]):
    # Function to plot various Heart Rate Variability statistics
    
    if not NZoom:
        NZoom = [0,np.amax(stats['time_0'])]
    
    fig = plt.figure(figsize=(16,12))
    ax1 = plt.subplot(421)
    
    plt.plot(stats['time_0'],(stats['hr']), label = 'Heart Rate', color='red',linewidth=2)
    plt.legend()
    ax1.set_xlim(NZoom)
    
    ax1 = plt.subplot(422)
    plt.plot(stats['time_0'],stats['ibi'], label = 'Inter-beat Intervals', color='purple',linewidth=2)
    plt.legend()
    ax1.set_xlim(NZoom)
    
    #SDNN (also equivalent to SD2)
    ax1 = plt.subplot(423)
    plt.plot(stats['time_5'],(stats['HRV_SDNN']), label = 'SDNN', color='plum',linewidth=2)
    plt.legend()
    ax1.set_xlim(NZoom)
    
    #RMSSD (also equivalent to SD1*sqrt(2))
    ax1 = plt.subplot(424)
    plt.plot(stats['time_5'],(stats['HRV_RMSSD']), label = 'RMSSD', color='black',linewidth=2)
    plt.legend()
    ax1.set_xlim(NZoom)
    
    ax1 = plt.subplot(425)
    plt.plot(stats['time_5'],(stats['HRV_pNN20']), label = 'pNN20', color='blue',linewidth=2)
    plt.legend()
    ax1.set_xlim(NZoom)
    
    ax1 = plt.subplot(426)
    plt.plot(stats['time_5'],(stats['HRV_IQRNN']), label = 'IQR NN', color='green',linewidth=2)
    plt.legend()
    ax1.set_xlim(NZoom)
    
    ax1 = plt.subplot(427)
    plt.plot(stats['time_300'],(stats['HRV_LF']), label = 'LF', color='orange',linewidth=2)
    plt.legend()
    ax1.set_xlim(NZoom)
    
    ax1 = plt.subplot(428)
    plt.plot(stats['time_300'],(stats['HRV_HF']), label = 'HF', color='navy',linewidth=2)
    plt.legend()
    ax1.set_xlim(NZoom)
    
    fig.suptitle(Title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)

#%% With better filter for interpolation (only use good values)

def hrSQI(hr,sr=sr):
    # Function to clean up the heart rate signal by interpolating through values that fall outside a local median
    Nfilt=41  # width of filter (e.g. Nfilt = 3 would include x-1, x, x+1)
    hr_med = median_filter(hr,size=Nfilt,mode='nearest')
    good_fact=1.2
    hr_med_low = hr_med/good_fact
    hr_med_up  = hr_med*good_fact
    Nhr   = len(hr)
    
    good_hr = (hr>hr_med_low) & (hr<hr_med_up)

    qual_hr= np.zeros((Nhr,2))
    new_hr  = np.copy(hr)
    
    
    for ind,x in enumerate(hr):
        
        # Number of good hr_indices inside proposed median filter
        qual_hr[ind,1] = np.sum(good_hr[max(0,int(ind-Nfilt/2)):min(Nhr,int(ind+Nfilt/2))])/len(good_hr[max(0,int(ind-Nfilt/2)):min(Nhr,int(ind+Nfilt/2+Nfilt%2))])
        
        if not good_hr[ind]:
            # Distance beyond the boundary at this index
            qual_hr[ind,0]=max((hr_med_low[ind]-x)/hr_med_low[ind],(x-hr_med_up[ind])/hr_med_up[ind])
            
            # if qual_hr[ind,1]<0.5:
            #     new_hr[ind]=0
            # else:
            
            start_i   = int(np.max([0,ind-20]))
            end_i     = int(np.min([ind+20,Nhr]))
            
            ind_lr = np.concatenate((np.arange(start_i,ind),np.arange(ind+1,end_i)))
            
            ind_lr = ind_lr[good_hr[ind_lr]]
            if len(ind_lr)==1:
                #print(ind_lr)
                new_hr[ind] = hr[ind_lr]
            elif len(ind_lr)==0:
                new_hr[ind] = hr[ind] # redundant, but clearly stating that a value is kept the same if nothing can replace it
            else:
                f=interp1d(ind_lr,hr[ind_lr],kind='slinear',fill_value="extrapolate")
                new_hr[ind] = f(ind)
            
    return new_hr, qual_hr



#%% Custom Changepoint Plot

def cp_plot(signal,bkps,time=[],NZoom=[],title='',ylab=''):
    #Function to create a changepoint plot 
    
    if len(time)==0:
        time = np.arange(0,len(signal)) 
    if len(NZoom)==0:
        NZoom = [np.min(time),np.max(time)]
    fig = plt.figure(figsize=(12,4))
    a0 = plt.subplot(1,1,1)
    a0.set_prop_cycle('color',cmvir(2))
    print( bkps[0]!=0)
    if bkps[0]!=0: #insert a zero if one doesn't exist
        if isinstance(bkps,list): bkps.insert(0,0)
        else:                     bkps = np.insert(bkps,0,0)
    if bkps[-1]!=len(time): #insert a zero if one doesn't exist
        if isinstance(bkps,list): bkps.insert(len(bkps),len(time)-1)
        else:                     bkps = np.insert(bkps,len(bkps),len(time)-1)
    idx = np.searchsorted(time,NZoom, side="left")
    yrange=[min(signal[idx[0]:idx[1]-1])-np.max(abs(signal[idx[0]:idx[1]-1]))/500,max(signal[idx[0]:idx[1]-1])+np.max(abs(signal[idx[0]:idx[1]-1]))/500] 
    for ind,x in enumerate(bkps[:-1]):
        plt.plot(time[bkps[ind]:bkps[ind+1]+1],signal[bkps[ind]:bkps[ind+1]+1],linestyle='-',linewidth=4)
        plt.plot([time[x],time[x]],yrange,color='grey',linestyle='--')
    fig.tight_layout()
    plt.title(title,fontsize=30)
    plt.ylabel(ylab,fontsize=24)
    plt.xlabel('Time (seconds)',fontsize=22)
    plt.xlim(NZoom)
    plt.ylim(yrange)
    plt.show()
    
    
    
#%% Define success metric

def hf_goodness(true_peaks, test_peaks, sig, margin=0, f_roc = False):
    # Function to quantify how closely "test peaks" matches to the "true peaks"
    
    # If f_roc is True, return sensitivity, specificity, and positive predictability 
    # Otherwise, give number of true positives, false positives, and false negatives
    
    # Allow recursive iteration through different allowed margins
    if isinstance(margin,list):
        if len(margin)<3:
            marg_gap=1;
        else:
            marg_gap=margin[2]
        if len(margin)<2:
            margin = [margin[0],margin[0]+1]
                
        Nmeth = int((margin[1]-margin[0])/marg_gap)

        tp=np.zeros(Nmeth)
        fp=np.zeros(Nmeth)
        fn=np.zeros(Nmeth)
       
        for marg_ind,marg in enumerate(np.arange(margin[0],margin[1],marg_gap)): #start off small

            # print(marg)
            [tp[marg_ind],fp[marg_ind],fn[marg_ind]] = hf_goodness(true_peaks,test_peaks,sig, margin=marg)
                     
            if not np.mod(marg_ind,10):
                print("Processing for margin = %2.1f is done" % (marg))
            
            
    else:
        tp=0            #True Positives (correctly Identified Peaks)
        tpList=[]       
        fp=0            #False Positives (incorrectly Identified Peaks)
        fpList=[]
        fn=0            #False Negatives (incorrectly Missed Peaks)
        fnList=[]
        # sp=0            #Shifted False Positives (A peak that incorrectly identifies, but near just before peak)
        # spList=[]
        # sn=0            #Shifted False negative (A peak incorrectly missed, but with a false positive just after)
        # snList=[]
        
        true_peaks = np.asarray(true_peaks)
        test_peaks = np.asarray(test_peaks)
        
        for y in true_peaks:
            if np.min(np.abs(y-test_peaks))>margin:
                fnList.append(y)
                fn+=1
                
        # remove peaks we know aren't going to be involved in the comparison
        #true_peaks = np.setxor1d(true_peaks,fnList)
       
        if true_peaks.size!=0:  # If true_peaks then isn't an empty array
                
            for x in test_peaks:
                min_val = np.min(np.abs(x-true_peaks))
                #print(min_val)
                if min_val>margin:
                    fp += 1
                    fpList.append(x)
                else:
                    min_ind = np.where(np.abs(x-np.asarray(true_peaks)) == min_val)
                    if min_val!=np.min(np.abs(true_peaks[min_ind]-test_peaks)):
                        fp += 1
                        fpList.append(x)
                    else:
                        tp += 1
                        tpList.append(x)
            
    if f_roc:
        tn = len(sig)-tp-fn-fp
        sens = tp/(tp+fn)
        spec = tn/(tn+fp)
        ppre = tp/(tp+fp)
        return sens,spec,ppre
    else:
        return tp,fp,fn 
    
#%% Interp function

def interp_vec(data,time_data,time_out,S_kind='nearest',S_fill='extrapolate'):
    # Function to allow 1-line interpolation from one set of time indices to another
    #S_kind =  ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’.
    if len(np.asarray(data).shape)==2:                 # If 2D, interpolate down each column individually
        tmp_out = np.zeros((len(time_out),np.asarray(data).shape[1]))
        for x in np.arange(np.asarray(data).shape[1]):
            tmp_fun = interp1d(time_data,data[:,x],kind=S_kind,fill_value=S_fill,bounds_error=False)
            if type(data[0][0]) is (bool or np.bool_): # If Boolean input, make output Boolean
                tmp_out[:,x] = [int(x) for x in tmp_fun(time_out)]
            else:
                tmp_out[:,x] = tmp_fun(time_out)
        return tmp_out
    else:
        tmp_fun = interp1d(time_data,data,kind=S_kind,fill_value=S_fill,bounds_error=False)
        tmp_out = tmp_fun(time_out)
        if type(data[0]) is (bool or np.bool_): tmp_out = [int(x) for x in tmp_out]  # If Boolean input, make output Boolean
        return tmp_out

