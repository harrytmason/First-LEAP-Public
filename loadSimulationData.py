# -*- coding: utf-8 -*-
"""
Created on Tue May 31 12:44:31 2022

@author: Dr Harry T. Mason, University of York




Examine the noisy simulation data
"""

#%% Load libraries

import csv
from   datetime import datetime
import decimal
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
#import pingouin as pg 
from   PyEMD import EMD
import pylab
import re
import ruptures as rpt
import scipy
from   scipy.interpolate import interp1d,UnivariateSpline
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

sys.path.append("G:\My Drive\BackUp\Documents\Python\Research")
from HarryFunctions import (hf_find,cmvir,hf_cp,hf_ecg,clean_peaks,find_local_peaks,calc_all_hrv,viz_all_HRV,hrSQI,cp_plot,hf_goodness, interp_vec)

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
locits = importr('locits')

sys.path.append("G:\\My Drive\\BackUp\\Documents\\Python\\NoiseDetectionCode")

from feature_extraction import window_stat
from agglomerativeClustering import MultiDimensionalClusteringAGG
from plot_tools import plotLinearData, plotClusters

#%% Simulate noisy data
sr_ecg = 150
Nnoise = 2
t_end = 300
t_ecg = np.arange(0,t_end*sr_ecg)/sr_ecg
hr_std = 0
ecg = nk.ecg_simulate(duration=t_end, length=t_end*sr_ecg, sampling_rate=sr_ecg, noise=Nnoise, heart_rate=150, heart_rate_std=hr_std, method='ecgsyn', random_state=42)


#%% Initialise Strings
Ssesh = "2022-06-22"#"ECGDATA_30-05-2022"
#%% Load ECG

paths = hf_find(('%s.TXT' % Ssesh), ("G:\\Shared drives\\FIRST Leap\\Data\\Simulations\\ECG noise simulations data\\SimulatorTest1\\" ));
paths.sort()
print(paths)
#%% Load ecg data from YorkSensor
Npath = 0
filename=paths[Npath]

df_ecg = pd.read_csv(filename,delimiter=' ',engine='python',header=None,names=['tmp','t_ecg','ecg'])
df_ecg.head() 
df_ecg=df_ecg.sort_values(by=['t_ecg'],ascending=True)

ecg = np.asarray(df_ecg['ecg'])
t_ecg = np.asarray(df_ecg['t_ecg'])

t_end_ecg   = df_ecg['t_ecg'][df_ecg.index[-1]]
t_start_ecg = df_ecg['t_ecg'][df_ecg.index[0]]
t_gap       = (t_end_ecg-t_start_ecg)/(len(ecg)-1)

#t_ecg = np.arange(t_start_ecg, t_end_ecg+t_gap,t_gap)


sr_ecg = 1/np.mean(np.diff(t_ecg))

#%% Load ECG from Plux
paths = hf_find(('open*%s*.txt' % Ssesh), ("G:\\Shared drives\\FIRST Leap\\Data\\Simulations\\ECG noise simulations data\\SimulatorTest2\\" ));
paths.sort()
print(paths)

Npath = 0; # Now it's just the first signal
filename = paths[Npath] # stitch later
cols =  ["nSeq", "DI", "CH1", "CH2", "CH3_ZYG", "CH4_CORR", "CH5_LGHT", "CH6_Y", "CH7_X", "CH8_Z"];

ecg     = pd.Series(dtype=int)     # ecg
time_stamps = []

for filename in paths:
    
    df = pd.read_csv(filename, skiprows=[0,1,2], delimiter='\t', header = None)
    df.head() 
    
    header = open(filename)
    all_lines_variable = header.readlines()
    
    tmp = all_lines_variable[1]
    tmpDict = eval(tmp[1:-1])
    
    # kid A
    dict1      = tmpDict[list(tmpDict.keys())[0]] #dictionary with header info
    Ncol1      = len(dict1['column'])             # add the columns which exist
    df         = df.loc[:,0:Ncol1-1]                      #pandas file with everything
    sr_ecg     = dict1['sampling rate']
    cols1      = cols[0:2]+[cols[x+1] for x in dict1['channels']]
    df.set_axis(cols1, axis=1, inplace=True) #rename columns
    #for measure in dict1.keys():
    #    print('%s: %s' %(measure, dict1[measure]))
    begin1      =     dict1['time']
        
    time_stamps.append(datetime.strptime(dict1['time'],'%H:%M:%S.%f'))

    
    # append zeros to fill gaps
    start_i = int((time_stamps[-1] - time_stamps[0]).total_seconds()*sr_ecg) # if ecg recording had continued, this would be the index the recording started at
    
    ecg     = pd.concat([ecg,pd.Series(np.zeros(start_i-len(ecg)))], ignore_index=True)
    ecg     = pd.concat([ecg,df['CH1']], ignore_index=True)     


t_ecg = np.arange(0,len(ecg))/sr_ecg
    
#print signal info (length, etc.)
print("\n\nFile: %s\n%d samples.\n%2.1f minutes long.\n\n" %(filename,len(ecg),len(ecg)/sr_ecg/60 ))
#%% Mini Plot Function

def viz_ECG_mini(ax,ecg_sig, peaks_new,time,sr=500, NZoom=[], Title = "Filtered Signal",ylab="",col="plum"):
    if len(NZoom)==0:
        NZoom = [min(time),max(time)]
    plt.plot(time,ecg_sig, label = 'ECG signal', color=col,linewidth=2)
    plt.scatter(time[peaks_new],ecg_sig[peaks_new], color='black', marker='x',label = 'identified peaks',zorder=10)
#plt.legend()
    ax.set_xlim(NZoom)
    ax.set_title(Title)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel(ylab,fontsize=22)

#%% Plot raw ECG

plt.figure(figsize=(8,3))
NZ = [10,15]
ax1 = plt.subplot(111)
viz_ECG_mini(ax1,ecg, [],t_ecg,sr=sr_ecg, NZoom=NZ, Title = "Raw ECG")

#%% Process ECG on its own

ecg_nk2 = hf_ecg(ecg,sr=sr_ecg,freqfilt=[0.5,20])           

#%% Plot processed ECG

plt.figure(figsize=(8,3))
ax1 = plt.subplot(111)
viz_ECG_mini(ax1,ecg_nk2, [],t_ecg,sr=sr_ecg, NZoom=NZ, Title = "Preprocessing")

#%% Detect peaks

r_nk2 = nk.ecg_findpeaks(ecg_nk2, sampling_rate=sr_ecg, method="neurokit")
r_nk2 = np.asarray(r_nk2['ECG_R_Peaks'])
t_r_nk2 = t_ecg[r_nk2]

#%% Visualise Peaks

plt.figure(figsize=(8,3))
ax1 = plt.subplot(111)
viz_ECG_mini(ax1,ecg_nk2, r_nk2,t_ecg,sr=sr_ecg, NZoom=NZ, Title = "Preprocessing with Peaks")

plt.figure(figsize=(8,3))
ax1 = plt.subplot(111)
viz_ECG_mini(ax1,ecg, r_nk2,t_ecg,sr=sr_ecg, NZoom=NZ, Title = "Raw ECG with Preprocessing Peaks")



#%% Calculate Heart Rate
hr_nk2 = 60/np.diff(r_nk2/sr_ecg)
t_hr_nk2 = t_r_nk2[0:-1]+np.diff(t_r_nk2)/2

#%% Clean v2_nk2 peaks

r_nk3  = clean_peaks(r_nk2,ecg)
t_r_nk3 = t_ecg[r_nk3]
hr_nk3 = 60/np.diff(np.asarray(r_nk3)/sr_ecg)
t_hr_nk3 = t_r_nk3[0:-1]+np.diff(t_r_nk3)/2

#%% Plot cleaned Peaks
plt.figure(figsize=(8,3))
NZ = [200,210]#[30100,30150]#[9990,10000]#[400,410]
ax1 = plt.subplot(111)
viz_ECG_mini(ax1,ecg, r_nk3,t_ecg,sr=sr_ecg, NZoom=NZ, Title = "Raw ECG with Corrected Peaks\n0.5mv 20% Deviation\nTremor Noise")
plt.grid(axis='x')
#plt.ylim([450,650])

#%% Iterative peak plots

ind_sus=0
Npeaks = 10
for x in np.arange(0,np.floor(len(r_emd2)/Npeaks)):
    plt.figure(figsize=(8,3))
    ax1 = plt.subplot(111)
    ind_sus = int(x)*Npeaks
    min_ind = max([0,r_emd2[ind_sus]-int(0.2*sr_ecg)])
    max_ind = min([r_emd2[min([len(r_emd2)-1,ind_sus+Npeaks+1])]+int(0.2*sr_ecg),len(ecg)-1])
    Peaks_old = np.array([x for x in r_emd2 if x>r_emd2[ind_sus] and x<r_emd2[ind_sus+Npeaks+1]])
    NZ = [t_ecg[min_ind],t_ecg[max_ind]]
    ecg_old_vec = np.array(ecg[min_ind:max_ind])
    viz_ECG_mini(ax1,ecg_old_vec,Peaks_old-min_ind,  t_ecg[min_ind:max_ind],sr=sr_ecg, NZoom=NZ, Title = ("%s, %d-%d seconds" % (Ssesh, t_ecg[min_ind],t_ecg[max_ind])))


#%% Median filter and add spline

t_hr_nk3 = t_r_nk3[0:-1]+np.diff(t_r_nk3)/2

hr_nk3_filt, qual_hr = hrSQI(hr_nk3,sr=sr_ecg)

hr_nk3_fun = UnivariateSpline(t_hr_nk3,hr_nk3_filt, k=5)
hr_nk3_spl = hr_nk3_fun(t_hr_nk3)

fig, ax = plt.subplots(figsize=(8,3))
NZ = [160,460]
# Plot linear sequence, and set tick labels to the same color
ax.plot(t_hr_nk3,hr_nk3_filt, color='red',marker='.',label='Filtered HR')
ax.plot(t_hr_nk3,hr_nk3,label='HR Derived from Peaks', color='green')
#ax.plot(t_hr_nk3,hr_nk3_spl,label='Splined HR', color='blue')
ax.set_xlim(NZ)
#ax.legend(loc='upper right')
ax.set_title('Noisy Simulation, v2 sensor, Median-Filtered Spline')#'Original Heart Rate (red) with Spline (blue)')
#ax.set_ylim([43, 60])
ax.set_ylabel('Heart Rate')
ax.set_xlabel('Time (seconds)')
plt.grid()
plt.legend(loc = "lower left")
plt.show()

#%% EMD processing
emd = EMD()

ecg_emd = hf_ecg(ecg,sr=sr_ecg,freqfilt=[0.01,40])    
ecg_imfs = emd.emd(np.array(ecg_emd)) 
#ecg_imfs = emd.emd(np.array(ecg)) # Pal wants me to remove baseline wander first
#-np.mean(ecg)
ecg_imfs_pow = [nk.signal_psd(x,method="welch",min_frequency=0, max_frequency=160) for x in ecg_imfs]

#%% EMD Processing

#%% Find IMFS with most of the energy above 0.5Hz
ind_low = np.where(ecg_imfs_pow[0]['Frequency']>0.5)[0][0] # -ind_05 for the negative index
ind_high = np.where(ecg_imfs_pow[0]['Frequency']>8)[0][0] # -ind_05 for the negative index
P_05 = np.zeros(len(ecg_imfs))
for ind, x in enumerate(ecg_imfs_pow):
    P_05[ind] = x['Power'][ind_low:ind_high].sum()/x['Power'].sum()
    

ecg_emd = np.sum(ecg_imfs[np.where(P_05<0.5)[0]],axis=0) # +np.mean(ecg)


# ind_05 = np.where(ecg_imfs_pow[0]['Frequency']>0.5)[0][0] # -ind_05 for the negative index
# ind_50 = np.where(ecg_imfs_pow[0]['Frequency']<50)[0][-1] # -ind_05 for the negative index

# P_05 = np.zeros(len(ecg_imfs))
# P_50 = np.zeros(len(ecg_imfs))
# for ind, x in enumerate(ecg_imfs_pow):
#     P_05[ind] = x['Power'][ind_05:].sum()/x['Power'].sum()
#     P_50[ind] = x['Power'][:ind_50].sum()/x['Power'].sum()
    
# ecg_emd = np.sum(ecg_imfs[np.intersect1d(np.where(P_05>0.1)[0],np.where(P_50>0.2)[0])],axis=0) # +np.mean(ecg)
#ecg_emd = hf_ecg(ecg_emd,sr=sr,freqfilt=[10,20])    

plt.figure(figsize=(8,3))
NZ = [3370,3390]
ax1 = plt.subplot(111)
viz_ECG_mini(ax1,ecg_emd, [],t_ecg,sr=sr_ecg, NZoom=NZ, Title = "Preprocessing")
#ax1.set_ylim([-12000,15000])


#%% Peak function
#peak calculation

r_emd = nk.ecg_findpeaks(ecg_emd, sampling_rate=sr_ecg, method="neurokit")
r_emd = np.asarray(r_emd['ECG_R_Peaks'])
t_r_emd = t_ecg[r_emd]

plt.figure(figsize=(8,3))
#NZ = [1320,1350]
ax1 = plt.subplot(111)
viz_ECG_mini(ax1,ecg_emd, r_emd,t_ecg,sr=sr_ecg, NZoom=NZ, Title = "Preprocessing with Peaks")
#ax1.set_ylim([-12000,15000])

plt.figure(figsize=(8,3))
#NZ = [1320,1350]
ax1 = plt.subplot(111)
viz_ECG_mini(ax1,ecg, r_emd,t_ecg,sr=sr_ecg, NZoom=NZ, Title = "Raw ECG with Preprocessing Peaks")
#ax1.set_ylim([12000,55000])


#%% HR calcs

hr_emd = 60/np.diff(r_emd/sr_ecg)
# hr_hpy = 60/np.diff(r_hpy/sr)
t_hr_emd = t_r_emd[0:-1]+np.diff(t_r_emd)/2
# t_hr_hpy = t_r_hpy[0:-1]+np.diff(t_r_hpy)/2


#%% Clean emd peaks

r_emd2  = clean_peaks(r_emd,ecg)
t_r_emd2 = t_ecg[r_emd2]
hr_emd2 = 60/np.diff(np.asarray(r_emd2)/sr_ecg)
t_hr_emd2 = t_r_emd2[0:-1]+np.diff(t_r_emd2)/2

#%% Visualise
plt.figure(figsize=(8,3))
NZ = [70,90]
ax1 = plt.subplot(111)
viz_ECG_mini(ax1,ecg, r_emd2,t_ecg,sr=sr_ecg, NZoom=NZ, Title = "Raw ECG with Corrected Peaks")
#ax1.set_ylim([12000,55000])
plt.grid(axis='x')

#%% Create Time index of time swtiches
cp_onoff = [[10,310],[320,620],[630,930],[940,1240],[1250,1550],[1560,1860],[1920,2220],[2230,2530],[2540,2840],[2850,3150],[3160,3460],[3490,3790]]
#[0,10,310,320,620,630,930,940,1240,1250,1550,1560,1860,1920,2220,2230,2530,2540,2840,2850,3150,3160,3490,3790]
labels=['2mv 5% Deviation\nNo Noise','2mv 5% Deviation\nPseudo-Gaussian','1mv 5% Deviation\nPseudo-Gaussian','0.5mv 5% Deviation\nPseudo-Gaussian','0.5mv 10% Deviation\nPseudo-Gaussian','0.5mv 20% Deviation\nPseudo-Gaussian','2mv 5% Deviation\nTremor Noise','1mv 5% Deviation\nTremor Noise','0.5mv 5% Deviation\nTremor Noise','0.5mv 10% Deviation\nTremor Noise','0.5mv 20% Deviation\nTremor Noise','2mv 5% Deviation\nBaseline Wander']
Npeaks_missed = [0,1,0,1,2,1,2,1,3,1,0,0] # how many I'd expect the algorithm to miss because no peak was detected due to the sensor skipping in time
Npro = len(cp_onoff) #Number of protocols

#%% Create 0-1 label
bool_ecg = np.zeros(len(ecg))
bool_hr  = np.zeros(len(hr_nk3_filt))
for x in cp_onoff:
    bool_ecg[np.intersect1d(np.where(t_ecg>x[0]),np.where(t_ecg<x[1]))]=1
    bool_hr[ np.intersect1d(np.where(t_hr_nk3>x[0]),np.where(t_hr_nk3<x[1]))]=1
    
ecg_onoff = ecg*bool_ecg
hr_onoff  = hr_nk3*bool_hr

#%% Create histogram of only relevant values

#t_ecg_diff = np.diff(t_ecg[])
t_ecg_diff_onoff = np.zeros(0)
for x in cp_onoff:
    t_ecg_diff_onoff=np.concatenate((t_ecg_diff_onoff,(np.diff(t_ecg[np.intersect1d(np.where(t_ecg>x[0]),np.where(t_ecg<x[1]))]))))
    
    
#%% Load True Peaks

ldStr = paths[Npath].split("\\", -1)
ldStr = ldStr[-1].split(".", 1)
ldStr = "G:\\My Drive\\BackUp\\Documents\\Data\\ProcessedData\\" + ldStr[0] + "_peaks.txt"
r_tru=np.loadtxt(ldStr,dtype=int)

#%% Calculate Specificity, Sensitivity, etc for each period
tp = np.zeros(Npro)
fp = np.zeros(Npro)
tn = np.zeros(Npro)
fn = np.zeros(Npro)
false_peaks = []
missed_peaks = [] 
for x in np.arange(0,Npro):
    min_ind = np.where(t_ecg>cp_onoff[x][0])[0][0]
    max_ind = np.where(t_ecg<cp_onoff[x][1])[0][-1]
    r_tru_onoff = [y for y in r_tru if (y>=min_ind and y <= max_ind)]
    r_nk3_onoff = [y for y in r_nk3 if (y>=min_ind and y <= max_ind)]
    tp[x] = len([y for y in r_nk3_onoff if np.isin(y,r_tru_onoff)])
    fp[x] = len(r_nk3_onoff) - tp[x]
    fn[x] = len(r_tru_onoff) - tp[x]
    tn[x] = max_ind-min_ind + 1 - tp[x] - fp[x] - fn[x]
    false_peaks = false_peaks+ [y for y in r_nk3_onoff if not np.isin(y,r_tru_onoff)]
    missed_peaks = missed_peaks+[y for y in r_tru_onoff if not np.isin(y,r_nk3_onoff)]
    print(int(tp[x]),int(fp[x]),int(fn[x]),int(tn[x]),Npeaks_missed[x], t_ecg[min_ind],"-",t_ecg[max_ind])

sens = tp/(tp+fn)
spec = tn/(fp+tn)
ppv  = tp/(tp+fp)

print(sens,spec,ppv)

#%% Plot missed peaks

for x in missed_peaks:
    fig = plt.figure(figsize=(12,4))
    min_ind = max(0,x-500)
    max_ind = min(len(t_ecg),x+500)
    NZ = [t_ecg[min_ind],t_ecg[max_ind]]
    plt.plot(t_ecg,ecg, label = 'ECG signal', color='plum',linewidth=1)

    plt.scatter(t_ecg[r_tru],[ecg[x] for x in r_tru], color='green', label = 'true peaks',linewidth=0.5)
    plt.scatter(t_ecg[r_nk3],[ecg[x] for x in r_nk3], color='red', label = 'calculated peaks',linewidth=0.5)
    

    plt.legend(loc='upper right')
    plt.xlim(NZ)
    idx = np.searchsorted(t_ecg,NZ, side="left")
    plt.ylim([min(ecg[idx[0]:idx[1]-1])-np.median(np.abs(ecg))*0.1,max(ecg[idx[0]:idx[1]-1])*1.1]+np.median(np.abs(ecg))*0.1)
    #a0.set_xlabel('s')
    plt.grid(axis = 'x')
    #plt.title("Original Signal")

#%% non-skip calculations
# done by manual inspection of the plots, to work out values not due to missed peaks

fp_ns = [0,0,0,2,0,1,1,0,0,5,3,1] #nk3
fn_ns = [0,0,0,2,0,2,1,0,0,5,3,2] # nk3
fp_ns = [0,0,0,2,0,1,0,0,0,2,4,0] #emd2
fn_ns = [0,0,0,2,0,1,0,0,1,1,4,0] # emd2
spec_ns = tn/(fp_ns+tn)
ppv_ns  = tp/(tp+fp_ns)
sens_ns = tn/(fp_ns+tn)

#%% Plot results

fig = plt.figure(figsize=(12,4))
a0  = plt.subplot(121)
plt.bar(np.arange(Npro),sens_ns)
a0.set_ylim([1-(1-min(sens_ns))*1.5,1])
a0.set_xlabel('Protocol')
a0.set_xticks(np.arange(Npro))
a0.set_xticklabels(labels,rotation=90)

#plt.grid(axis = 'x')
a0.set_title("Sensitivity")

a1  = plt.subplot(122)
plt.bar(np.arange(Npro),ppv_ns)
a1.set_ylim([1-(1-min(ppv_ns))*1.5,1])
a1.set_xlabel('Protocol')
a1.set_xticks(np.arange(Npro))
a1.set_xticklabels(labels,rotation=90)

#plt.grid(axis = 'x')
a1.set_title("PPV")

#%% Translate to noise detection format
#
min_ind_1 = np.where(t_ecg>cp_onoff[3][0])[0][0]
max_ind_1 = np.where(t_ecg>cp_onoff[5][1])[0][0]
min_ind_2 = np.where(t_ecg>cp_onoff[8][0])[0][0]
max_ind_2 = np.where(t_ecg>cp_onoff[10][1])[0][0]

ecg_05 = np.concatenate((ecg[min_ind_1:max_ind_1],ecg[min_ind_2:max_ind_2]))
t_05   = np.concatenate((t_ecg[min_ind_1:max_ind_1],t_ecg[min_ind_2:max_ind_2]))

signal = ecg_05#ecg
fs = int(sr_ecg)
time = t_05#t_ecg

#window size
win = 500
print("Window Size: %d" % win)

#number of clusters
n_clusters = 2
#----------------------------------------------------------------------------------------------------------------------
#%%                                        Extract Features
#----------------------------------------------------------------------------------------------------------------------
print("Extracting features...")

#1 - Std Window
signalSTD = window_stat(signal, fs=fs, statTool='std', window_len=int((win*fs)/256))
print("...feature 1 - STD")

#2 - ZCR
signalZCR64 = window_stat(signal, fs=fs, statTool='zcr', window_len=int((win*fs)/512))
print("...feature 2 - ZCR")

#3 - Sum
signalSum64 = window_stat(signal, fs=fs, statTool='sum', window_len=int((win*fs)/256))
signalSum128 = window_stat(signal, fs=fs, statTool='sum', window_len=int((win*fs)/100))
print("...feature 3 - Sum")

# #4 - Number of Peaks above STD
# signalPKS = window_stat(signal, fs=fs, statTool='findPks', window_len=int((win*fs)/128))
# signalPKS2 = window_stat(signal, fs=fs, statTool='findPks', window_len=int((64 * fs) / 100))
# print("...feature 4 - Pks")

# #5 - Amplitude Difference between successive PKS
# signalADF32 = window_stat(signal, fs=fs, statTool='AmpDiff', window_len=int((win*fs)/128))
# # signalADF128 = WindowStat(signal, fs=fs, statTool='AmpDiff', window_len=(2*win*fs)/100)
# print("...feature 5 - AmpDif")

# #6 - Medium Frequency feature
# signalMF = window_stat(signal, fs=fs, statTool='MF', window_len=int((32*fs)/128))
# print("...feature 6 - MF")
#%% Cluster and plot
#Feature Matrix
FeatureNames = ["Standard Deviation", 'Sum', 'ZCR']
FeatureMatrix = np.array(
    [signalSTD, signalSum64, signalZCR64]).transpose()


#Plot Feature Data
plotLinearData(time, FeatureMatrix, signal, FeatureNames)


print("Starting Clustering...")
X, y_pred, XPCA, params = MultiDimensionalClusteringAGG(FeatureMatrix, n_clusters=n_clusters, Linkage='ward',
	                                                        Affinity='euclidean')

print("Plotting...")
Clusterfig = plotClusters(y_pred, signal, time, XPCA, n_clusters)

#%% Finding indexes of the start of periods, and the end

def startend_blocks(event):
    ev=np.asarray(event)
    np.insert(ev,0,0)       #ensure clear periods at start and end
    np.insert(ev,len(ev),0)
    
    per_start = []
    per_end   = []
    for x in np.arange(len(ev)-1):
        if ev[x]<ev[x+1]:
            per_start.append(x+1)
        elif ev[x]>ev[x+1]:
            per_end.append(x+1)
    
    out = list(zip(per_start,per_end)) 
    return out

#%% Plot HR

y_pred_hr = interp_vec(y_pred,time,t_hr_nk3)
#Clusterfig = plotClusters(y_pred_hr, hr_nk3_spl, t_hr_nk3, XPCA, n_clusters)


block_y_pred = startend_blocks(y_pred_hr)
y_pred_cps = np.reshape(np.asarray(block_y_pred),(1,-1))[0]
NZ=[200,t_hr_nk3[-1]]
cp_plot(hr_nk3_spl,y_pred_cps,t_hr_nk3,NZoom=NZ,title=('Bad HR check, Simulation %s' % (Ssesh)))











#%% Mini Save function for heart rate,ecg
mlab_hr_peaks = [x+1 for x in r_nk3 if x < len(t_hr_nk3)]
identifier_string = Ssesh+'_'+Sp+Sb
sio.savemat("G:\\My Drive\\BackUp\\Documents\\Data\\ProcessedData\\"+identifier_string+"_ecg+hr.mat",{'Subject_ID':identifier_string,'heart_rate':hr_nk3_spl,'heart_rate_time':t_hr_nk3,'heart_rate_peaks':mlab_hr_peaks,'start_time':start_t,'ecg_raw':ecg,'ecg_time':t_sens,'ecg_processed':ecg_nk2})


#%% Plotting functions

def plt_signal(ax,time,sig,ylab='',NZoom=NZ,col='blue',YZoom=[],title=''):
    ax.plot(time,sig,color=col)
    ax.set_ylabel(ylab,fontsize=22)
    if len(NZoom)==0:
        NZoom = [np.min(time),np.max(time)]
    ax.set_xlim(NZoom)
    if len(YZoom)==0:
        idx = np.searchsorted(time,NZ, side="left")
        YZoom = [min(sig[idx[0]:idx[1]-1])-abs(max(sig[idx[0]:idx[1]-1]))*0.05,max(sig[idx[0]:idx[1]-1])+abs(max(sig[idx[0]:idx[1]-1]))*0.05]
    ax.set_ylim(YZoom)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.grid(axis="x")

def plt_labels(ax,time,data,labels,ylab='',NZoom=[],title='',):
    if len(NZoom)==0:
        NZoom = [np.min(time),max(time)]
    Ncol = int(data.size/len(data))
    #ax.set_prop_cycle('color',cmvir(6))
    ax.plot(time,data+np.arange(0,Ncol)*2,linewidth=4)
    ax.set_yticks(np.arange(0,Ncol)*2+0.5)
    ax.set_yticklabels(labels[0:Ncol])
    ax.set_ylabel('The Separate Signals',fontsize=22)
    if len(NZoom)==0:
        NZoom = [np.min(time),np.max(time)]
    ax.set_xlim(NZoom)
    ax.set_xlabel('Time (seconds)',fontsize=14)
    plt.setp(ax.get_xticklabels(), visible=True)
    plt.grid(axis="x")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    
def plt_sa(ax,time,data,ylab='Sustained Attention',NZoom=[]):
    if len(NZoom)==0:
        NZoom = [np.min(time),max(time)]
    ax.plot(time,data, linewidth=4)
    ax.set_yticks([0,1])
    ax.set_ylim([-0.1,1.1])
    ax.set_yticklabels(['off','on'])
    ax.set_xlim(NZoom)
    ax.set_ylabel(ylab,fontsize=22)
    ax.set_xlabel('Time (seconds)',fontsize=22)
    plt.grid(axis="x")
    plt.setp(ax.get_xticklabels(), visible=True)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)




#%% Plot HR/dHR bar

def plt_bar(data_mean,data_std,labels,f_std=True,title="",ylab=""):
    Ncol = len(data_mean)
    tmp , ax=plt.subplots(figsize=(3,5))
    if f_std:
        plt.bar(np.arange(0,Ncol),data_mean,
                yerr=data_std,
                align='center',
                alpha=0.5,
                ecolor='black',
                capsize=10)
        yrange = ([np.min(data_mean-data_std)-np.abs(np.max(data_mean+data_std))*0.05,np.max(data_mean+data_std)+np.abs(np.max(data_mean+data_std))*0.05])
    else:
        plt.bar(np.arange(0,Ncol),data_mean)
        yrange = ([np.min(data_mean)-np.abs(np.max(data_mean))*0.05,np.max(data_mean)+np.abs(np.max(data_mean))*0.05])
    plt.xticks(rotation=0)
    ax.set_ylabel(ylab)
    ax.set_xticks((np.arange(0,Ncol)))
    ax.set_xticklabels(labels)
    ax.set_ylim(yrange)
    plt.title(title)
    #plt.tight_layout()
    plt.grid()


#%% Function to calculate stats

def mat_stats(data_mat, boo_mat, act_ind):
    Ncol     = len(act_ind)
    mat_all  = []
    mat_mean = np.zeros(Ncol)
    mat_std  = np.zeros(Ncol)
    for x in np.arange(Ncol):
        att_vec = np.sum(boo_mat[:,act_ind[x]],axis=1)
        att_vec = att_vec>0
        
        try: 
            mat_all.append(data_mat[att_vec[1:]]) # heart rate change boolean (take the second label as the relevant one)
        except: 
            mat_all.append(data_mat[att_vec])    # heart rate boolean
        mat_mean[x] = np.mean(mat_all[x])
        mat_std[x]  = np.std(mat_all[x])
       
    return mat_all,mat_mean,mat_std

#%% Changepoint plot
def plt_cp(ax,signal,bkps,time=[],NZoom=[],ylab='',title='',Ncolors=2):
    if len(time)==0:
        time = np.arange(0,len(signal)) 
    if len(NZoom)==0:
        NZoom = [np.min(time),np.max(time)]
    col_vec = cmvir(Ncolors)+cmvir(Ncolors)
    col_vec.sort()
    ax.set_prop_cycle('color',col_vec)
    if bkps[0]!=0: bkps.insert(0,0)
    idx = np.searchsorted(time,NZoom, side="left")
    yrange=[min(signal[idx[0]:idx[1]-1])-np.max(abs(signal[idx[0]:idx[1]-1]))/500,max(signal[idx[0]:idx[1]-1])+np.max(abs(signal[idx[0]:idx[1]-1]))/500] 
    for ind,x in enumerate(bkps[:-1]):
        plt.plot(time[bkps[ind]:bkps[ind+1]+1],signal[bkps[ind]:bkps[ind+1]+1],linestyle='-',linewidth=4)
        plt.plot([time[x],time[x]],yrange,linestyle='--')
    #fig.tight_layout()
    # plt.title(title,fontsize=30)
    # plt.xlabel('Time (seconds)',fontsize=22)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.xlim(NZoom)
    plt.ylim(yrange)
    plt.ylabel(ylab,fontsize=22)
    plt.title(title,fontsize=24)
    # plt.show()

def plt_cp_bp(ax,bkps,time,NLim=[-0.5,1.5],Ncolors=2):
    col_vec=cmvir(Ncolors)
    col_vec.sort()
    ax.set_prop_cycle('color',col_vec) 
    for x in bkps[:-1]:
            plt.plot([time[x],time[x]],NLim,linestyle='--')
    plt.grid(False)
    YLim = ax.get_ylim()
    YLim = [max([YLim[0],NLim[0]]),min([YLim[-1],NLim[-1]])]
    ax.set_ylim(YLim)



#%% Changepoint Analysis for HR
bkps_hr = hf_cp(hr_nk3_spl,cp_type='linear',time=t_hr_nk3)
#bkps_hr = hf_cp(hr_nk3_spl,cp_type='cptnp',time=t_hr_nk3)

#%% Plotting options

# NZ=[0,100]
cp_plot(hr_nk3_spl,bkps_hr,t_hr_nk3,NZoom=NZ,title='Change Point Detection: Pelt Search Method')

#rpt.display(signal, result, figsize=(10, 6))
#plt.title('Change Point Detection: Pelt Search Method')
#plt.show()  

#%% Changepoint and Labels plot for heart rate
NZ=[0,460]

fig = plt.figure(figsize=(16,12))
gs=gridspec.GridSpec(3,1,height_ratios=[3,2,2])

a0= plt.subplot(gs[0])
plt_cp(a0,hr_nk3_spl,bkps_hr,t_hr_nk3,NZoom=NZ,ylab='Change Point Detection\nPelt Search Method\n',title="Heart Rate")


#%% Mini Save function for heart rate,bodym motion, changepoints
mlab_bkps_hr = [x+1 for x in bkps_hr if x < len(t_hr_nk3)]
identifier_string = Ssesh
sio.savemat("G:\\My Drive\\BackUp\\Documents\\Data\\ProcessedData\\"+identifier_string+"_signals+breakpoint.mat",{'Subject_ID':identifier_string,'heart_rate':hr_nk3_spl,'heart_rate_time':t_hr_nk3,'heart_rate_changepoints':mlab_bkps_hr})



#%% Finding indexes of the start of periods, and the end

def startend_blocks(event):
    ev=np.asarray(event)
    np.insert(ev,0,0)       #ensure clear periods at start and end
    np.insert(ev,len(ev),0)
    
    per_start = []
    per_end   = []
    for x in np.arange(len(ev)-1):
        if ev[x]<ev[x+1]:
            per_start.append(x+1)
        elif ev[x]>ev[x+1]:
            per_end.append(x+1)
    
    out = list(zip(per_start,per_end)) 
    return out


#%% Find maximum values within a given block (for priscilla and mari)

b_ind = 4
b_test = startend_blocks(data[:,b_ind])

print("The maximum block length of %s was %2.1f seconds long and occurred at %2.1f seconds (or %2.2f minutes)" %(events[b_ind], np.max(np.diff(b_test,axis=1) )/sr_att,b_test[np.argmax(np.diff(b_test,axis=1))][0]/sr_att,b_test[np.argmax(np.diff(b_test,axis=1))][0]/sr_att/60))


#%% Calculate stats

res_nk3 = calc_all_hrv(r_nk3, ecg, sr=sr_ecg, f_freq=False,t_width=5)
sdnn_nk3= res_nk3['HRV_SDNN']
rmssd_nk3 = res_nk3['HRV_RMSSD']
t_hrv = res_nk3['time_5']
rmssd_nk3 = np.asarray(rmssd_nk3)





#%% Create Boolean array function
def create_boolean(bkps,array_len,zero_start=True):
    bool_vec = np.zeros(array_len)        # create array
    
    if not zero_start:      bool_vec +=1      # choose if startng at zero or 1
    if bkps[0]==0:          bkps = bkps[1:]   # ignore any leading zero on breakpoints
    if bkps[-1]==array_len: bkps = bkps[:-1]  # igrnoe the end breakpoint if just the lenght of the vector
    
    for x in bkps:
        if   bool_vec[x]==0: bool_vec[x:] = 1
        elif bool_vec[x]==1: bool_vec[x:] = 0
    #bool_vec = np.asarray([ int(x) for x in bool_vec]) #If I need it as integers
    return bool_vec

#%% Quantifying success metrics

# Create list of changepoints
block_s_a = startend_blocks(sus_att[:,0])
sus_att_cps = np.reshape(np.asarray(block_s_a),(1,-1))[0]
bool_s_a = np.reshape(sus_att[:,0],(1,-1))[0]

#%% Hausdorff/ari
bool_bkps_hr = create_boolean(bkps_hr,len(hr_nk3_filt),zero_start=True)
bool_bkps_hr = interp_vec(bool_bkps_hr,t_hr_nk3,t_att,S_kind='nearest')

from sklearn.metrics import adjusted_rand_score
ari_hr = adjusted_rand_score(bool_bkps_hr, bool_s_a)

from scipy.spatial.distance import directed_hausdorff 

haus_hr = directed_hausdorff([bool_bkps_hr,t_att], [bool_s_a,t_att])

print("Ari score: %1.5f" % (ari_hr))
print("Haus score: %1.5f" % (haus_hr[0]))

#%% "Peak detection" style verification metric

# Needs work, great start

[tp,fp,fn] = hf_goodness(t_att[sus_att_cps],t_hr_nk3[bkps_hr[:-1]],sus_att,margin=[0,20,1])

#%% Calculate slopes
s_a_ind   = np.unique(np.concatenate(([0],sus_att_cps,[len(sus_att)])))
hr_bkps_ind =  np.unique(np.concatenate(([0],bkps_hr,[len(hr_nk3)])))
hr_bkps_slope = np.zeros(len(hr_bkps_ind)-1)
hr_bkps_icept  = np.zeros(len(hr_bkps_ind)-1)

for x in np.arange(0,len(hr_bkps_ind)-1):
    #print(x,hr_bkps_ind[x])
    hr_bkps_slope[x],hr_bkps_icept[x] =  np.polyfit(t_hr_nk3[hr_bkps_ind[x]:hr_bkps_ind[x+1]], hr_nk3_filt[hr_bkps_ind[x]:hr_bkps_ind[x+1]], 1)
    
#%% Plot the HR slopes
fig = plt.figure(figsize=(12,8))
gs=gridspec.GridSpec(2,1,height_ratios=[3,2])

a0= plt.subplot(gs[0])
plt_cp(a0,hr_nk3_filt,bkps_hr,t_hr_nk3,NZoom=NZ,ylab='Post-Regression\nHeart\nRate',title="Heart Rate Changepoint")

for x in np.arange(0,len(hr_bkps_ind)-1):
    plt.plot(t_hr_nk3[hr_bkps_ind[x]:hr_bkps_ind[x+1]],hr_bkps_slope[x]*t_hr_nk3[hr_bkps_ind[x]:hr_bkps_ind[x+1]]+hr_bkps_icept[x],color='black',linewidth=4)


#%% Phase Plot

hr_nk3_vel = np.diff(hr_nk3_filt)
hr_nk3_acc = np.diff(hr_nk3_vel)
fig = plt.figure(figsize=(12,8))
plt.plot(hr_nk3_vel[0:50],hr_nk3_acc[0:50])
plt.plot(hr_nk3_vel[50:100],hr_nk3_acc[50:100])
plt.plot(hr_nk3_vel[100:150],hr_nk3_acc[100:150])
plt.plot([hr_nk3_vel[0]], [hr_nk3_acc[0]], 'o',markersize=10) # start
plt.plot([hr_nk3_vel[149]], [hr_nk3_acc[149]], 's',markersize=10) # end
plt.legend(['1','2','3'])



#%% Create CSV variables

# Heart Rate
hr_csv = hr_nk3_fun(t_ecg)
hr_filt_csv = interp_vec(hr_nk3_filt,t_hr_nk3,t_ecg)
dhr_csv= np.insert(np.diff(hr_csv),0,0)

t_ind=np.where((t_ecg)>=0) #from start of labels
t_ind = t_ind[0][0]                    #the index of first non-negative value



# hrv labels
sdnn_csv= interp_vec(sdnn_nk3,t_hrv,t_ecg)
rmssd_csv = interp_vec(rmssd_nk3,t_hrv,t_ecg)



# Create Lists for csv
header_csv = ['Time (s)', 'ECG','Heart Rate','Pre-Spline Heart Rate','Heart Rate Difference', 'SDNN','RMSSD']
data_csv =list(zip(t_ecg,ecg,hr_csv,hr_filt_csv,dhr_csv,sdnn_csv,rmssd_csv))

data_csv = data_csv[t_ind:] #only values from start of labelling (so as not to falsely equate things with zeros just because that part wasn't labelled)


# Save String
svStr = filename.split("\\", -1)
svStr = svStr[-1].split(".", 1)
svStr = "G:\\My Drive\\BackUp\\Documents\\Data\\ProcessedData\\" + svStr[0] + "_processed.csv"
print("About to save data in %s" % svStr)


#%%
#%%
#%%
#%%
#%%
#%% Save to csv

with open(svStr, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header_csv)

    # write multiple rows
    writer.writerows(data_csv)
    
    
# #%% HR csv#
# header_HR = ['Time (s)', 'Pre-Spline HR']
# data_HR =list(zip(t_hr_nk3,hr_nk3_filt))
# svStrHR = filename_mat.split("\\", -1)
# svStrHR = svStrHR[-1].split(".", 1)
# svStrHR = "G:\\My Drive\\BackUp\\Documents\\Data\\ProcessedData\\" + svStrHR[0] + "_HR_processed.csv"

# #%% Save to csv

# with open(svStrHR, 'w', encoding='UTF8', newline='') as f:
#     writer = csv.writer(f)

#     # write the header
#     writer.writerow(header_HR)

#     # write multiple rows
#     writer.writerows(data_HR)