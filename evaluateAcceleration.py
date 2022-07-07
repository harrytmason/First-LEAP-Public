# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 12:47:12 2022

@author: Dr. Harry T. Mason,

Evaluate Acceleration script, to deal with acceleration data
"""

#%% Load libraries
import math
import numpy as np
import pandas as pd
import os
import re
import fnmatch
import pylab
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN, AffinityPropagation
pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('lr', LinearRegression())])
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage.metrics import peak_signal_noise_ratio
import heartpy as hp
from scipy.signal import (welch, find_peaks)
import  pywt
from ecgdetectors import Detectors
import hrv
import scipy.stats as spyst
from scipy.interpolate import interp1d
import neurokit2 as nk 
import scipy
import pingouin as pg 
from statsmodels.multivariate.manova import MANOVA
import sys
sys.path.append("G:\My Drive\BackUp\Documents\Python\Research")
from HarryFunctions import *


#%% Find relevant files if working with BF data

paths = hf_find('*.txt', "C:\\Users\\htm516\\Documents\\Data\\BF_sensorData\\");

print(paths)


#%% Find relevant files if working with BF data


Npath = 2; 
filename = paths[Npath]

cols =  ["nSeq", "DI", "CH1", "CH2", "CH3_ZYG", "CH4_CORR", "CH5_LGHT", "CH6_Y", "CH7_X", "CH8_Z"];
df = pd.read_csv(filename, skiprows=[0,1,2], delimiter='\t', header = None)
df.head() 

header = open(filename)
all_lines_variable = header.readlines()

tmp = all_lines_variable[1]
tmpDict = eval(tmp[1:-1])

# kid A
dict1       = tmpDict[list(tmpDict.keys())[0]] #dictionary with header info
Ncol1       = len(dict1['column'])             # add the columns which exist
df1         = df.loc[:,0:Ncol1-1]                      #pandas file with everything
sr1         = dict1['sampling rate']
cols1 = cols[0:2]+[cols[x+1] for x in dict1['channels']]
df1.set_axis(cols1, axis=1, inplace=True) #rename columns
#for measure in dict1.keys():
#    print('%s: %s' %(measure, dict1[measure]))
    

# # kid B
dict2       = tmpDict[list(tmpDict.keys())[1]] #dictionary with header info
Ncol2       = len(dict2['column'])             # add the columns which exist
df2         = df.loc[:,Ncol1:(Ncol2+Ncol1-1)]
sr2         = dict2['sampling rate']
cols2 = cols[0:2]+[cols[x+1] for x in dict2['channels']]
df2.set_axis(cols2, axis=1, inplace=True) #rename columns


#%% Pick which child to use

df      = df2
sr      = sr2

lum     = df['CH5_LGHT'] # luminence data
ecg     = df['CH1']      # ecg

accX    = df['CH7_X'].to_numpy() 
accY    = df['CH6_Y'].to_numpy()
accZ    = df['CH8_Z'].to_numpy()
acc     = np.sqrt(accX**2+accY**2+accZ**2)
acc_hpy = hp.remove_baseline_wander(acc, sr) # remove baseline
acc_0   = acc_hpy - np.mean(acc_hpy)  # demean

t_sens = np.arange(0,len(ecg))/sr
Ssuf = "_peaks.txt"
    
#print signal info (length, etc.)
print("\n\nFile: %s\nSignal %d.\n%d samples.\n%2.1f minutes long.\n\n" %(filename,Npath,len(ecg),len(ecg)/sr1/60 ))



#%% Finding start time
#Could use both signals, but I've got a simple say here to find the start (take top percentile of peaks and just find the first). Rewrite if needed

#Get luminescence data 
lum1 = df1['CH5_LGHT']
lum2 = df2['CH5_LGHT']


#Attempt 2 using diff method for steepest slope (seems better, keeping)

peaks1 = find_peaks(-np.diff(lum1), -100, distance=sr*10)
start_i = [ind for ind,x in enumerate(peaks1[1]['peak_heights']) if x>np.mean(peaks1[1]['peak_heights'])]
start_t = peaks1[0][start_i[0]]/sr

print("Obervations will be indexed from %3.2fs" % (start_t))

# To do - compare with signal 2, check width of peak using +np.diff(lum1)

#%% Mini Plot Function

def viz_ECG_mini(ax,ecg, peaks_new,sr=500, NZoom=[], Title = "Filtered Signal",col='plum'):
    if not NZoom:
        NZoom = [0,int(len(ecg)/sr)]
    plt.plot(np.arange(len(ecg))/sr,ecg, label = 'ECG signal', color=col,linewidth=2)
    plt.scatter([x/sr for x in peaks_new],[ecg[x] for x in peaks_new], color='green', label = 'identified peaks')
#plt.legend()
    ax.set_xlim(NZoom)
    ax.set_title(Title)
    
#%% Visualise and print basic information

fig=plt.figure(figsize=(10,8))
    
# NZ = [1300,1400]
# ax1 = plt.subplot(411)
# viz_ECG_mini(acc, [],sr=sr, NZoom=NZ, Title = "Total Acceleration",col='saddlebrown')
# ax1 = plt.subplot(412)
# viz_ECG_mini(accX, [],sr=sr, NZoom=NZ, Title = "X Acceleration",col='saddlebrown')
# ax1 = plt.subplot(413)
# viz_ECG_mini(accY, [],sr=sr, NZoom=NZ, Title = "Y Acceleration",col='saddlebrown')
# ax1 = plt.subplot(414)
# viz_ECG_mini(accZ, [],sr=sr, NZoom=NZ, Title = "Z Acceleration",col='saddlebrown')

NZ = [1300,1400]

gs=gridspec.GridSpec(4,1,height_ratios=[3,1,1,1])
ax0 = plt.subplot(gs[0])
# viz_ECG_mini(ax0,acc, [],sr=sr, NZoom=NZ, Title = " ",col='saddlebrown')
plt.setp(ax0.get_xticklabels(), visible=False)
plt.setp(ax0.get_yticklabels(), visible=False)

ax1= plt.subplot(gs[1], sharex=ax0)
viz_ECG_mini(ax1,accX, [],sr=sr, NZoom=NZ, Title = " ",col='saddlebrown')
plt.setp(ax1.get_xticklabels(), visible=False)

ax1= plt.subplot(gs[2], sharex=ax0)
viz_ECG_mini(ax1,accY, [],sr=sr, NZoom=NZ, Title = " ",col='saddlebrown')
plt.setp(ax1.get_xticklabels(), visible=False)

ax1= plt.subplot(gs[3], sharex=ax0)
viz_ECG_mini(ax1,accZ, [],sr=sr, NZoom=NZ, Title = " ",col='saddlebrown')
plt.setp(ax1.get_xticklabels(), visible=True)
#plt.grid(axis="x")
fig.tight_layout()
fig.subplots_adjust(hspace=0)

#%% Difference Information

plt.figure(figsize=(16,10))
    
NZ = [1300,1400]
ax1 = plt.subplot(411)
viz_ECG_mini(np.diff(acc), [],sr=sr, NZoom=NZ, Title = "Total Acceleration")
ax1 = plt.subplot(412)
viz_ECG_mini(np.diff(accX), [],sr=sr, NZoom=NZ, Title = "X Acceleration")
ax1 = plt.subplot(413)
viz_ECG_mini(np.diff(accY), [],sr=sr, NZoom=NZ, Title = "Y Acceleration")
ax1 = plt.subplot(414)
viz_ECG_mini(np.diff(accZ), [],sr=sr, NZoom=NZ, Title = "Z Acceleration")

#%% stats

acc_stats=spyst.describe(acc)
accX_stats=spyst.describe(accX)
accY_stats=spyst.describe(accY)
accZ_stats=spyst.describe(accZ)
print("Summary Stats.\nOverall Acceleration:")
print(acc_stats)
print("X Acceleration")
print(accX_stats)
print("Y Acceleration")
print(accY_stats)
print("Z Acceleration")
print(accZ_stats)



#%% HR, acc info

plt.figure(figsize=(16,10))
    
NZ = [1300,1400]
NZ = [300,320]
ax1 = plt.subplot(121)
viz_ECG_mini(acc, [],sr=sr, NZoom=NZ, Title = "Total Acceleration")
plt.grid()
ax1 = plt.subplot(122)
plt.plot(res_nk2['time_0'],res_nk2['hr'],color='red')
ax1.set_xlim(NZ)
ax1.set_title('HR')
plt.grid()

#%% ECG, light, acc info

plt.figure(figsize=(16,10))
    
NZ = [1300,1400]
NZ = [300,320]
ax1 = plt.subplot(121)
viz_ECG_mini(acc, [],sr=sr, NZoom=NZ, Title = "Total Acceleration")
plt.grid()
# ax1 = plt.subplot(312)
# viz_ECG_mini(ecg, [],sr=sr, NZoom=NZ, Title = "ECG")
ax1 = plt.subplot(122)
plt.plot(res_nk2['time_0'],res_nk2['hr'],color='red')
ax1.set_xlim(NZ)
ax1.set_title('HR')
plt.grid()
#viz_ECG_mini(lum, [],sr=sr, NZoom=NZ, Title = "Luminescence")

#%% Trying to plot on the same graph

fig, ax = plt.subplots()
NZ = [320,375]
# Plot linear sequence, and set tick labels to the same color
ax.plot(np.arange(0,len(acc))/sr,acc, color='red')
ax.tick_params(axis='y', labelcolor='red')
ax.set_xlim(NZ)

# Generate a new Axes instance, on the twin-X axes (same position)
ax2 = ax.twinx()

# change tick color
ax2.plot(res_nk2['time_0'],res_nk2['hr'], color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_xlim(NZ)

plt.show()

#%% Try removing baseline wander from acceleration


acc_hpy = hp.remove_baseline_wander(acc, sr)

plt.figure(figsize=(16,10))
    
NZ = [489,490]
#NZ = [100,200]
ax1 = plt.subplot(311)
viz_ECG_mini(acc, [],sr=sr, NZoom=NZ, Title = "Total Acceleration")
ax1 = plt.subplot(312)
viz_ECG_mini(acc_hpy, [],sr=sr, NZoom=NZ, Title = "Acceleration without baseline")
ax1 = plt.subplot(313)
viz_ECG_mini(acc_hpy-np.mean(acc_hpy), [],sr=sr, NZoom=NZ, Title = "Demeaned Acceleration without baseline")


#%% Removing High freq and low freq




acc_notch = hp.filter_signal(acc,cutoff = 0.05,sample_rate = sr,filtertype='notch')            
acc_hp = hp.filter_signal(acc,cutoff = 20,sample_rate=sr,     filtertype='highpass')


plt.figure(figsize=(16,10))
    
NZ = [1300,1400]
#NZ = [100,200]
ax1 = plt.subplot(311)
viz_ECG_mini(acc, [],sr=sr, NZoom=NZ, Title = "Total Acceleration")
ax1 = plt.subplot(312)
viz_ECG_mini(acc_notch, [],sr=sr, NZoom=NZ, Title = "Acceleration with notch")
ax1 = plt.subplot(313)
viz_ECG_mini(acc_hp, [],sr=sr, NZoom=NZ, Title = "Acceleration with high pass")



#%% Creating figure for report
sr_acc=sr
t_acc = np.arange(0,len(acc))/sr_acc
acc_hpy = hp.remove_baseline_wander(acc, sr_acc)
acc_dmn = acc_hpy - np.mean(acc_hpy)
acc_eng = np.square(acc_dmn)




plt.figure(figsize=(12,15))

NZ = [1700,1705]

freqs, psd = scipy.signal.welch(acc_dmn[NZ[0]*sr_acc:NZ[1]*sr_acc])

#NZ = [100,200]

ax1 = plt.subplot(311)
plt.plot(t_acc,acc,color='black')
#plt.title("Measured Acceleration")
plt.xlim(NZ) 
plt.ylim([min(acc[NZ[0]*sr_acc:NZ[1]*sr_acc])/1.001,max(acc[NZ[0]*sr_acc:NZ[1]*sr_acc])*1.001])
ax1 = plt.subplot(312)
plt.plot(t_acc,acc_eng)
plt.xlim(NZ) 
plt.ylim([min(acc_eng[NZ[0]*sr_acc:NZ[1]*sr_acc])/1.05,max(acc_eng[NZ[0]*sr_acc:NZ[1]*sr_acc])*1.05])

#plt.title("De-meaned Energy")
ax1 = plt.subplot(313)
plt.semilogx(freqs, psd,color='purple')
#plt.title('PSD: power spectral density')
#plt.xlabel('Frequency')
#plt.ylabel('Power')