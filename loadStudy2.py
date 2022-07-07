# -*- coding: utf-8 -*-
"""
Created on Mon May 30 12:33:01 2022

loadStudy2.py

Examine the new pilot study

@author: Dr Harry T. Mason, University of York
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
import skfda
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

#%% Initialise Strings
Ssesh = "B007"
#"B114","B201","B044", "B064", "B015"


#%% Load ECG

paths = hf_find('open*.txt', ("G:\\My Drive\\BackUp\\Documents\\Data\\Study2\\EyeTracker + ECG study\\%s\\ECG\\" % (Ssesh)));
paths.sort()
print(paths)
Npath = 0; # Now it's just the first signal
filename = paths[Npath] # stitch later
cols =  ["nSeq", "DI", "CH1", "CH2", "CH3_ZYG", "CH4_CORR", "CH5_LGHT", "CH6_Y", "CH7_X", "CH8_Z"];

lum     = pd.Series(dtype=int) # luminence data
ecg     = pd.Series(dtype=int)     # ecg
acc     = np.empty(0)
acc_hpy = np.empty(0)
acc_0   = np.empty(0)
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
    Ncol1       = len(dict1['column'])             # add the columns which exist
    df         = df.loc[:,0:Ncol1-1]                      #pandas file with everything
    sr         = dict1['sampling rate']
    cols1 = cols[0:2]+[cols[x+1] for x in dict1['channels']]
    df.set_axis(cols1, axis=1, inplace=True) #rename columns
    #for measure in dict1.keys():
    #    print('%s: %s' %(measure, dict1[measure]))
    begin1      =     dict1['time']
        
    time_stamps.append(datetime.strptime(dict1['time'],'%H:%M:%S.%f'))

    
    # append zeros to fill gaps
    start_i = int((time_stamps[-1] - time_stamps[0]).total_seconds()*sr) # if ecg recording had continued, this would be the index the recording started at
    
    lum     = pd.concat([lum,pd.Series(np.zeros(start_i-len(lum)))], ignore_index=True)
    lum     = pd.concat([lum,df['CH2']], ignore_index=True) # luminence data
    #lum     = pd.concat([lum,df['CH5_LGHT']], ignore_index=True) # luminence data
    #trig     = df['CH3_ZYG'].to_numpy() 
    ecg     = pd.concat([ecg,pd.Series(np.zeros(start_i-len(ecg)))], ignore_index=True)
    ecg     = pd.concat([ecg,df['CH1']], ignore_index=True)      # ecg
    # if Ssesh=="B039":
    #     ecg=-ecg
    accX    = df['CH7_X'].to_numpy() 
    accY    = df['CH6_Y'].to_numpy()
    #accZ    = df['CH8_Z'].to_numpy()
    accZ    = df['CH5_LGHT'].to_numpy()
    acc     = np.append(acc,np.sqrt(accX**2+accY**2+accZ**2))
    acc_hpy = np.append(acc_hpy,hp.remove_baseline_wander(np.sqrt(accX**2+accY**2+accZ**2), sr)) # remove baseline
    acc_0   = np.append(acc_0,hp.remove_baseline_wander(np.sqrt(accX**2+accY**2+accZ**2), sr) - np.mean(hp.remove_baseline_wander(np.sqrt(accX**2+accY**2+accZ**2), sr)))  # demean


t_sens = np.arange(0,len(ecg))/sr
    
#print signal info (length, etc.)
print("\n\nFile: %s\n%d samples.\n%2.1f minutes long.\n\n" %(filename,len(ecg),len(ecg)/sr/60 ))

#%% Initial processing
ecg=-ecg
#start_t=0

#%% Finding start time
lum300 = median_filter(lum[0:int(300*sr)],size=int(1*500),mode='constant')

bad_ind = np.intersect1d(np.where(lum300==0)[0],np.where(ecg==0)[0]) #Find where stitching occurs, so it wont be included

peaks_d = find_peaks(-np.diff(lum300), -100, distance=sr*10)  #Steep slopes down
peaks_u = find_peaks(np.diff(lum300), -100, distance=sr*10)  # steep slopes up

peaks_d_ind = np.setdiff1d(peaks_d[0],bad_ind) # Maybe bad_ind-1?
peaks_u_ind = np.setdiff1d(peaks_u[0],bad_ind)
peak_d_height = [peaks_d[1]['peak_heights'][ind] for ind,x in enumerate(peaks_d[0]) if np.isin(peaks_d_ind,x).any()]
peak_u_height = [peaks_u[1]['peak_heights'][ind] for ind,x in enumerate(peaks_u[0]) if np.isin(peaks_u_ind,x).any()]

max_peaks_d_ind = np.argpartition(peak_d_height,kth=-20)[-20:] # select 7 biggest peaks
max_peaks_u_ind = np.argpartition(peak_u_height,kth=-20)[-20:] # select 7 biggest peaks

# Now, find first slope (out of largest N peaks) that has a big upslope in the 1-8s after it
peak_candidates = []
for x in peaks_d_ind[max_peaks_d_ind]:
    if x+sr <= max(peaks_u_ind[max_peaks_u_ind]):    # make sure a valid up peak exists after 
        min_gap = min([y-x for y in peaks_u_ind[max_peaks_u_ind] if y-x >= sr*1])
        if min_gap<(8*sr):
            peak_candidates.append(x)
if len(peak_candidates)==0:
    start_t=0 
    print('Warning! No clear peaks found in luminosity data. Syncing may be an issue.')
elif len(peak_candidates)>1:
    start_t = [peaks_d[0][ind] for ind,x in enumerate(peaks_d[1]['peak_heights']) if x==max([lum300[x]-lum300[x+1] for x in peak_candidates])][0]/sr
else:
    start_t = peak_candidates[0]/sr
    #start_t = min(peak_candidates)/sr
    # maximum luminosity gap


#%% Plot Luminence

NZ = [150,250]#len(lum)/sr]

fig = plt.figure(figsize=(12,8))
ax1=plt.subplot(2,1,1)
plt.plot(t_sens,lum)
plt.plot([start_t,start_t],[np.min(lum),np.max(lum)],linewidth=4,color='black')
for x in peak_candidates:
    plt.plot([x/sr,x/sr],[np.min(lum),np.max(lum)])
ax1.set_xlim(NZ)
ax1.set_ylim([np.min(lum),np.max(lum)])
ax1.set_title('Light signal and start time (with other peak candidates), %s'% Ssesh)
ax1=plt.subplot(2,1,2)
plt.plot(t_sens[0:len(lum300)],lum300)
plt.plot([start_t,start_t],[np.min(lum300),np.max(lum300)],linewidth=4,color='black')
for x in peak_candidates:
    plt.plot([x/sr,x/sr],[np.min(lum300),np.max(lum300)])
ax1.set_xlim(NZ)
ax1.set_title('Filtered Light signal and start time (with other peak candidates), %s'% Ssesh)



# #%% Load sustained Attention
# Npath_mat = 0
# paths_sus_mat = hf_find(('Sustained*%s*%s%s*.mat' % (Ssesh)), "G:\\My Drive\\BackUp\\Documents\\Data\\LabelledData\\") #finds xlsx files that aren't being worked on (ones being work on have ~$ in front)

# print(paths_sus_mat)
# paths_sus_mat.sort()
# filename_sus_mat = paths_sus_mat[Npath_mat] #keeps the same as the other labels


# df_sus_mat = sio.loadmat(filename_sus_mat)
# #print(df_sus_mat['data'].dtype) # names of variables
# sr_att=df_sus_mat['data']['FrameRate'][0][0][0][0]
# sr_att=np.round(sr_att)

# sus_label = df_sus_mat['data']['events'][0][0]
# sus_label = [x[0][0] for x in sus_label]
# # #sus_label=['sustained_attention', 'something_else', 'NO CHANGE']
# sus_att = df_sus_mat['data']['data'][0][0] #equivalent to the excel

# sus_label=sus_label[0:2]# sometimes there's a third column for some reason
# sus_att=sus_att[:,0:2] # sometimes there's a third column for some reason

# t_att=np.arange(0,len(sus_att[:,0]))/sr_att

# #%% Bar Plot the information
# fig=plt.figure(figsize=(10,5))
# ax=plt.subplot(1,1,1)
# plt.bar(np.arange(0,2),np.sum(sus_att,0)/sr_att)
# plt.xticks(rotation=0)

# ax.set_ylabel('Number of Seconds of Data',fontsize=18)
# ax.set_xticks([0,1])
# ax.set_xticklabels(sus_label[0:2])
# plt.title('Prominance of Sustained Attention',fontsize=24)


# #%% Show all labels

# fig = plt.figure(figsize=(16,12))
# gs=gridspec.GridSpec(2,1,height_ratios=[1,3])

# a0= plt.subplot(gs[0])
# a0.plot(t_att/60,np.sum(sus_att,1))
# a0.set_title('Sum of Signals')
# a1= plt.subplot(gs[1])
# #a1.set_prop_cycle('color',cmvir(2))
# a1.plot(t_att/60,sus_att+[0,2], linewidth=4)
# #a1.plot(t_att/60,sus_att[:,0], linewidth=2,color='black')
# a1.set_yticks([0.5,2.5]) #
# a1.set_yticklabels(sus_label[0:2])
# a1.set_title('Sustained Attention')
# a1.set_xlabel('Time (minutes)')

#%% Mini Plot Function

def viz_ECG_mini(ax,ecg_sig, peaks_new,time,sr=sr, NZoom=[], Title = "Filtered Signal",ylab="",xlab='Time (seconds)',col="plum"):
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
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab,fontsize=22)

    
#%% Visualise and print basic information

plt.figure(figsize=(8,3))
    
NZ = [180,190]
ax1 = plt.subplot(111)
viz_ECG_mini(ax1,ecg, [],t_sens,sr=sr, NZoom=NZ, Title = "Raw ECG")
#ax1.set_ylim([12000,55000])

#%% Do various ECG preprocessing

#NK2 processing (with custom additions)
#ecg_nk2 = nk.signal_detrend(ecg)
ecg_nk2 = hf_ecg(ecg,sr=sr,freqfilt=[0.5,20])           

plt.figure(figsize=(8,3))
#NZ = [1320,1350]
ax1 = plt.subplot(111)
viz_ECG_mini(ax1,ecg_nk2, [],t_sens,sr=sr, NZoom=NZ, Title = "Preprocessing")
#ax1.set_ylim([-12000,15000])


#%% Peak function
#peak calculation

r_nk2 = nk.ecg_findpeaks(ecg_nk2, sampling_rate=sr, method="neurokit")
r_nk2 = np.asarray(r_nk2['ECG_R_Peaks'])
t_r_nk2 = t_sens[r_nk2]

plt.figure(figsize=(8,3))
#NZ = [1320,1350]
ax1 = plt.subplot(111)
viz_ECG_mini(ax1,ecg_nk2, r_nk2,t_sens,sr=sr, NZoom=NZ, Title = "Preprocessing with Peaks")
#ax1.set_ylim([-12000,15000])

plt.figure(figsize=(8,3))
#NZ = [1320,1350]
ax1 = plt.subplot(111)
viz_ECG_mini(ax1,ecg, r_nk2,t_sens,sr=sr, NZoom=NZ, Title = "Raw ECG with Preprocessing Peaks")
#ax1.set_ylim([12000,55000])


#%% HR calcs

hr_nk2 = 60/np.diff(r_nk2/sr)
# hr_hpy = 60/np.diff(r_hpy/sr)
t_hr_nk2 = t_r_nk2[0:-1]+np.diff(t_r_nk2)/2
# t_hr_hpy = t_r_hpy[0:-1]+np.diff(t_r_hpy)/2


#%% Clean nk2 peaks

r_nk3  = clean_peaks(r_nk2,ecg)
t_r_nk3 = t_sens[r_nk3]
hr_nk3 = 60/np.diff(np.asarray(r_nk3)/sr)
t_hr_nk3 = t_r_nk3[0:-1]+np.diff(t_r_nk3)/2

#%% EMD denoising

emd = EMD()
ecg_emd = hf_ecg(ecg,sr=sr,freqfilt=[0.01,40])    
ecg_imfs = emd.emd(np.array(ecg_emd)) # Pal wants me to remove baseline wander first
#-np.mean(ecg)
#%%  IMF visualisation plot

NZ=[143,145]

fig = plt.figure(figsize=(12,9))
gs=gridspec.GridSpec(7,4,height_ratios=[3,1,2,2,2,2,2],width_ratios = [2,2,2,2])

a0= plt.subplot(gs[0:4])
viz_ECG_mini(a0,ecg, [],t_sens,sr=sr, NZoom=NZ, Title ="",ylab= "ECG",col='black')
plt.setp(a0.get_yticklabels(), visible=True)

a1= plt.subplot(gs[4:7])
plt.setp(a1.get_xticklabels(), visible=False)
plt.setp(a1.get_yticklabels(), visible=False)
a1.axis('off')

for x in np.arange(0,min([int(len(ecg_imfs)/4)*4,16])):
    ax = plt.subplot(gs[8+x])
    viz_ECG_mini(ax,ecg_imfs[x] ,[],t_sens,sr=sr, NZoom=NZ, Title ="",ylab= ("IMF %d" % (x+1)),xlab="Time",col='black')
#     plt.setp(ax.get_yticklabels(), visible=False) 


fig.tight_layout()
fig.subplots_adjust(hspace=0.0)

#%% Frequency content of imfs

ecg_imfs_k = scipy.fft.fft(ecg_imfs)
ecg_imfs_kt = scipy.fft.fftfreq(len(ecg_imfs[0]),1/sr)

NZ=[0.1,1]

fig = plt.figure(figsize=(12,7))
gs=gridspec.GridSpec(6,4,height_ratios=[3,1,2,2,2,2],width_ratios = [2,2,2,2])

a0= plt.subplot(gs[0:4])
viz_ECG_mini(a0,scipy.fft.fft(ecg), [],ecg_imfs_kt,sr=sr, NZoom=NZ, Title ="",ylab= "ECG", xlab="freq",col='black')
plt.setp(a0.get_yticklabels(), visible=True)

a1= plt.subplot(gs[4:7])
plt.setp(a1.get_xticklabels(), visible=False)
plt.setp(a1.get_yticklabels(), visible=False)
a1.axis('off')

for x in np.arange(0,min([int(len(ecg_imfs_k)/4)*4,16])):
    ax = plt.subplot(gs[8+x])
    viz_ECG_mini(ax,ecg_imfs_k[x] ,[],ecg_imfs_kt,sr=sr, NZoom=NZ, Title ="",ylab= ("IMF %d" % (x+1)),xlab="Freq",col='black')
    plt.setp(ax.get_yticklabels(), visible=False) 

fig.tight_layout()
fig.subplots_adjust(hspace=0.0)

#%% Power calculations
ecg_imfs_pow = [nk.signal_psd(x,method="welch",min_frequency=0, max_frequency=160) for x in ecg_imfs]
ecg_pow = nk.signal_psd(ecg,method="welch",min_frequency=0, max_frequency=160)

NZ=[0,5]

fig = plt.figure(figsize=(12,7))
gs=gridspec.GridSpec(7,4,height_ratios=[3,1,2,2,2,2,2],width_ratios = [2,2,2,2])

a0= plt.subplot(gs[0:4])
viz_ECG_mini(a0,ecg_pow['Power'], [],ecg_pow['Frequency'],sr=sr, NZoom=NZ, Title ="",ylab= "ECG Power",xlab="Frequency",col='black')
plt.setp(a0.get_yticklabels(), visible=True)

a1= plt.subplot(gs[4:7])
plt.setp(a1.get_xticklabels(), visible=False)
plt.setp(a1.get_yticklabels(), visible=False)
a1.axis('off')

for x in np.arange(0,min([int(len(ecg_imfs_pow)/4)*4,20])):
    ax = plt.subplot(gs[8+x])
    viz_ECG_mini(ax,ecg_imfs_pow[x]['Power'] ,[],ecg_imfs_pow[x]['Frequency'],sr=sr, NZoom=NZ, Title ="",ylab= ("IMF %d" % (x+1)),xlab="Freq",col='black')
    plt.setp(ax.get_yticklabels(), visible=False) 

fig.tight_layout()
fig.subplots_adjust(hspace=0.0)
#%% Find IMFS with most of the energy above 0.5Hz
ind_low = np.where(ecg_imfs_pow[0]['Frequency']>0.5)[0][0] # -ind_05 for the negative index
ind_high = np.where(ecg_imfs_pow[0]['Frequency']>8)[0][0] # -ind_05 for the negative index
P_05 = np.zeros(len(ecg_imfs))
for ind, x in enumerate(ecg_imfs_pow):
    P_05[ind] = x['Power'][ind_low:ind_high].sum()/x['Power'].sum()
    
    
# #%% Trying out the Pal formulation?
# Ak = np.zeros(len(ecg_imfs))
# Pk = np.zeros(len(ecg_imfs))
# for ind, x in enumerate(ecg_imfs):
#     Ak[ind] = Ak[ind-1] + (np.mean(x))
#     Pk[ind] = 10*np.log( np.sum([np.abs(ecg_imfs[y])**2 for y in np.arange(0,ind+1)]))
#     print(Ak[ind],Pk[ind])#np.sum(np.abs(x-np.mean(x))**2)**0.5

#%% Rebuilding with relevant IMFs
print("removing IMFS:"+str(np.where(P_05>=0.5)[0]))
ecg_emd = np.sum(ecg_imfs[np.where(P_05<0.5)[0]],axis=0) # +np.mean(ecg)

#ecg_emd = np.sum(ecg_imfs[[0,1,3,4,5,6,7]],axis=0)
# plt.figure(figsize=(8,3))
NZ = [190,210]
# ax1 = plt.subplot(111)
#viz_ECG_mini(ax1,ecg_emd, [],t_sens,sr=sr, NZoom=NZ, Title = "Preprocessing")
#ax1.set_ylim([-12000,15000])


#%% Peak function
#peak calculation

r_emd = nk.ecg_findpeaks(ecg_emd, sampling_rate=sr, method="neurokit")
r_emd = np.asarray(r_emd['ECG_R_Peaks'])
t_r_emd = t_sens[r_emd]

plt.figure(figsize=(8,3))
#NZ = [1320,1350]
ax1 = plt.subplot(111)
viz_ECG_mini(ax1,ecg_emd, r_emd,t_sens,sr=sr, NZoom=NZ, Title = "Preprocessing with Peaks")
#ax1.set_ylim([-12000,15000])

plt.figure(figsize=(8,3))
#NZ = [1320,1350]
ax1 = plt.subplot(111)
viz_ECG_mini(ax1,ecg, r_emd,t_sens,sr=sr, NZoom=NZ, Title = "Raw ECG with Preprocessing Peaks")
#ax1.set_ylim([12000,55000])


#%% HR calcs

hr_emd = 60/np.diff(r_emd/sr)
# hr_hpy = 60/np.diff(r_hpy/sr)
t_hr_emd = t_r_emd[0:-1]+np.diff(t_r_emd)/2
# t_hr_hpy = t_r_hpy[0:-1]+np.diff(t_r_hpy)/2


#%% Clean emd peaks

r_emd2  = clean_peaks(r_emd,ecg)
t_r_emd2 = t_sens[r_emd2]
hr_emd2 = 60/np.diff(np.asarray(r_emd2)/sr)
t_hr_emd2 = t_r_emd2[0:-1]+np.diff(t_r_emd2)/2

#%% Visualise
plt.figure(figsize=(8,3))
NZ = [1020,1030]
ax1 = plt.subplot(111)
viz_ECG_mini(ax1,ecg, r_emd2,t_sens,sr=sr, NZoom=NZ, Title = "Raw ECG with Corrected Peaks")
#ax1.set_ylim([12000,55000])
plt.grid(axis='x')


#%% Median filter and add spline

hr_nk3_filt, qual_hr = hrSQI(hr_nk3,sr=sr)

hr_nk3_fun = UnivariateSpline(t_hr_nk3,hr_nk3_filt, k=5)
hr_nk3_spl = hr_nk3_fun(t_hr_nk3)

fig, ax = plt.subplots(figsize=(12,4))
NZ = [0,t_hr_nk3[-1]]
# Plot linear sequence, and set tick labels to the same color
ax.plot(t_hr_nk3-start_t,hr_nk3_filt, color='red',marker='.',label='Filtered HR')
ax.plot(t_hr_nk3-start_t,hr_nk3,label='HR Derived from Peaks', color='green')
ax.plot(t_hr_nk3-start_t,hr_nk3_spl,label='Splined HR', color='blue')
ax.set_xlim(NZ)
#ax.legend(loc='upper right')
ax.set_title('%s, Median-Filtered Spline' % (Ssesh))#'Original Heart Rate (red) with Spline (blue)')
#ax.set_ylim([110, 210])
ax.set_ylabel('Heart Rate',fontsize=14)
ax.set_xlabel('Time (seconds)',fontsize=14)
plt.grid()
plt.legend()
plt.show()

#%% Quality Check

fig, ax = plt.subplots(figsize=(12,4))
NZ = [0-start_t,t_hr_nk3[-1]-start_t]
# Plot linear sequence, and set tick labels to the same color
ax.plot(t_hr_nk3-start_t,qual_hr[:,0], color='purple',marker='.',label='Distance beyond expected boundary')
ax.plot(t_hr_nk3-start_t,qual_hr[:,1], color='brown',marker='.',label='Proportion of good heart rates within boundary')
ax.set_xlim(NZ)
#ax.legend(loc='upper right')
ax.set_title('%s, Quality Check' % (Ssesh))#'Original Heart Rate (red) with Spline (blue)')
#ax.set_ylim([110, 210])
ax.set_ylabel('Heart Rate',fontsize=14)
ax.set_xlabel('Time (seconds)',fontsize=14)
plt.grid()
plt.legend()
plt.show()


#%% Median filter and add spline

hr_emd2_filt, qual_hr = hrSQI(hr_emd2,sr=sr)

hr_emd2_fun = UnivariateSpline(t_hr_emd2,hr_emd2_filt, k=5)
hr_emd2_spl = hr_emd2_fun(t_hr_emd2)

fig, ax = plt.subplots(figsize=(12,4))
NZ = [0-start_t,t_hr_emd2[-1]-start_t]
# Plot linear sequence, and set tick labels to the same color
ax.plot(t_hr_emd2-start_t,hr_emd2_filt, color='red',marker='.',label='Filtered HR')
ax.plot(t_hr_emd2-start_t,hr_emd2,label='HR Derived from Peaks', color='green')
ax.plot(t_hr_emd2-start_t,hr_emd2_spl,label='Splined HR', color='blue')
ax.set_xlim(NZ)
#ax.legend(loc='upper right')
ax.set_title('%s, EMD peaks, Median-Filtered Spline' % (Ssesh))#'Original Heart Rate (red) with Spline (blue)')
#ax.set_ylim([110, 210])
ax.set_ylabel('Heart Rate',fontsize=14)
ax.set_xlabel('Time (seconds)',fontsize=14)
plt.grid()
plt.legend()
plt.show()

#%% Quality Check

fig, ax = plt.subplots(figsize=(12,4))
NZ = [0-start_t,t_hr_emd2[-1]-start_t]
# Plot linear sequence, and set tick labels to the same color
ax.plot(t_hr_emd2-start_t,qual_hr[:,0], color='purple',marker='.',label='Distance beyond expected boundary')
ax.plot(t_hr_emd2-start_t,qual_hr[:,1], color='brown',marker='.',label='Proportion of good heart rates within boundary')
ax.set_xlim(NZ)
#ax.legend(loc='upper right')
ax.set_title('%s, Quality Check' % (Ssesh))#'Original Heart Rate (red) with Spline (blue)')
#ax.set_ylim([110, 210])
ax.set_ylabel('Heart Rate',fontsize=14)
ax.set_xlabel('Time (seconds)',fontsize=14)
ax.set_ylim([0,1])
plt.grid()
plt.legend()
plt.show()

#%% Create objects for smoothing

grid_points = t_hr_emd2  # Grid points of the curves
data_matrix = hr_emd2_filt

fd = skfda.FDataGrid(
    data_matrix=data_matrix,
    grid_points=grid_points,
)

#fd.plot()
#plt.show()

#%% Smoothing 

import skfda.preprocessing.smoothing as ks

   
NZ = [1000,1510]
YZoom = [80,180]
# Plot original
plt.figure(figsize=(12,4))
a0 = plt.subplot(1,1,1)
fd.plot(chart=a0)
plt.ylabel('Original HR, %s\n No Smoothing' % (Ssesh),fontsize=18)
plt.xlabel('Time (seconds)')
plt.xlim(NZ)
plt.ylim(YZoom)
plt.grid(axis='x')
plt.show()

Nbasis = int(np.ceil(np.ptp(t_hr_emd)/2.5))  # range of time / 5, roughly one basis per 5 seconds of signal
basis = skfda.representation.basis.BSpline(n_basis=Nbasis)
fd_basis = fd.to_basis(basis)
fd_back  = fd_basis.to_grid()
smoother = ks.kernel_smoothers.NadarayaWatsonSmoother(smoothing_parameter=(np.mean(hr_emd2_filt)/60))
fd_back_smooth = smoother.fit_transform(fd_back)
plt.figure(figsize=(12,4))
a1 = plt.subplot(1,1,1)
fd_back.plot(chart=a1)
plt.ylabel('basis HR, %s\nLocalLinearRegressor' % (Ssesh),fontsize=18)
plt.xlabel('Time (seconds)')
plt.xlim(NZ)
plt.ylim(YZoom)
plt.grid(axis='x')
plt.show()

plt.figure(figsize=(12,4))
a2 = plt.subplot(1,1,1)
fd_back_smooth.plot(chart=a2)
plt.ylabel('Smoothed basis HR, %s\nLocalLinearRegressor' % (Ssesh),fontsize=18)
plt.xlabel('Time (seconds)')
plt.xlim(NZ)
plt.ylim(YZoom)
plt.grid(axis='x')
plt.show()

#%% CREATE DIFFERENTIAL OF SMOOTHED SIGNAL
fd_smooth_diff = fd_back_smooth.derivative(order=1)

plt.figure(figsize=(12,4))
a2 = plt.subplot(1,1,1)
fd_smooth_diff.plot(chart=a2)
plt.ylabel('Smoothed basis HR, %s\nDifferential' % (Ssesh),fontsize=18)
plt.xlabel('Time (seconds)')
plt.xlim(NZ)
#plt.ylim(YZoom)
plt.grid(axis='x')
plt.show()
#%% COnvert back to numpy form

hr_smo = fd_back_smooth.data_matrix[0]
hr_smo = hr_smo.reshape(len(hr_smo),)
t_smo = fd_back_smooth.grid_points[0]

hr_smo_diff = fd_smooth_diff.data_matrix[0]
hr_smo_diff = hr_smo_diff.reshape(len(hr_smo_diff),)


#%% Noise calculation

signal = ecg
fs = int(sr)
time = t_sens

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
NZ=[0,310]
cp_plot(hr_nk3_spl,y_pred_cps,t_hr_nk3,NZoom=NZ,ylab=('Bad HR check, %s' % (Ssesh)))



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


#%% Plot HR, ECG, acc_0, and labels
NZ=[00,100]

fig = plt.figure(figsize=(16,12))
gs=gridspec.GridSpec(4,1,height_ratios=[1,1,1,3])

a0= plt.subplot(gs[0])
plt_signal(a0,t_hr_nk3-start_t,hr_nk3_spl,ylab='Splined HR\n')

a1= plt.subplot(gs[1], sharex=a0)
plt_signal(a1,t_sens-start_t,ecg,ylab='ECG with\n Peaks',col='plum')
a1.scatter(t_r_nk3-start_t,ecg[r_nk3], color='green', label = 'identified peaks')

a2= plt.subplot(gs[2], sharex=a0)
plt_signal(a2,t_sens-start_t,acc_0,ylab='Acceleration\n Data',col='black')

a3= plt.subplot(gs[3], sharex=a0)
plt_sa(a3,t_att,sus_att[:,0],ylab='Sustained Attention',NZoom=NZ)
#plt_labels(a3,t_att,data[:,:6],events,ylab='The Separate Signals',NZoom=NZ)


#%% Plot HR, acc_0, labels
NZ=[0,100]

fig = plt.figure(figsize=(16,12))
gs=gridspec.GridSpec(3,1,height_ratios=[1,1,3])

a0= plt.subplot(gs[0])
plt_signal(a0,t_hr_nk3-start_t,hr_nk3_spl,ylab='Splined HR\n')

a1= plt.subplot(gs[1], sharex=a0)
plt_signal(a1,t_sens-start_t,acc_0,ylab='Acceleration\n Data',col='black')

a2= plt.subplot(gs[2], sharex=a0)
plt_sa(a2,t_att,sus_att[:,0],ylab='Sustained Attention',NZoom=NZ)
#plt_labels(a2,t_att,data[:,:6],events,ylab='The Separate Signals',NZoom=NZ)


#%% Create Boolean of focused attention
sus_att_boo = sus_att!=0

data_boo = data!=0
data_boo = data_boo[:,0:6]

p_s = data_boo[:,0] # passive: social
a_s = data_boo[:,1] # active: social
p_o = data_boo[:,2] # passive: object
a_o = data_boo[:,3] # active: object
a_l = data_boo[:,4] # active: looking away
p_l = data_boo[:,5] # passive: looking away



#idx = np.searchsorted(t_sens-start_t,t_att[f_a], side="left")

# Demonstrate principle just on focused attention
idx = np.searchsorted(t_hr_nk3-start_t,t_att[a_o], side="left")
hr_a_o = hr_nk3_spl[idx]
hr_a_o_tot = np.mean(hr_a_o)


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
        yrange = ([min(0,np.min(data_mean-data_std)-np.abs(np.max(data_mean+data_std))*0.05),max(0,np.max(data_mean+data_std)+np.abs(np.max(data_mean+data_std))*0.05)])
    else:
        plt.bar(np.arange(0,Ncol),data_mean)
        yrange = ([min(0,np.min(data_mean)-np.abs(np.max(data_mean))*0.05),max(0,np.max(data_mean)+np.abs(np.max(data_mean))*0.05)])
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

#%% Look at heart rate characteristics in each group

# Calculate for Sustained attention
hr_sus,hr_sus_mean,hr_sus_std = mat_stats(hr_nk3_fun(t_att), sus_att_boo,[[0],[1]])

# Calculate for all groups
hr_att,hr_att_mean,hr_att_std = mat_stats(hr_nk3_fun(t_att), data_boo, list(np.vstack(np.arange(0,6))))
hr_grp,hr_grp_mean,hr_grp_std = mat_stats(hr_nk3_fun(t_att), data_boo, [[0,1],[2,3],[4,5]])
hr_act,hr_act_mean,hr_act_std = mat_stats(hr_nk3_fun(t_att), data_boo, [[0,2,5],[1,3,4]])


#%% Plot HR bar charts

# Calculate for Sustained attention
ax=plt_bar(hr_sus_mean,hr_sus_std,['sustained_attention', 'something_else'],f_std=True,title="Mean Heart Rate during Sustained Attention",ylab="Mean Heart Rate")

ax=plt_bar(hr_att_mean,hr_att_std,events[0:6],f_std=True,title="Mean Heart Rate of different labels",ylab="Mean Heart Rate")
#plt.ylim([120,150])
ax=plt_bar(hr_grp_mean,hr_grp_std,['object','social','looking'],f_std=True,title="Mean Heart Rate of different groups",ylab="Mean Heart Rate")
#plt.ylim([120,150])
ax=plt_bar(hr_act_mean,hr_act_std,['passive','active'],f_std=True,title="Mean Heart Rate of active/passive",ylab="Mean Heart Rate")
#plt.ylim([120,150])

    
#%% Look at Heart Rate Change

dhr_nk3_spl = np.diff(hr_nk3_spl)
t_dhr_nk3 = t_hr_nk3[:-1]+np.diff(t_hr_nk3)/2
dhr_att_spl = np.diff(hr_nk3_fun(t_att))*sr_att # (*rate) is the same as (/dt)

# Calculate for all groups
dhr_sus,dhr_sus_mean,dhr_sus_std = mat_stats(dhr_att_spl, sus_att_boo,[[0],[1]])
dhr_att,dhr_att_mean,dhr_att_std = mat_stats(dhr_att_spl, data_boo, [[0],[1],[2],[3],[4],[5]])
dhr_grp,dhr_grp_mean,dhr_grp_std = mat_stats(dhr_att_spl, data_boo, [[0,1],[2,3],[4,5]])
dhr_act,dhr_act_mean,dhr_act_std = mat_stats(dhr_att_spl, data_boo, [[0,2,5],[1,3,4]])

#%% Plot dHR bar charts
    
# ax=plt_bar(dhr_att_mean,dhr_att_std,events[0:6],f_std=True,title="Mean Heart Rate Change of different labels",ylab="Mean Heart Rate Change")
# #plt.ylim([120,150])
# ax=plt_bar(dhr_grp_mean,dhr_grp_std,['object','social','looking'],f_std=True,title="Mean Heart Rate Change of different groups",ylab="Mean Heart Rate Change")
# #plt.ylim([120,150])
# ax=plt_bar(dhr_act_mean,dhr_act_std,['passive','active'],f_std=True,title="Mean Heart Rate Change of active/passive",ylab="Mean Heart Rate Change")
# #plt.ylim([120,150])
# ax=plt_bar(dhr_sus_mean,dhr_sus_std,['sustained_attention', 'something_else'],f_std=True,title="Mean Heart Rate Change during Sustained Attention",ylab="Mean Heart Rate Change")

#%% Plot Heart Rate Change Bar Charts
ax=plt_bar(dhr_sus_mean,dhr_sus_std,['sustained\nattention', 'something\nelse'],f_std=False,title="Mean Heart Rate Change\nduring Sustained Attention",ylab="bpm/second")

ax=plt_bar(dhr_att_mean,dhr_att_std,events[0:6],f_std=False,title="Mean Heart Rate Change during different labels",ylab="Mean Heart Rate Change/second")

ax=plt_bar(dhr_grp_mean,dhr_grp_std,['object','social','looking'],f_std=False,title="Mean Heart Rate Change during object/social/looking groups",ylab="Mean Heart Rate Change/second")
#plt.ylim([-0.5,0.5])

ax=plt_bar(dhr_act_mean,dhr_act_std,['passive','active'],f_std=False,title="Mean Heart Rate Change during active/passive",ylab="Mean Heart Rate Change/second")
#plt.ylim([-0.5,0.5])

    
#%% Statistical test (only two, so hopefully easy)
print('passive vs active')
print(scipy.stats.ttest_ind(dhr_act[0],dhr_act[1])[1])
print('\nobject vs social')
print(scipy.stats.ttest_ind(dhr_grp[0],dhr_grp[1])[1])
print('\nobject vs looking')
print(scipy.stats.ttest_ind(dhr_grp[0],dhr_grp[2])[1])
print('\nsocial vs looking')
print(scipy.stats.ttest_ind(dhr_grp[1],dhr_grp[2])[1])
print('\nSustained Attention vs Not')
print(scipy.stats.ttest_ind(dhr_sus[0],dhr_sus[1])[1])

#%% Anova statistical test

# Run the ANOVA
# aov = pg.anova(data=df, dv='Scores', between='Group', detailed=True)
# print(aov)

# ANOVa requires equal sized distributions. Could regenerate based on measured mean and standard dsitribution, but that seems disingenuous (although a great way to check for degree of normality)

#This is essentially it though
fvalue, pvalue = scipy.stats.f_oneway(dhr_grp[0], dhr_grp[1],dhr_grp[2])

#%% Autocorrelation test
# tmp = np.random.randn(1010)
lb_hr = acorr_ljungbox(hr_nk3_filt)

# from statsmodels.graphics.tsaplots import plot_acf
# plot_acf(hr_nk3_filt,lags=50) #shaded region is confidence interval, lags below this have no significant value

#lb_hr_nk3 = acorr_ljungbox(hr_nk3[0:1000]+hr_nk3[1:1001]) # returns two array, displaying the test statistic and the p-value for autocorrelation lags. I found no significance when I tried it

#%% GLM model?

# # Construct X with intercept
# data_glm = np.concatenate((data,np.ones((len(data),1))),axis=1) #adding an intercept to the model

# # Generate true beta
# beta = np.random.randn(9)

# ## FIT STUFF ##
# #beta_hat = np.linalg.inv(data_glm.T @ data_glm) @ data_glm.T @ dhr_att_spl[1:]
# beta_hat = np.matmul(np.linalg.pinv(data_glm[1:]),dhr_att_spl)

# ## PREDICT STUFF ##
# y_hat = np.sum(data_glm* beta_hat,axis=1)
# glm_model = sm.GLM(dhr_att_spl, data_glm[1:],family=sm.families.Gaussian)
# glm_results = glm_model.fit()

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
#bkps_hr = hf_cp(hr_smo,cp_type='cptmeanvar',time=t_smo)
bkps_hr = hf_cp(hr_emd2_filt,cp_type='linear',time=t_hr_emd2)
NZ=[1000,1300]
#%% Plotting options
#plt.figure(figsize=(12,4))
# NZ=[0,100]
cp_plot(hr_emd2_filt,bkps_hr,t_hr_emd2-start_t,NZoom=NZ,ylab='Heart Rate Changepoint\nOld Smoothing')
#cp_plot(hr_smo,bkps_hr,t_smo-start_t,NZoom=NZ,ylab='Heart Rate Changepoint\nBasis-Smoothed')

#rpt.display(signal, result, figsize=(10, 6))
#plt.title('Change Point Detection: Pelt Search Method')
#plt.show()  

#%% Changepoint and Labels plot for heart rate
NZ=[]

fig = plt.figure(figsize=(16,12))
gs=gridspec.GridSpec(3,1,height_ratios=[3,2,2])

a0= plt.subplot(gs[0])
plt_cp(a0,hr_emd2_filt,bkps_hr,t_hr_emd2-start_t,NZoom=NZ,ylab='Change Point Detection\nPelt Search Method\n',title="Heart Rate")


#Sustained attention
a1= plt.subplot(gs[1],sharex=a0)
# plt_sa(a1,t_att,sus_att[:,0],NZoom=NZ)
# plt.setp(a1.get_xticklabels(), visible=False)
# plt_cp_bp(a1,bkps_hr,t_hr_nk3-start_t,NLim=[-0.5,1.5])

# Labelled data
a2= plt.subplot(gs[2], sharex=a0)
# plt_labels(a2,t_att,data[:,:6],events,ylab='The Separate Signals',NZoom=NZ)
# plt_cp_bp(a2,bkps_hr,t_hr_nk3-start_t,NLim=[-0.5,11.5])


#%% Changepoint Analysis for HR smooth
bkps_hr_smo = hf_cp(hr_smo,cp_type='linear',time=t_smo)
NZ=[100,300]
#%% Plotting options HR smooth
#plt.figure(figsize=(12,4))
# NZ=[0,100]
cp_plot(hr_smo,bkps_hr_smo,t_smo-start_t,NZoom=NZ,ylab='Heart Rate Changepoint\nBasis-Smoothed')

#rpt.display(signal, result, figsize=(10, 6))
#plt.title('Change Point Detection: Pelt Search Method')
#plt.show()  

#%% Changepoint and Labels plot for heart rate smooth
NZ=[]

fig = plt.figure(figsize=(16,12))
gs=gridspec.GridSpec(3,1,height_ratios=[3,2,2])

a0= plt.subplot(gs[0])
plt_cp(a0,hr_smo,bkps_hr_smo,t_smo-start_t,NZoom=NZ,ylab='Change Point Detection\nPelt Search Method\n',title="Heart Rate Smooth")


#Sustained attention
a1= plt.subplot(gs[1],sharex=a0)
# plt_sa(a1,t_att,sus_att[:,0],NZoom=NZ)
# plt.setp(a1.get_xticklabels(), visible=False)
# plt_cp_bp(a1,bkps_hr,t_hr_nk3-start_t,NLim=[-0.5,1.5])

# Labelled data
a2= plt.subplot(gs[2], sharex=a0)
# plt_labels(a2,t_att,data[:,:6],events,ylab='The Separate Signals',NZoom=NZ)
# plt_cp_bp(a2,bkps_hr,t_hr_nk3-start_t,NLim=[-0.5,11.5])



#%% Changepoint Analysis for dHR
#bkps_dhr = hf_cp(dhr_nk3_spl,cp_type='l1',time=t_dhr_nk3)
bkps_dhr_smo = hf_cp(hr_smo_diff,cp_type='linear',time=t_smo)
NZ=[0,300]

cp_plot(hr_smo_diff,bkps_dhr_smo,t_smo-start_t,NZoom=NZ,ylab='Heart Rate Changepoint\nDiff of Smoothing')

#%% Changepoint and Labels plot for heart rate change
NZ=[0,1600]

fig = plt.figure(figsize=(16,12))
gs=gridspec.GridSpec(3,1,height_ratios=[3,2,2])

a0= plt.subplot(gs[0])
plt_cp(a0,hr_smo,bkps_dhr_smo,t_smo-start_t,NZoom=NZ,ylab='Change Point Detection\nPelt Search Method\n',title="Heart Rate Change (plotted on HR)")


#Sustained attention
a1= plt.subplot(gs[1],sharex=a0)
# plt_sa(a1,t_att,sus_att[:,0],NZoom=NZ)
# plt.setp(a1.get_xticklabels(), visible=False)
# plt_cp_bp(a1,bkps_dhr,t_dhr_nk3-start_t,NLim=[-0.5,1.5])

# Labelled data
a2= plt.subplot(gs[2], sharex=a0)
# plt_labels(a2,t_att,data[:,:6],events,ylab='The Separate Signals',NZoom=NZ)
# plt_cp_bp(a2,bkps_dhr,t_dhr_nk3-start_t,NLim=[-0.5,11.5])



#%% Acceleration Breakpoint analysis
t_jump = 1
t_width = 5
Ntime   = len(acc)/sr
acc_win = np.zeros(int(np.ceil((Ntime-t_width+t_jump)/t_jump)-1))
t_win   = np.zeros(int(np.ceil((Ntime-t_width+t_jump)/t_jump)-1))
for x in np.arange(0,int(np.ceil((Ntime-t_width+t_jump)/t_jump)-1)):
    start_i   = int(x*t_jump*sr)
    end_i     = int(np.min([(x*t_jump+t_width), Ntime])*sr)
    if   not np.any(acc[start_i:end_i]==0): acc_win[x]    = np.std(acc[start_i:end_i])
    elif not np.all(acc[start_i:end_i]==0): acc_win[x]    = np.std(acc[start_i+np.nonzero(acc[start_i:end_i])[0]])
    else:                                   acc_win[x]    = 0
    t_win[x]  = x + t_width/2

plt.figure(figsize=(8,3))
NZ=[0,4]
#NZ = [1320,1350]
ax1 = plt.subplot(111)
viz_ECG_mini(ax1,-acc, [],t_sens,sr=sr, NZoom=NZ, Title = "",col="saddlebrown")
#ax1.set_ylim([-55500,-56100])


# changepoint
bkps_acc = hf_cp(acc_win,cp_type='linear',time=t_win)

#%% Plot Acceleration breakpoints

NZ=[]#[1300,1400]

fig = plt.figure(figsize=(16,12))
gs=gridspec.GridSpec(4,1,height_ratios=[3,3,2,2])

a0= plt.subplot(gs[0])
plt_cp(a0,acc_win,bkps_acc,t_win-start_t,NZoom=NZ,ylab='Change Point Detection\nPelt Search Method\n')
a0.set_title('Acceleration',fontsize=30)

#Sustained attention
a1= plt.subplot(gs[1],sharex=a0)
# plt_sa(a1,t_att,sus_att[:,0],NZoom=NZ)
# plt.setp(a1.get_xticklabels(), visible=False)
# plt_cp_bp(a1,bkps_acc,t_win-start_t,NLim=[-0.5,1.5])

# Labelled data
a2= plt.subplot(gs[2], sharex=a0)
# plt_labels(a2,t_att,data[:,:6],events,ylab='The Separate Signals',NZoom=NZ)
# plt_cp_bp(a2,bkps_acc,t_win-start_t,NLim=[-0.5,11.5])

#%% Plot Acceleration and Heart Rate breakpoints


NZ=[230,380]

fig = plt.figure(figsize=(16,8))
gs=gridspec.GridSpec(6,1,height_ratios=[3,2,0,3,2,0])

## HEART RATE
a0= plt.subplot(gs[0])
plt_cp(a0,hr_emd2_filt,bkps_hr,t_hr_emd2-start_t,NZoom=NZ,ylab='Heart\nRate',Ncolors=2)
#a0.set_title('Heart Rate vs Acceleration comparison',fontsize=30)
#a0.set_ylim([120, 170])

#Sustained attention
a1= plt.subplot(gs[1],sharex=a0)
# plt_sa(a1,t_att,sus_att[:,0],ylab='Sustained\nAttention',NZoom=NZ)
# plt.setp(a1.get_xticklabels(), visible=False)
# plt_cp_bp(a1,bkps_hr,t_hr_nk3-start_t,NLim=[-0.5,1.5],Ncolors=2)

# Labelled data
# a2= plt.subplot(gs[2], sharex=a0)
# plt_labels(a2,t_att,data[:,:6],events,ylab='The Separate Signals',NZoom=NZ)
# plt_cp_bp(a2,bkps_hr,t_hr_nk3-start_t,NLim=[-0.5,11.5],Ncolors=4)

## ACCELERATION
a3= plt.subplot(gs[3],sharex=a0)
plt_cp(a3,acc_win,bkps_acc,t_win-start_t,NZoom=NZ,ylab='Acceleration',Ncolors=2)


#Sustained attention
a4= plt.subplot(gs[4],sharex=a0)
# plt_sa(a4,t_att,sus_att[:,0],ylab='Sustained\nAttention',NZoom=NZ)
# plt.setp(a4.get_xticklabels(), visible=False)
# plt_cp_bp(a4,bkps_acc,t_win-start_t,NLim=[-0.5,1.5],Ncolors=2)
# a4.set_xlabel('')

# Labelled data
a5= plt.subplot(gs[5], sharex=a0)
# plt_labels(a5,t_att,data[:,:6],events,ylab='The Separate Signals',NZoom=NZ)
# plt_cp_bp(a5,bkps_acc,t_win-start_t,NLim=[-0.5,11.5],Ncolors=4)

#%% Mini Save function for heart rate, windowed accel, timepoints, and breakpoints
mlab_bkps_hr = [x+1 for x in bkps_hr if x < len(t_hr_emd2)]
mlab_bkps_hr_smo = [x+1 for x in bkps_hr_smo if x < len(t_smo)]
mlab_bkps_acc = [x+1 for x in bkps_acc if x < len(t_win)]
identifier_string = Ssesh
sio.savemat(identifier_string+"_signals+breakpoint.mat",{'Subject_ID':identifier_string,'start_time':start_t, 'heart_rate':hr_emd2_filt,'heart_rate_time':t_hr_emd2,'heart_rate_changepoints':mlab_bkps_hr, 'heart_rate_smooth':hr_smo,'heart_rate_smooth_time':t_smo,'heart_rate_smooth_changepoints':mlab_bkps_hr_smo, 'body_motion':acc_win,'body_motion_time':t_win,'body_motion_changepoints':mlab_bkps_acc})

#%% Distribution Plotting Function

def plt_distr(data_mat, labels,title='Heart Rate Change',xlab="bpm/second"):    
    Ncols = len(data_mat)
    minmax_test = [np.min([np.min(x) for x in data_mat]),np.max([np.max(x) for x in data_mat])]
    
    fig=plt.figure(figsize=(8+2*len(data_mat),8+2*len(data_mat)-4))
    
    for x in np.arange(Ncols):
        ax = plt.subplot(Ncols,1,x+1)
        n, bins, patches =plt.hist(data_mat[x],bins=100,range=minmax_test,label=labels[x])
        if x==0: ax.set_title(title,fontsize=24)
        if x==Ncols-1: 
            ax.set_xlabel(xlab,fontsize=18),
            plt.setp(ax.get_xticklabels(), visible=True)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
        plt.grid(axis="x")
        plt.plot([np.mean(data_mat[x])-np.std(data_mat[x]),np.mean(data_mat[x])+np.std(data_mat[x])],[0,0],'rd',linewidth=10,markersize=20,label='standard deviation')
        plt.plot([np.mean(data_mat[x]),np.mean(data_mat[x])],[0,np.max(n)],'k--',linewidth=3,label='distribution mean')
        
        plt.legend(fontsize=18)
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.0)
    
#%% Plot Distributions

plt_distr(dhr_att, events)
plt_distr(dhr_grp, ['object','social','looking'])
plt_distr(dhr_act, ['passive','active'])
plt_distr(hr_att, events,title='Heart Rate Distribution',xlab="bpm")
plt_distr(hr_sus, sus_label,title='Distribution of Calculated Heart Rate',xlab="bpm")
plt_distr(dhr_sus, sus_label,title='Distribution of Calculated Heart Rate Change')

#%% Calculate Kurtosis  # Can also use spyst.jarque_bera(dhr_att[x])[1]
# for x in np.arange(0,len(dhr_att)):
#     print("Mean Heart Rate Change of %s has a Kurtosis of %2.2f, with p=%2.2e" % (events[x],spyst.kurtosis(dhr_att[x]),spyst.kurtosistest(dhr_att[x])[1]))

# for x in np.arange(0,len(hr_att)):
#     print("Mean Heart Rate of %s has a Kurtosis of %2.2f, with p=%2.2e" % (events[x],spyst.kurtosis(hr_att[x]),spyst.kurtosistest(hr_att[x])[1]))
    

#%% Plot HR, dHR, acc, and labels quick
NZ=[430,460]

fig = plt.figure(figsize=(16,12))
gs=gridspec.GridSpec(4,1,height_ratios=[1,1,1,1])

a0= plt.subplot(gs[0])
plt_signal(a0,t_hr_nk3-start_t,hr_nk3_spl,ylab='Splined HR\n')

a1= plt.subplot(gs[1], sharex=a0)
a1.plot([0-start_t,max(t_sens)],[0,0],color='grey',marker='.',linewidth=0.5)
plt_signal(a1,t_dhr_nk3-start_t,dhr_nk3_spl,ylab='Heart Rate\nChange')

a2= plt.subplot(gs[2], sharex=a0)
plt_signal(a2,t_sens-start_t,acc_0,ylab='Acceleration\n Data',col='black')

# a3= plt.subplot(gs[3], sharex=a0)
# plt_labels(a3,t_att,data[:,:6],events,ylab='The Separate Signals',NZoom=NZ)

#Sustained attention
a3= plt.subplot(gs[3],sharex=a0)
plt_sa(a3,t_att,sus_att[:,0],NZoom=NZ)


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

#%% Create Blocks

block_p_s = startend_blocks(data[:,0])# passive social
block_a_s = startend_blocks(data[:,1])# active social
block_p_o = startend_blocks(data[:,2])# focused attention
block_a_o = startend_blocks(data[:,3]) # casual attention
block_a_l = startend_blocks(data[:,4])# looking away active
block_p_l = startend_blocks(data[:,5])# looking away passive

#%% Find maximum values within a given block (for priscilla and mari)

b_ind = 4
b_test = startend_blocks(data[:,b_ind])

print("The maximum block length of %s was %2.1f seconds long and occurred at %2.1f seconds (or %2.2f minutes)" %(events[b_ind], np.max(np.diff(b_test,axis=1) )/sr_att,b_test[np.argmax(np.diff(b_test,axis=1))][0]/sr_att,b_test[np.argmax(np.diff(b_test,axis=1))][0]/sr_att/60))


#%% Calculate stats

res_nk3 = calc_all_hrv(r_nk3, ecg, sr=sr, f_freq=False,t_width=5)
sdnn_nk3= res_nk3['HRV_SDNN']
rmssd_nk3 = res_nk3['HRV_RMSSD']
t_hrv = res_nk3['time_5']
rmssd_nk3 = np.asarray(rmssd_nk3)
#%% calculate corrected stats with filter (doesn't make much impact, so ignoring unless I need it)
# t_r_nk3_filt = np.concatenate((t_r_nk3,np.cumsum(60/hr_nk3_filt)+t_r_nk3[0]))
# r_nk3_filt = [int(x*sr) for x in t_r_nk3]
# res_nk3_filt = calc_all_hrv(r_nk3_filt, ecg, sr=sr, f_freq=False)

# sdnn_nk3_filt= res_nk3_filt['HRV_SDNN']
# rmssd_nk3_filt = res_nk3_filt['HRV_RMSSD']
# t_hrv_filt = res_nk3_filt['time_5']


#%% Changepoint Analysis for RMSSD

bkps_rmssd = hf_cp(np.log(rmssd_nk3),cp_type='cptnp',time=t_hrv)


#%% Changepoint and Labels plot for RMSSD
NZ=[0,480]

fig = plt.figure(figsize=(16,12))
gs=gridspec.GridSpec(3,1,height_ratios=[3,2,2])

a0= plt.subplot(gs[0])
plt_cp(a0,np.asarray(np.log(rmssd_nk3)),bkps_rmssd,t_hrv-start_t,NZoom=NZ,ylab='Change Point\nDetection\n',title="ln RMSSD")


#Sustained attention
a1= plt.subplot(gs[1],sharex=a0)
plt_sa(a1,t_att,sus_att[:,0],NZoom=NZ)
plt.setp(a1.get_xticklabels(), visible=False)
plt_cp_bp(a1,bkps_rmssd,t_hrv-start_t,NLim=[-0.5,1.5])

# Labelled data
# a2= plt.subplot(gs[2], sharex=a0)
# plt_labels(a2,t_att,data[:,:6],events,ylab='The Separate Signals',NZoom=NZ)
# plt_cp_bp(a2,bkps_rmssd,t_hrv-start_t,NLim=[-0.5,11.5])

#%% Plot HR, Acc, RMSSD, and labels quick
NZ=[000,1000]


fig = plt.figure(figsize=(16,12))
gs=gridspec.GridSpec(5,1,height_ratios=[1,1,1,1,1])

a0= plt.subplot(gs[0])
plt_cp(a0,hr_nk3_spl,bkps_hr,t_hr_nk3-start_t,NZoom=NZ,ylab='Heart\nRate',Ncolors=2)
#plt_signal(a0,t_hr_nk3-start_t,hr_nk3_spl,ylab='Splined HR\n')
#a0.set_title('t window = 5s',fontsize=30)

a1= plt.subplot(gs[1], sharex=a0)
# plt_signal(a1,t_hrv-start_t,sdnn_nk3,ylab='SDNN\n',col='green')
plt_cp(a1,acc_win,bkps_acc,t_win-start_t,NZoom=NZ,ylab='Local\nAcceleration\nStandard\nDeviation')

a2= plt.subplot(gs[2], sharex=a0)
plt_cp(a2,np.asarray(np.log(rmssd_nk3)),bkps_rmssd,t_hrv-start_t,NZoom=NZ,ylab="ln RMSSD")
#plt_signal(a2,t_hrv-start_t,np.log(rmssd_nk3),ylab='log RMSSD\n',col='green')

a3= plt.subplot(gs[3], sharex=a0)
plt_sa(a3,t_att,sus_att[:,0],NZoom=NZ,ylab='Sustained\nAttention')

#a4= plt.subplot(gs[4], sharex=a0)
# plt_labels(a4,t_att,data[:,:6],events,ylab='The Separate Signals',NZoom=NZ)


#%% regressing out acceleration from heart rate

t_common = t_sens
hr_filt =  hr_nk3_fun(t_common)
acc_filt = interp_vec(acc,t_sens,t_common,S_kind='slinear') #create time-index matched acceleration
df_reg = pd.DataFrame(data={'hr_filt':hr_filt,'acc_filt':acc_filt}) #store both in dataframe
tmp = statsr.lm(ro.Formula('%s ~ %s' % ('hr_filt','acc_filt')),data=robj_from_df(df_reg))

hr_sans_acc = np.array(statsr.resid(tmp))

#%% Plot the regressing out acceleration from heart rate output

NZ=[0,100]

fig = plt.figure(figsize=(16,12))
gs=gridspec.GridSpec(4,1,height_ratios=[1,1,1,1])
a0= plt.subplot(gs[0])
plt.plot(t_common,hr_filt)
viz_ECG_mini(a0,hr_filt, [],t_common,sr=sr, NZoom=NZ, Title ="",ylab= "Heart Rate",col='black')
plt.setp(a0.get_xticklabels(), visible=False)

a1= plt.subplot(gs[1], sharex=a0)
viz_ECG_mini(a1,hr_sans_acc, [],t_common,sr=sr, NZoom=NZ, Title ="",ylab=  "Acceleration\nfiltered\nHeart Rate",col='black')
plt.setp(a1.get_xticklabels(), visible=False)

a2= plt.subplot(gs[2], sharex=a0)
viz_ECG_mini(a2,hr_filt-hr_sans_acc, [],t_common,sr=sr, NZoom=NZ, Title ="",ylab=  "Difference",col='black')
plt.setp(a2.get_xticklabels(), visible=False)

a3 = plt.subplot(gs[3], sharex=a0)
viz_ECG_mini(a3,acc_filt, [],t_common,sr=sr, NZoom=NZ, Title ="",ylab= "Acceleration\nWindowed",col='black')


#%% regressing out heart rate from acceleration
t_common = t_win
acc_filt = acc_win #interp_vec(acc_win,t_win,t_hr_nk3,S_kind='slinear') #create time-index matched acceleration
hr_filt = hr_nk3_fun(t_common) #interp_vec(hr_nk3_filt,t_hr_nk3,t_common,S_kind='slinear') #create time-index
df_reg = pd.DataFrame(data={'hr_filt':hr_filt,'acc_filt':acc_filt}) #store both in dataframe
tmp = statsr.lm(ro.Formula('%s ~ %s' % ('acc_filt','hr_filt')),data=robj_from_df(df_reg))

acc_sans_hr = np.array(statsr.resid(tmp))


#%% Plot the acceleration output

NZ=[0,100]

fig = plt.figure(figsize=(16,12))
gs=gridspec.GridSpec(4,1,height_ratios=[1,1,1,1])
a0= plt.subplot(gs[0])
#plt.plot(t_hr_nk3,hr_nk3_filt)
viz_ECG_mini(a0,acc_filt, [],t_common,sr=sr, NZoom=NZ, Title ="",ylab= "Acceleration\nWindowed",col='black')
plt.setp(a0.get_xticklabels(), visible=False)

a1= plt.subplot(gs[1], sharex=a0)
viz_ECG_mini(a1,acc_sans_hr, [],t_common,sr=sr, NZoom=NZ, Title ="",ylab=  "Acceleration\nfiltered\nHeart Rate",col='black')
plt.setp(a1.get_xticklabels(), visible=False)

a2= plt.subplot(gs[2], sharex=a0)
viz_ECG_mini(a2,acc_filt-acc_sans_hr, [],t_common,sr=sr, NZoom=NZ, Title ="",ylab=  "Difference",col='black')
plt.setp(a2.get_xticklabels(), visible=False)

a3 = plt.subplot(gs[3], sharex=a0)
viz_ECG_mini(a3,hr_filt, [],t_common,sr=sr, NZoom=NZ, Title ="",ylab= "Heart\nRate",col='black')

#%% Changepoint Analysis for Acceleration with HR regressed out

bkps_accshr = hf_cp(acc_sans_hr,cp_type='cptnp',time=t_common)


#%% Changepoint and Labels plot for acceleration
NZ=[0,480]

fig = plt.figure(figsize=(16,12))
gs=gridspec.GridSpec(5,1,height_ratios=[3,2,0,3,2])

a0= plt.subplot(gs[0])
plt_cp(a0,np.asarray(acc_sans_hr),bkps_accshr,t_common-start_t,NZoom=NZ,ylab='Acceleration\nwith HR\nregressed out\n',title="Acceleration Changepoint comparison")


#Sustained attention
a1= plt.subplot(gs[1],sharex=a0)
plt_sa(a1,t_att,sus_att[:,0],NZoom=NZ)
plt.setp(a1.get_xticklabels(), visible=False)
plt_cp_bp(a1,bkps_accshr,t_common-start_t,NLim=[-0.5,1.5])


# Labelled data
# a2= plt.subplot(gs[2], sharex=a0)
# plt_labels(a2,t_att,data[:,:6],events,ylab='The Separate Signals',NZoom=NZ)
# plt_cp_bp(a2,bkps_rmssd,t_hrv-start_t,NLim=[-0.5,11.5])

a3= plt.subplot(gs[3],sharex=a0)
plt_cp(a3,np.asarray(acc_win),bkps_acc,t_win-start_t,NZoom=NZ,ylab='Windowed\nAcceleration\nChangepoint',title="")


#Sustained attention
a4= plt.subplot(gs[4],sharex=a0)
plt_sa(a4,t_att,sus_att[:,0],NZoom=NZ)
plt.setp(a4.get_xticklabels(), visible=False)
plt_cp_bp(a4,bkps_acc,t_win-start_t,NLim=[-0.5,1.5])

#%% PCA investigation

pca = PCA(n_components=2)
tmpX = np.column_stack((acc_filt.reshape(-1, 1), hr_filt.reshape(-1, 1)))
tmp=pca.fit(tmpX)
print('Component contribution breakdown to a combined PCA:')
print(tmp.components_)

#%% EMD investigation

# create signals
t_common = t_win
hr_filt =  hr_nk3_fun(t_common)
acc_filt = interp_vec(acc,t_sens,t_common,S_kind='slinear') #create time-index matched acceleration

# create imfs
emd = EMD()
imfs = emd.emd(acc_filt)

pca = PCA(n_components=min(imfs.shape)+1)
tmp = pca.fit(np.column_stack((hr_filt.reshape(-1, 1), np.transpose(imfs))))
print('Component contribution breakdown to a combined PCA:')
print(tmp.components_)

plt.plot(t_common,np.transpose(imfs))


#%% Regression of imfs of acceleration from heart rate

# Sort strings
Sformula = 'hr_filt ~ '   
Scolumns = ['hr_filt']
for x in np.arange(0,len(imfs)):
    Scolumns.append('imf_'+str(x))
    Sformula += ' +'+str(Scolumns[x+1])

# Create Dataframe    
df_reg = pd.DataFrame(np.column_stack((hr_filt.reshape(-1, 1), np.transpose(imfs))),columns=Scolumns)    

# Do regression
tmp = statsr.lm(ro.Formula('%s' % Sformula),data=robj_from_df(df_reg))
hr_sans_acc = np.array(statsr.resid(tmp))

#%% Changepoint

bkps_hrsacc = hf_cp(hr_sans_acc,cp_type='linear',time=t_common)
# bkps_hrsacc = hf_cp(hr_sans_acc,cp_type='cptnp',time=t_common)

#%%  IMF visualisation plot

NZ=[0,500]

fig = plt.figure(figsize=(12,6))
gs=gridspec.GridSpec(5,3,height_ratios=[3,1,2,2,2],width_ratios = [2,2,2])

a0= plt.subplot(gs[0:3])
viz_ECG_mini(a0,acc_filt, [],t_common,sr=sr, NZoom=NZ, Title ="",ylab= "Windowed\nAcceleration",col='black')
plt.setp(a0.get_yticklabels(), visible=True)

a1= plt.subplot(gs[3:5])
plt.setp(a1.get_xticklabels(), visible=False)
plt.setp(a1.get_yticklabels(), visible=False)
a1.axis('off')

a2 = plt.subplot(gs[6])
viz_ECG_mini(a2,imfs[0] ,[],t_common,sr=sr, NZoom=NZ, Title ="",ylab= "IMF 1",col='black')
plt.setp(a2.get_yticklabels(), visible=False)
plt.setp(a2.get_xticklabels(), visible=False)

a3 = plt.subplot(gs[7])
viz_ECG_mini(a3,imfs[1] ,[],t_common,sr=sr, NZoom=NZ, Title ="",ylab= "IMF 2",col='black')
plt.setp(a3.get_yticklabels(), visible=False)
plt.setp(a3.get_xticklabels(), visible=False)

a4 = plt.subplot(gs[8])
viz_ECG_mini(a4,imfs[2] ,[],t_common,sr=sr, NZoom=NZ, Title ="",ylab= "IMF 3",col='black')
plt.setp(a4.get_yticklabels(), visible=False)
plt.setp(a4.get_xticklabels(), visible=False)

a5 = plt.subplot(gs[9],sharex=a2)
viz_ECG_mini(a5,imfs[3] ,[],t_common,sr=sr, NZoom=NZ, Title ="",ylab= "IMF 4",col='black')
plt.setp(a5.get_yticklabels(), visible=False)
plt.setp(a5.get_xticklabels(), visible=False)

a6 = plt.subplot(gs[10],sharex=a3)
viz_ECG_mini(a6,imfs[4] ,[],t_common,sr=sr, NZoom=NZ, Title ="",ylab= "IMF 5",col='black')
plt.setp(a6.get_yticklabels(), visible=False)
plt.setp(a6.get_xticklabels(), visible=False)

a7 = plt.subplot(gs[11],sharex=a4)
viz_ECG_mini(a7,imfs[5] ,[],t_common,sr=sr, NZoom=NZ, Title ="",ylab= "IMF 6",col='black')
plt.setp(a7.get_yticklabels(), visible=False)
plt.setp(a7.get_xticklabels(), visible=False)

a8 = plt.subplot(gs[12],sharex=a2)
viz_ECG_mini(a8,imfs[6] ,[],t_common,sr=sr, NZoom=NZ, Title ="",ylab= "IMF 7",col='black')
plt.setp(a8.get_yticklabels(), visible=False)

a9 = plt.subplot(gs[13],sharex=a3)
viz_ECG_mini(a9,imfs[7] ,[],t_common,sr=sr, NZoom=NZ, Title ="",ylab= "IMF 8",col='black')
plt.setp(a9.get_yticklabels(), visible=False)

a10 = plt.subplot(gs[14],sharex=a4)
viz_ECG_mini(a10,imfs[8] ,[],t_common,sr=sr, NZoom=NZ, Title ="",ylab= "IMF 9",col='black')
plt.setp(a10.get_yticklabels(), visible=False)

fig.tight_layout()
fig.subplots_adjust(hspace=0.0)


#%%  Changepoint and Labels plot for imf-regression of acceleration from heart rate
NZ=[370,380]

fig = plt.figure(figsize=(18,8))
gs=gridspec.GridSpec(5,1,height_ratios=[3,2,0, 3,2])

a0= plt.subplot(gs[0])
plt_cp(a0,np.asarray(hr_sans_acc),bkps_hrsacc,t_common-start_t,NZoom=NZ,ylab='Post-Regression\nHeart\nRate',title="Heart Rate Changepoint comparison")

#Sustained attention
a1= plt.subplot(gs[1],sharex=a0)
plt_sa(a1,t_att,sus_att[:,0],NZoom=NZ,ylab="Sustained\nAttention")
plt.setp(a1.get_xticklabels(), visible=False)
plt_cp_bp(a1,bkps_hrsacc,t_common-start_t,NLim=[-0.5,1.5])

# Labelled data
# a2= plt.subplot(gs[2], sharex=a0)
# plt_labels(a2,t_att,data[:,:6],events,ylab='The Separate Signals',NZoom=NZ)
# plt_cp_bp(a2,bkps_rmssd,t_hrv-start_t,NLim=[-0.5,11.5])

a3= plt.subplot(gs[3],sharex=a0)
plt_cp(a3,hr_nk3_spl,bkps_hr,t_hr_nk3-start_t,NZoom=NZ,ylab='Heart\nRate',title="")

#Sustained attention
a4= plt.subplot(gs[4],sharex=a0)
plt_sa(a4,t_att,sus_att[:,0],NZoom=NZ,ylab="Sustained\nAttention")
plt.setp(a4.get_xticklabels(), visible=True)
plt_cp_bp(a4,bkps_hr,t_hr_nk3-start_t,NLim=[-0.5,1.5])


#%% Visualise affect of imf regression
NZ=[0,500]

fig = plt.figure(figsize=(12,8))
gs=gridspec.GridSpec(5,1,height_ratios=[5,5,5,2,5])
a0= plt.subplot(gs[0])
#plt.plot(t_hr_nk3,hr_nk3_filt)
viz_ECG_mini(a0,hr_filt, [],t_common,sr=sr, NZoom=NZ, Title ="",ylab= "Heart\nRate",col='black')
plt.setp(a0.get_xticklabels(), visible=False)

a1= plt.subplot(gs[1], sharex=a0)
viz_ECG_mini(a1,hr_sans_acc, [],t_common,sr=sr, NZoom=NZ, Title ="",ylab=  "Post-Regression\nHeart\nRate",col='black')
plt.setp(a1.get_xticklabels(), visible=False)

a2= plt.subplot(gs[2], sharex=a0)
viz_ECG_mini(a2,hr_filt-hr_sans_acc, [],t_common,sr=sr, NZoom=NZ, Title ="",ylab=  "Difference",col='black')
plt.setp(a2.get_xticklabels(), visible=True)

a4 =plt.subplot(gs[3], sharex=a0)
plt.setp(a4.get_xticklabels(), visible=False)
plt.setp(a4.get_yticklabels(), visible=False)
a4.axis('off')

a3 = plt.subplot(gs[4], sharex=a0)
viz_ECG_mini(a3,acc_filt, [],t_common,sr=sr, NZoom=NZ, Title ="",ylab= "Windowed\nAcceleration",col='black')
fig.tight_layout()
fig.subplots_adjust(hspace=0.0)

#%% CHangepoint on the difference measure (the part of acceleration which does feed into the HR)

bkps_hr_res = hf_cp(hr_filt-hr_sans_acc,cp_type='cptnp',time=t_common)
# bkps_hrsacc = hf_cp(hr_sans_acc,cp_type='cptnp',time=t_common)

#%%  Changepoint and Labels plot for imf-regression of acceleration from heart rate
NZ=[1000,1480]

fig = plt.figure(figsize=(16,12))
gs=gridspec.GridSpec(5,1,height_ratios=[3,2,0, 3,2])

a0= plt.subplot(gs[0])
plt_cp(a0,np.asarray(hr_filt-hr_sans_acc),bkps_hr_res,t_common-start_t,NZoom=NZ,ylab='Heart Rate -\nHeart Rate\nwith Acceleration\nregressed out\nUsing IMFs',title="Heart Rate Changepoint comparison")

#Sustained attention
a1= plt.subplot(gs[1],sharex=a0)
plt_sa(a1,t_att,sus_att[:,0],NZoom=NZ)
plt.setp(a1.get_xticklabels(), visible=False)
plt_cp_bp(a1,bkps_hr_res,t_common-start_t,NLim=[-0.5,1.5])

# Labelled data
# a2= plt.subplot(gs[2], sharex=a0)
# plt_labels(a2,t_att,data[:,:6],events,ylab='The Separate Signals',NZoom=NZ)
# plt_cp_bp(a2,bkps_rmssd,t_hrv-start_t,NLim=[-0.5,11.5])

a3= plt.subplot(gs[3],sharex=a0)
plt_cp(a3,acc_filt,bkps_acc,t_common-start_t,NZoom=NZ,ylab='Windowed\n Acceleration',title="")

#Sustained attention
a4= plt.subplot(gs[4],sharex=a0)
plt_sa(a4,t_att,sus_att[:,0],NZoom=NZ)
plt.setp(a4.get_xticklabels(), visible=False)
plt_cp_bp(a4,bkps_acc,t_common-start_t,NLim=[-0.5,1.5])

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
plt_cp(a0,hr_nk3_filt,bkps_hr,t_hr_nk3-start_t,NZoom=NZ,ylab='Post-Regression\nHeart\nRate',title="Heart Rate Changepoint")

for x in np.arange(0,len(hr_bkps_ind)-1):
    plt.plot(t_hr_nk3[hr_bkps_ind[x]:hr_bkps_ind[x+1]]-start_t,hr_bkps_slope[x]*t_hr_nk3[hr_bkps_ind[x]:hr_bkps_ind[x+1]]+hr_bkps_icept[x],color='black',linewidth=4)

#Sustained attention
a1= plt.subplot(gs[1],sharex=a0)
plt_sa(a1,t_att,sus_att[:,0],NZoom=NZ,ylab="Sustained\nAttention")
plt.setp(a1.get_xticklabels(), visible=True)
plt_cp_bp(a1,bkps_hr,t_hr_nk3-start_t,NLim=[-0.5,1.5])

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


#%% Regress acceleration with ECG
# # This may be an issue due to time scales. Let's try.

# # create signals
# t_common = t_sens # this just is the common timescale
# ecg_filt =  interp_vec(ecg,t_sens,t_common,S_kind='slinear')
# acc_filt = interp_vec(acc,t_sens,t_common,S_kind='slinear') #create time-index matched acceleration

# # create imfs
# emd = EMD()
# imfs = emd.emd(acc_filt)
# plt.plot(t_common,np.transpose(imfs))
# pca = PCA(n_components=min(imfs.shape)+1)
# tmp = pca.fit(np.column_stack((hr_filt.reshape(-1, 1), np.transpose(imfs))))
# print('Component contribution breakdown to a combined PCA:')
# print(tmp.components_)


# #%% Regression of imfs of acceleration from ecg

# # Sort strings
# Sformula = 'ecg_filt ~ '   
# Scolumns = ['ecg_filt']
# for x in np.arange(0,len(imfs)):
#     Scolumns.append('imf_'+str(x))
#     Sformula += ' +'+str(Scolumns[x+1])

# # Create Dataframe    
# df_reg = pd.DataFrame(np.column_stack((ecg_filt.reshape(-1, 1), np.transpose(imfs))),columns=Scolumns)    

# # Do regression
# tmp = statsr.lm(ro.Formula('%s' % Sformula),data=robj_from_df(df_reg))
# ecg_sans_acc = np.array(statsr.resid(tmp))

# #%% Visualise

# plt.figure(figsize=(16,10))
    
# NZ = [500,550]
# ax1 = plt.subplot(311)
# viz_ECG_mini(ax1,ecg, [],t_sens,sr=sr, NZoom=NZ, Title="",ylab = "Original ECG")
# #ax1.set_ylim([12000,55000])
# ax1 = plt.subplot(312)
# viz_ECG_mini(ax1,ecg_sans_acc, [],t_sens,sr=sr, NZoom=NZ, Title="",ylab = "ECG with Acceleration\nRegressed Out")
# #ax1.set_ylim([12000,55000])

# ax1 = plt.subplot(313)
# viz_ECG_mini(ax1,ecg-ecg_sans_acc, [],t_sens,sr=sr, NZoom=NZ, Title="",ylab = "ECG - Regessed ECG")
# #ax1.set_ylim([12000,55000])

# #%% Changepoint

# bkps_hrsacc = hf_cp(hr_sans_acc,cp_type='linear',time=t_common)
# # bkps_hrsacc = hf_cp(hr_sans_acc,cp_type='cptnp',time=t_common)

# #%%  Changepoint and Labels plot for imf-regression of acceleration from heart rate
# NZ=[1000,1480]

# fig = plt.figure(figsize=(16,12))
# gs=gridspec.GridSpec(5,1,height_ratios=[3,2,0,3,2])

# a0= plt.subplot(gs[0])
# plt_cp(a0,np.asarray(hr_sans_acc),bkps_hrsacc,t_common-start_t,NZoom=NZ,ylab='Heart Rate\nwith Acceleration\nregressed out\nUsing IMFs',title="Heart Rate Changepoint comparison")


# #Sustained attention
# a1= plt.subplot(gs[1],sharex=a0)
# plt_sa(a1,t_att,sus_att[:,0],NZoom=NZ)
# plt.setp(a1.get_xticklabels(), visible=False)
# plt_cp_bp(a1,bkps_hrsacc,t_common-start_t,NLim=[-0.5,1.5])


# # Labelled data
# # a2= plt.subplot(gs[2], sharex=a0)
# # plt_labels(a2,t_att,data[:,:6],events,ylab='The Separate Signals',NZoom=NZ)
# # plt_cp_bp(a2,bkps_rmssd,t_hrv-start_t,NLim=[-0.5,11.5])

# a3= plt.subplot(gs[3],sharex=a0)
# plt_cp(a3,hr_nk3_spl,bkps_hr,t_hr_nk3-start_t,NZoom=NZ,ylab='Heart Rate\nDirect\nChangepoint',title="")


# #Sustained attention
# a4= plt.subplot(gs[4],sharex=a0)
# plt_sa(a4,t_att,sus_att[:,0],NZoom=NZ)
# plt.setp(a4.get_xticklabels(), visible=False)
# plt_cp_bp(a4,bkps_hr,t_hr_nk3-start_t,NLim=[-0.5,1.5])

#%% Create CSV variables

# Heart Rate
hr_csv = hr_nk3_fun(t_sens)
hr_filt_csv = interp_vec(hr_nk3_filt,t_hr_nk3-start_t,t_sens-start_t)
dhr_csv= np.insert(np.diff(hr_csv),0,0)

# Labels
p_s_csv = interp_vec(data[:,0],t_att,t_sens-start_t)
a_s_csv = interp_vec(data[:,1],t_att,t_sens-start_t)
p_o_csv = interp_vec(data[:,2],t_att,t_sens-start_t)
a_o_csv = interp_vec(data[:,3],t_att,t_sens-start_t)
a_l_csv = interp_vec(data[:,4],t_att,t_sens-start_t)
p_l_csv = interp_vec(data[:,5],t_att,t_sens-start_t)

t_ind=np.where((t_sens-start_t)>=0) #from start of labels
t_ind = t_ind[0][0]                    #the index of first non-negative value

#ensure none of the labels inadvertently interpolate before start of labelling
p_s_csv[:t_ind]=0
a_s_csv[:t_ind]=0
p_o_csv[:t_ind]=0
a_o_csv[:t_ind]=0
a_l_csv[:t_ind]=0
p_l_csv[:t_ind]=0

acc_win_csv = interp_vec(acc_win,t_win-start_t,t_sens-start_t,S_kind='linear',S_fill='extrapolate')

# hrv labels
sdnn_csv= interp_vec(sdnn_nk3,t_hrv,t_sens)
rmssd_csv = interp_vec(rmssd_nk3,t_hrv,t_sens)



# Create Lists for csv
header_csv = ['Time (s)', 'ECG','Heart Rate','Pre-Spline Heart Rate','Heart Rate Difference', 'SDNN','RMSSD','Acceleration','Acceleration de-baselined','Acceleration Local Stdev','Passive_Social','Active_Social','Passive_Object','Active_Object','Active_Looking_Away','Passive_looking_Away']
data_csv =list(zip(t_sens-start_t,ecg,hr_csv,hr_filt_csv,dhr_csv,sdnn_csv,rmssd_csv,acc,acc_0,acc_win_csv,p_s_csv,a_s_csv,p_o_csv,a_o_csv,a_l_csv,p_l_csv))

data_csv = data_csv[t_ind:] #only values from start of labelling (so as not to falsely equate things with zeros just because that part wasn't labelled)


# Save String
svStr = filename_mat.split("\\", -1)
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
# data_HR =list(zip(t_hr_nk3-start_t,hr_nk3_filt))
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