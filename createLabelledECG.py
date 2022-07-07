# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 14:16:40 2021

Load in an unlabelled ECG signal, and create a set of labelled peaks.

STEPS:
    1. create filtered version of the signal
    2. Create initial peak list from labelled signal
    3. Iterate through signal, adding/removing peak labels as necessary
        Shift any peak labels which are attached to the wrong local peak
    4. Automate a local maximum algorithm w.r.t. original unfiltered ECG signal
    5. Re-iterate through signal as final sanity check

@author: Dr. Harry T. Mason
"""

#%% Load libraries
from   datetime import datetime
from   ecgdetectors import Detectors
import fnmatch
import heartpy as hp
import hrv
import math
import matplotlib.pyplot as plt
import matplotlib.cm     as mplcm
import matplotlib.colors as colors
import neurokit2 as nk 
import numpy as np
import os
import pandas as pd
from   PyEMD import EMD
import re
import pylab
import pywt
from   scipy.ndimage import median_filter
from   scipy.signal  import welch, butter, filtfilt, sosfiltfilt
from   scipy.stats   import median_abs_deviation
import seaborn as sns
from   sklearn.metrics         import r2_score
from   sklearn.model_selection import train_test_split
from   sklearn.preprocessing   import StandardScaler
from   sklearn.linear_model    import LinearRegression
from   sklearn.pipeline        import Pipeline
pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('lr', LinearRegression())])
from   skimage.restoration     import (denoise_wavelet, estimate_sigma)
import sys

sys.path.append("G:\My Drive\BackUp\Documents\Python\Research")
from   HarryFunctions import (hf_find,cmvir,hf_cp,hf_ecg,clean_peaks,fill_peaks,remove_peaks,find_local_peaks,calc_all_hrv,viz_all_HRV,hrSQI,cp_plot,hf_goodness, interp_vec)


#%% Initialise Strings
Ssesh = "10M"#"8M"#"15M"#"10M"#
Sp    = "P2"#"P3"#"P1"#"P1"#
Sb    = "B2"#"B1"#"B2"#"B1"#
Speak = "_peaks_2.txt"


#%% Load ECG (BF Study)

paths = hf_find('*.txt', ("G:\\My Drive\\BackUp\\Documents\\Data\\BF_sensorData\\BF_%s_%s" % (Ssesh,Sp)));
paths.sort()
print(paths)

Npath = 0;
filename = paths[Npath]
#18: "C:/Users/htm516/Documents/Data/BF_sensorData/BF_8m_P1/opensignals_000780D8AB6E_000780F9DDEE_2018-10-30_12-13-06.txt"

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
    dict1       = tmpDict[list(tmpDict.keys())[0]] #dictionary with header info
    Ncol1       = len(dict1['column'])             # add the columns which exist
    df1         = df.loc[:,0:Ncol1-1]                      #pandas file with everything
    sr1         = dict1['sampling rate']
    cols1 = cols[0:2]+[cols[x+1] for x in dict1['channels']]
    df1.set_axis(cols1, axis=1, inplace=True) #rename columns
    #for measure in dict1.keys():
    #    print('%s: %s' %(measure, dict1[measure]))
    begin1      =     dict1['time']
    
    # # kid B
    dict2       = tmpDict[list(tmpDict.keys())[1]] #dictionary with header info
    Ncol2       = len(dict2['column'])             # add the columns which exist
    df2         = df.loc[:,Ncol1:(Ncol2+Ncol1-1)]
    sr2         = dict2['sampling rate']
    cols2 = cols[0:2]+[cols[x+1] for x in dict2['channels']]
    df2.set_axis(cols2, axis=1, inplace=True) #rename columns
    begin1      =     dict2['time']
    
    if Sb[-1]=="1":
        df      = df1
        sr      = sr1
        time_stamps.append(datetime.strptime(dict1['time'],'%H:%M:%S.%f'))
    elif Sb[-1]=="2":
        df      = df2
        sr      = sr2
        time_stamps.append(datetime.strptime(dict2['time'],'%H:%M:%S.%f'))
    else:
        print("Warning: Baby Option not properly selected")
    
    # append zeros to fill gaps
    start_i = int((time_stamps[-1] - time_stamps[0]).total_seconds()*sr) # if ecg recording had continued, this would be the index the recording started at
    
    lum     = pd.concat([lum,pd.Series(np.zeros(start_i-len(lum)))], ignore_index=True)
    lum     = pd.concat([lum,df['CH5_LGHT']], ignore_index=True) # luminence data
    ecg     = pd.concat([ecg,pd.Series(np.zeros(start_i-len(ecg)))], ignore_index=True)
    ecg     = pd.concat([ecg,df['CH1']], ignore_index=True)      # ecg
    
    accX    = df['CH7_X'].to_numpy() 
    accY    = df['CH6_Y'].to_numpy()
    accZ    = df['CH8_Z'].to_numpy()
    acc     = np.append(acc,np.sqrt(accX**2+accY**2+accZ**2))
    acc_hpy = np.append(acc_hpy,hp.remove_baseline_wander(np.sqrt(accX**2+accY**2+accZ**2), sr)) # remove baseline
    acc_0   = np.append(acc_0,hp.remove_baseline_wander(np.sqrt(accX**2+accY**2+accZ**2), sr) - np.mean(hp.remove_baseline_wander(np.sqrt(accX**2+accY**2+accZ**2), sr)))  # demean


t_sens = np.arange(0,len(ecg))/sr
    
#print signal info (length, etc.)
print("\n\nFile: %s\nBaby: %s\n%d samples.\n%2.1f minutes long.\n\n" %(filename,Sb[-1],len(ecg),len(ecg)/sr/60 ))



#%% Load data (V1 sensor)

paths = hf_find('[!~]*.xlsx', "G:\\My Drive\\BackUp\\Documents\\Data\\NewECG\\") #finds xlsx files that aren't being worked on (ones being work on have ~$ in front)

print(paths)

Npath = 0;
filename = paths[Npath]

df = pd.read_excel(filename)

ecg      = np.asarray(df['ECG'])
t_sens   = np.asarray(df['UNIX Timestamp'])
t_sens   = (t_sens-t_sens[0])/1000
sr       = 1/(np.max(t_sens)/len(ecg))
Speak    = "_peaks.txt"

ecg_90 = np.percentile(abs(ecg),90)
thresh = ecg_90*100

ecg = np.asarray([y if abs(y) <= thresh else np.median(ecg[ind-4:ind+5]) for ind,y in enumerate(ecg)])

ecg = -ecg

filt     = ecg


#%% Load data (Simulation)

Ssesh = "ECGDATA_30-05-2022"

paths = hf_find(('%s.TXT' % Ssesh), ("G:\\Shared drives\\FIRST Leap\\Data\\Simulations\\ECG noise simulations data\\SimulatorTest1\\" ));
paths.sort()
print(paths)

Npath = 0;
filename = paths[Npath]

df_ecg = pd.read_csv(filename,delimiter=' ',engine='python',header=None,names=['tmp','t_ecg','ecg'])
df_ecg.head() 
df_ecg=df_ecg.sort_values(by=['t_ecg'],ascending=True)

ecg = np.asarray(df_ecg['ecg'])
t_sens = np.asarray(df_ecg['t_ecg'])

t_end_ecg   = df_ecg['t_ecg'][df_ecg.index[-1]]
t_start_ecg = df_ecg['t_ecg'][df_ecg.index[0]]
t_gap       = (t_end_ecg-t_start_ecg)/(len(ecg)-1)

#t_ecg = np.arange(t_start_ecg, t_end_ecg+t_gap,t_gap)


sr = 1/np.mean(np.diff(t_sens))

Speak    = "_peaks.txt"
#Speak = '_peaks_including_gaps.txt'

#%% Load Data (eye tracker study)
Ssesh = "B064"

paths = hf_find('open*.txt', ("G:\\My Drive\\BackUp\\Documents\\Data\\Study2\\EyeTracker + ECG study\\%s\\" % (Ssesh)));
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
    
    dict1      = tmpDict[list(tmpDict.keys())[0]] #dictionary with header info
    Ncol1       = len(dict1['column'])             # add the columns which exist
    df         = df.loc[:,0:Ncol1-1]                      #pandas file with everything
    sr         = dict1['sampling rate']
    cols1 = cols[0:2]+[cols[x+1] for x in dict1['channels']]
    df.set_axis(cols1, axis=1, inplace=True) #rename columns
    begin1      =     dict1['time']
    time_stamps.append(datetime.strptime(dict1['time'],'%H:%M:%S.%f'))

    # append zeros to fill gaps
    start_i = int((time_stamps[-1] - time_stamps[0]).total_seconds()*sr) # if ecg recording had continued, this would be the index the recording started at
    ecg     = pd.concat([ecg,pd.Series(np.zeros(start_i-len(ecg)))], ignore_index=True)
    ecg     = pd.concat([ecg,df['CH1']], ignore_index=True)      # ecg    
    
ecg=-ecg
t_sens = np.arange(0,len(ecg))/sr
Speak    = "_peaks.txt"
#%% print ecg signal info (length, etc.)

print("\n\nFile: %s.\n%d ECG samples.\n%2.1f minutes long.\n\n" %(filename,len(ecg),(max(t_sens))/60 ))

#%% Create Peaklist(current approach)
filt1 = hf_ecg(ecg,sr=sr,freqfilt=[0.5,20])   
Peaks = nk.ecg_findpeaks(filt1, sampling_rate=sr, method="neurokit")
Peaks = np.asarray(Peaks['ECG_R_Peaks'])
#%% Create initial peaklist (heartpy approach)

# filt2=ecg

# # Enhance peaks
# #filt2 = hp.enhance_ecg_peaks(hp.scale_data(filt2), sr1, 
# #                                aggregation='median', iterations=5)

# # baseline wander
# filt2 = hp.remove_baseline_wander(filt2, sr)

# # # Notch
# filt2 = hp.filter_signal(filt2,                           # data
#                           cutoff = 0.05,                  #where to apply notch
#                           sample_rate = sr,              # sample rate
#                           filtertype='notch')             # filter type

# # #BandPass
# filt2 = hp.filter_signal(filt2,                           # data
#                           cutoff = [0.003,20],                  #where to apply band limits
#                           sample_rate = sr,              # sample rate
#                           filtertype='bandpass')             # filter type

# ECG1_pp, meas1_pp = hp.process(filt2, sr, bpmmax=220) #calc_freq=True, clean_rr=True, clean_rr_method='quotient-filter'

# #Peaks = [x for x in ECG1_pp['peaklist'] if x not in ECG1_pp['removed_beats']]
# #print('Currently, %d accepted peaks' %( len(Peaks)))


# #%% second peak list (Martinez approach)

# r_mar = nk.ecg_findpeaks(filt2, sampling_rate=sr, method="martinez2003")
# r_mar = r_mar['ECG_R_Peaks']
# Peaks=r_mar


#%% Third peak list (neurokit2 + peak cleaning approach)
filt1=ecg
filt1 = nk.ecg_clean(filt1, sampling_rate=sr, method="neurokit") # pre-processed signal

# sos = butter(5, [0.5,15], btype='bandpass', output="sos", fs=sr)
# filt1 = sosfiltfilt(sos, filt1)

# b = np.ones(int(sr / 50)) # filter with moving average kernel width of 50Hz
# a = [len(b)]
# ecg_hpy = filtfilt(b, a, filt1, method="pad")
#peak calculation

r_nk2 = nk.ecg_findpeaks(filt1, sampling_rate=sr, method="neurokit")
r_nk2 = np.asarray(r_nk2['ECG_R_Peaks'])
t_r_nk2 = t_sens[r_nk2]

r_nk3  = clean_peaks(r_nk2,ecg)
t_r_nk3 = t_sens[r_nk3]

Peaks = r_nk3

#%% Fourth Peak List (Band Pass + EMD)
emd = EMD()
ecg_emd = hf_ecg(ecg,sr=sr,freqfilt=[0.5,20])    
ecg_imfs = emd.emd(np.array(ecg_emd))
ecg_imfs_pow = [nk.signal_psd(x,method="welch",min_frequency=0, max_frequency=160) for x in ecg_imfs]

ind_low = np.where(ecg_imfs_pow[0]['Frequency']>0.5)[0][0] # -ind_05 for the negative index
ind_high = np.where(ecg_imfs_pow[0]['Frequency']>8)[0][0] # -ind_05 for the negative index
P_05 = np.zeros(len(ecg_imfs))
for ind, x in enumerate(ecg_imfs_pow):
    P_05[ind] = x['Power'][ind_low:ind_high].sum()/x['Power'].sum()
    
 
ecg_emd = np.sum(ecg_imfs[np.where(P_05<0.5)[0]],axis=0) # +np.mean(ecg)

r_emd = nk.ecg_findpeaks(ecg_emd, sampling_rate=sr, method="neurokit")
r_emd = np.asarray(r_emd['ECG_R_Peaks'])
Peaks  = clean_peaks(r_emd,ecg)
filt1=ecg_emd

#%% Load Existing Peaklist
ldStr = paths[Npath].split("\\", -1)
ldStr = ldStr[-1].split(".", 1)
ldStr = "G:\\My Drive\\BackUp\\Documents\\Data\\ProcessedData\\" + ldStr[0] + Speak
Peaks=np.loadtxt(ldStr,dtype=int)
filt1=ecg
#%% Plotting function
def viz_ECG(ecg_old, ecg_new, times, peaks_new, sr=500, NZoom=[]):
    if not NZoom:
        NZoom = [0,len(ecg_old)/sr] 
    
    plt.figure(figsize=(12,10))
    
    a0  = plt.subplot(311)
    plt.plot(times,ecg_old, label = 'ECG signal', color='plum',linewidth=1)
    plt.scatter(times[peaks_new],[ecg_old[x] for x in peaks_new], color='green', label = 'identified peaks',linewidth=0.5)
    #plt.legend(loc='lower right')
    a0.set_xlim(NZoom)
    idx = np.searchsorted(times,NZoom, side="left")
    a0.set_ylim([min(ecg_old[idx[0]:idx[1]-1])-np.median(np.abs(ecg_old))*0.1,max(ecg_old[idx[0]:idx[1]-1])*1.1]+np.median(np.abs(ecg_old))*0.1)
    #a0.set_xlabel('s')
    plt.grid(axis = 'x')
    a0.set_title("Original Signal")
    
    a1 = plt.subplot(312)
    plt.plot(times,ecg_new, label = 'ECG signal', color='plum',linewidth=1)
    plt.scatter(times[peaks_new],[ecg_new[x] for x in peaks_new], color='green', label = 'identified peaks',linewidth=0.5)
    #plt.legend(loc='lower right')
    a1.set_xlim(NZoom)
    #ax1.set_xlabel('s')
    plt.grid(axis = 'x')
    a1.set_title("Filtered Signal")
    idx = np.searchsorted(times,NZoom, side="left")
    a1.set_ylim([min(ecg_new[idx[0]:idx[1]-1])-np.median(np.abs(ecg_new))*0.1,max(ecg_new[idx[0]:idx[1]-1])*1.1]+np.median(np.abs(ecg_new))*0.1)
    
    
    
    a2 = plt.subplot(313)
    plt.scatter(times[peaks_new],np.arange(len(peaks_new)), color='green')
    a2.set_xlim(NZoom)
    ind = [ind for ind,x in enumerate(peaks_new) if x>NZoom[0]*sr and x<NZoom[1]*sr]
    #print(ind,NZoom)
    a2.set_ylim([ind[0],ind[-1]+1])
    a2.set_xlabel('s')
    a2.set_title("Index of peaks")
    plt.grid()
    plt.show()
    
#%% Visualise
tmp = [1250,1260]
viz_ECG(ecg,filt1, t_sens,Peaks, sr=sr,NZoom=tmp)

#%% Buffer
#%% 
#%% PROCESSING DOWN HERE
#%%
#%% Set suspect index
ind_sus=0
#%% 
#%% Buffer

#%% Search for bad peaks quickly
PeakDiff = np.diff(Peaks)
peak_ind=[x for x,peak_ind in enumerate(PeakDiff) if PeakDiff[x]<(np.mean(PeakDiff)*0.75) or PeakDiff[x]>(np.mean(PeakDiff)*1.5)]
print(peak_ind)
print("There are "+str(len(peak_ind))+" bad peaks currently")

# #%% Plot all bad peaks (do at the end of labelling as a double-check)
# for x in peak_ind:
#     viz_ECG(ecg, filt1,t_sens, Peaks, sr=sr,NZoom=[t_sens[Peaks[x-2]]-1.5,t_sens[Peaks[x+2]]+0.5])

#%% Buffer
#%% Buffer
#%% Change existing Peaks

ind_sus = 3467
shift   =   -30 #positive = right, negative = left
Peaks[ind_sus] += shift

viz_wid = 1
t_buf = 1
viz_ECG(ecg, filt1,t_sens, Peaks, sr=sr,NZoom=[t_sens[Peaks[ind_sus-viz_wid]]-t_buf,t_sens[Peaks[ind_sus+viz_wid]]+t_buf])

print("Changing Peak at index %d to time %2.3fs and ecg index %d, giving an original ecg value of %2.1f\nTime gap is +%3.3fs from previous peak and -%3.3fs from next peak" % (ind_sus, t_sens[Peaks[ind_sus]], Peaks[ind_sus], ecg[Peaks[ind_sus]],t_sens[Peaks[ind_sus]]-t_sens[Peaks[ind_sus-1]],t_sens[Peaks[ind_sus+1]]-t_sens[Peaks[ind_sus]]))




#%% Scan ahead

viz_ECG(ecg, filt1,t_sens, Peaks, sr=sr,NZoom=[t_sens[Peaks[ind_sus]]-0.2,t_sens[Peaks[ind_sus+6]]+0.2])
print("Gaps for index " + str(ind_sus) + "-" + str(ind_sus+5) + " are: " + str(np.diff([x for x in Peaks[ind_sus:ind_sus+6]])/sr))
#%% If good, add step to the index
step=5
ind_sus += step
print('Current Index is',(ind_sus))

#%% Reset ind_sus step
ind_sus -= np.mod(ind_sus-1,step)+1
print('Current Index is',(ind_sus))
#%% Buffer
#%% Buffer
#%% find local maximum 
Peaks = find_local_peaks(Peaks,ecg,width=1, f_twidth = False, sr=sr)
#%% Buffer
#%% Buffer
#%% Set suspect index
ind_sus=6130
#%% 
#%% Buffer
#%% Buffer
#%% Visualise potential Insert

ind_sus  = 2290
time     = 1107.75

#new_val  = round(time*sr)
new_val  = np.where(t_sens>=time)[0][0]

print('Visualising inserting Peak at index %d, time %2.3f and ecg index %d, giving an original ecg value of %2.1f\nTime gap is +%3.3fs from previous peak and -%3.3fs from next peak' % (ind_sus, t_sens[new_val], new_val, ecg[new_val],(t_sens[new_val]-t_sens[Peaks[ind_sus-1]]),(t_sens[Peaks[ind_sus]]-t_sens[new_val]) ))

viz_ECG(ecg, filt1, t_sens, np.insert(Peaks,ind_sus,new_val), sr=sr,NZoom=[time-1,time+1])

#%% Insert Peak

#ind_sus  = 1820
#time     = 820.43
#new_val  = round(time*sr)
Peaks = np.insert(Peaks,ind_sus,new_val)
    
print('Inserting Peak at index %d, time %2.3f and ecg index %d, giving an original ecg value of %2.1f' % (ind_sus, time, new_val, ecg[new_val]))

viz_ECG(ecg, filt1,t_sens, Peaks, sr=sr,NZoom=[time-1,time+1])

#%% Buffer
#%% Buffer
#%% Remove Peak

ind_sus = 2972

print('Removing Peak at index %d, time %2.3f and ecg index %d' % (ind_sus, t_sens[Peaks[ind_sus]], Peaks[ind_sus]))

Peaks = np.delete(Peaks,ind_sus)


#%% Check the things are hunky-dory after insert/removal
print(np.arange(ind_sus-3,ind_sus+3))
print(Peaks[ind_sus-3:ind_sus+3]/sr)




#%% Buffer
#%% Buffer
#%% Clean Peaks List (good for accidental wrong index on insert)
Peaks = np.unique(Peaks)
np.sort(Peaks)
Peaks = [int(x) for x in Peaks]

#%% Buffer
#%% Buffer


#%% List of questionable indices
ind_odd = [21,179,421,530,590,670,1222,1223,1323,1641,1909,1930,1950]
times_odd = t_sens[Peaks[ind_odd]]
print(times_odd)

#%% Buffer
#%% Buffer

#%%All plots


for x in np.arange(0,np.ceil(len(Peaks)/10)):
    ind_sus = int(x)*10
    max_ind = min(ind_sus+11,len(Peaks)-1)
    viz_ECG(ecg, filt1,t_sens, Peaks, sr=sr,NZoom=[t_sens[Peaks[ind_sus]]-0.2,t_sens[Peaks[max_ind]]+0.2])
    

#%% If good, add step to the index
step=5
ind_sus += step
print('Current Index is',(ind_sus))



#%% Save current values
svStr = paths[Npath].split("\\", -1)
svStr = svStr[-1].split(".", 1)
svStr = "G:\\My Drive\\BackUp\\Documents\\Data\\ProcessedData\\" + svStr[0] + Speak
print("Saving peaks in %s" % svStr)

np.savetxt(svStr,Peaks)
