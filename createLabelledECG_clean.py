# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 14:16:40 2021

Load in an unlabelled ECG signal, and create a set of labelled peaks.

STEPS:
    1. Load Libraries
    2. Load Functions
    3. Define user variable (flip ECG, is it a child, location of data, etc)
    4. Load ECG data
    5.a Create an initial peak list (chose only one of these)
    5.b Load an initial peak list (chose only one of these)
    ...
    6. Iteratively improve the peak list
        6a Find bad peaks
        6b Shift mislabelled peaks
        6c Add missed peaks
        6d Remove false peaks
    7. Check every peak at end
    8. Save "true" peak list (can be intermittently done throughout 6/7)
    
@author: Dr. Harry T. Mason

Sections 3 and 6 may require more advanced user interaction. All other steps should be able to be run without code manipulation

Either 5a or 5b should be run to create a list of peaks. 
"""

#%% 1. Load libraries
from   datetime import datetime
import fnmatch
import matplotlib.pyplot as plt
import neurokit2 as nk 
import numpy as np
import os
import pandas as pd
import scipy as scipy


#%% 2. Load functions

# File find function
def hf_find(pattern, path): # A function to search folders for tiles which match a certain string
  
    result = []
    for root,dirs,  files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

# ECG Processing
def hf_ecg(ecg,sr=500,freqfilt=[0.5,20]):
    # ECG preprocessing function
    # Does the standard neurokit2 preprocessing first, then custom frequency filter
    # ecg: the ecg signal to process
    # sr: the sampling_rate
    # freq_filt: [lower_limit, higher_limit] for bandwidth filter. If only one entry, assumed to be high pass filter. Recommended: [0.5] for adults, [15] for children.
    # returns ecg_out: the preprocessed ECG
    
    ecg_out = nk.ecg_clean(ecg, sampling_rate=sr, method="neurokit")    
    if len(freqfilt)==1:  #assume removing low freq noise
        sos = scipy.signal.butter(5, freqfilt, btype='highpass', output="sos", fs=sr)
    elif len(freqfilt)==2:
        if freqfilt[1]<=freqfilt[0]:
            return ecg_out*0
        else:
            sos = scipy.signal.butter(5, freqfilt, btype='bandpass', output="sos", fs=sr)
    ecg_out = scipy.signal.sosfiltfilt(sos, ecg_out)
    return ecg_out

    
# remove excess peaks (rough function)
def remove_peaks(r_peaks,sr=500):
    # Function to remove peaks that have been incorrectly added
    N_in = len(r_peaks)
    r_out=np.copy(r_peaks)      # Make a copy that can be changed
    ibi = np.diff(r_out)/sr     # Work out the timing difference between peaks
    ibi_mid = scipy.ndimage.median_filter(ibi,size=11, mode="nearest") # creates an array the length of ibi, where each entrant is the median over the specified window size# was np.median(ibi)
    gap_2 = (np.asarray(r_out[2:])-np.asarray(r_out[:-2]))/(ibi_mid[:-1]*sr) # works out the gap between point n and point n+2
    while(np.min(gap_2)<1.5):   # while a gap exists petween points n and n+2 that is less than 1.3 * the local median
          for ind in reversed(range(len(gap_2))):
              if gap_2[ind]<1.5:
                  r_out=np.delete(r_out,ind) # remove offending index
                  ibi = np.diff(r_out)/sr
                  ibi_mid = scipy.ndimage.median_filter(ibi,size=11, mode="nearest") 
                  gap_2 = (np.asarray(r_out[2:])-np.asarray(r_out[:-2]))/(ibi_mid[:-1]*sr)
    ibi = np.diff(r_out)/sr
    ibi_mid = scipy.ndimage.median_filter(ibi,size=41, mode="nearest")
    gap_1 = ibi/ibi_mid
    while(np.min(gap_1)<0.5):
        for ind in reversed(range(len(gap_1))):
              if gap_1[ind]<0.5:
                  r_out=np.delete(r_out,ind)
                  ibi = np.diff(r_out)/sr
                  ibi_mid = scipy.ndimage.median_filter(ibi,size=41, mode="nearest")
                  gap_1 = ibi/ibi_mid
    # print("Removed %d peaks" % (N_in-len(r_out)))
    return(r_out)
    
# Do local correction w.r.t. original ECG
def find_local_peaks(r_peaks,ecg_sig,width=5, f_twidth = False, sr=500):
    # Function to correct peak identified in preprocessing by looking for a local peak on the raw signal
    # r_peaks:  index of peaks found from preprocessed signal
    # ecg_sig:  the ecg signal
    # width:    the width (in raw samples or in time) of the searching function
    # f_twidth: whether the width indicated is the time range (f_twidth=True) or the number of indicies (f_twidth=False)
    # sr:       sampling rate
    
    if f_twidth:
        width = int(np.ceil(width*sr)) #Round up, ensure a width of at least 1
    
    r_out = np.copy(r_peaks)
    for x in np.arange(0,len(r_peaks)):  #len(Peaks)
        r_new=r_peaks[x]
        r_old=r_new+1   #just choosing some initial other value
        while r_old!=r_new:
            r_old=r_new
            min_i = np.max([r_old-width,0])
            max_i = np.min([r_old+width+1,len(ecg_sig)])
            miniSearch = ecg_sig[min_i:max_i]
            #print(min_i,max_i)
            maxLoc = np.where(miniSearch==np.amax(miniSearch))
            if ecg_sig[r_new]<ecg_sig[max(r_new+maxLoc[0][0]-width,0)]:
                r_new += maxLoc[0][0]-width
                if r_new>len(ecg_sig):  r_new=len(ecg_sig)
                elif r_new<0:           r_new=0
        r_out[x]=r_new
    r_out = np.unique(r_out)
    return(r_out)


# Plotting Function
def viz_ECG(ecg_old, ecg_new, times, peaks_new, sr=500, NZoom=[]):
    if not NZoom:
        NZoom = [0,len(ecg_old)/sr] 
    
    plt.figure(figsize=(18,10))
    
    a0  = plt.subplot(311)
    plt.plot(times,ecg_old, label = 'ECG signal', color='plum',linewidth=1)
    plt.scatter(times[peaks_new],[ecg_old[x] for x in peaks_new], color='green', label = 'identified peaks',linewidth=0.5)
    #plt.legend(loc='lower right')
    a0.set_xlim(NZoom)
    #plt.legend()
    try:
        a0.set_ylim([min(ecg_old[(times>NZoom[0])   & (times<NZoom[1])])-abs(max(ecg_old[(times>NZoom[0]) & (times<NZoom[1])])-min(ecg_old[(times>NZoom[0]) & (times<NZoom[1])]))*0.1,max(ecg_old[(times>NZoom[0]) & (times<NZoom[1])])+abs(max(ecg_old[(times>NZoom[0]) & (times<NZoom[1])])-min(ecg_old[(times>NZoom[0]) & (times<NZoom[1])]))*0.1])
    except:
        print("Warning: Limits of signal not found")
    plt.grid(visible=True,axis = 'x')
    a0.set_title("Original Signal")
    
    a1 = plt.subplot(312)
    plt.plot(times,ecg_new, label = 'ECG signal', color='plum',linewidth=1)
    plt.scatter(times[peaks_new],[ecg_new[x] for x in peaks_new], color='green', label = 'identified peaks',linewidth=0.5)
    #plt.legend(loc='lower right')
    a1.set_xlim(NZoom)
    plt.grid(visible=True,axis = 'x')
    a1.set_title("Filtered Signal")
    try:
        a1.set_ylim([min(ecg_new[(times>NZoom[0])   & (times<NZoom[1])])-abs(max(ecg_new[(times>NZoom[0]) & (times<NZoom[1])])-min(ecg_new[(times>NZoom[0]) & (times<NZoom[1])]))*0.1,max(ecg_new[(times>NZoom[0]) & (times<NZoom[1])])+abs(max(ecg_new[(times>NZoom[0]) & (times<NZoom[1])])-min(ecg_new[(times>NZoom[0]) & (times<NZoom[1])]))*0.1])
    except:
        print("Warning: Limits of signal not found")
    
    a2 = plt.subplot(313)
    plt.scatter(times[peaks_new],np.arange(len(peaks_new)), color='green')
    a2.set_xlim(NZoom)
    ind = [ind for ind,x in enumerate(peaks_new) if x>NZoom[0]*sr and x<NZoom[1]*sr]
    a2.set_ylim([ind[0],ind[-1]+1])
    a2.set_xlabel('Time (Seconds)')
    a2.set_title("Index of peaks")
    plt.grid(visible=True,axis='both')
    plt.show()
    

#%% 3. Define user variable (flip ECG, is it a child, location of data, etc)


f_child = 1
f_ecg_flip = 0

# Plux example
Ssesh = "SAS0611"
Sfol  = "G:\\My Drive\\BackUp\\Documents\\Data\\Study4\\EyeTracker + ECG study"

# Lauren Data example
Sfol = "G:\\.shortcut-targets-by-id\\1RdqXSr1qOd-Az8l0-4j3yeefkVVysmyo\\ECG_export_files\\B290_9mo\\"
Ssesh = "B290_9mo_pilot_20230713_104256"

# Sfol = "G:\\.shortcut-targets-by-id\\1RdqXSr1qOd-Az8l0-4j3yeefkVVysmyo\\ECG_export_files\\B303_9mo\\"
# Ssesh = "B303_9mo_ENI_pilot_20230717_102756"

Sfol = "G:\\.shortcut-targets-by-id\\1RdqXSr1qOd-Az8l0-4j3yeefkVVysmyo\\ECG_export_files\\B310_9mo\\"
Ssesh = "B310_9mo_pilot_20230724_015832"

# Sfol = "G:\\.shortcut-targets-by-id\\1RdqXSr1qOd-Az8l0-4j3yeefkVVysmyo\\ECG_export_files\\B316_9mo\\"
# Ssesh = "B316_9mo_pilot_20230721_110543"

#%% 4.a Load Data (Plux data as example)

Npath = 0;
Speak="_peaks.txt"

paths = hf_find(('opensignals*.txt'), ("%s\\%s\\" % (Sfol,Ssesh))) 
paths.sort()
print(paths)

cols =  ["nSeq", "DI", "CH1", "CH2", "CH3", "CH4_CORR", "CH5_LGHT", "CH6", "CH7", "CH8"]

ecg     = pd.Series(dtype=int)     # ecg
time_stamps = []

for filename in paths:

    df = pd.read_csv(filename, skiprows=[0,1,2], delimiter='\t', header = None)
    header = open(filename)
    all_lines_variable = header.readlines()
    tmp = all_lines_variable[1]
    tmpDict = eval(tmp[1:-1])
    
    # load child
    dict1       = tmpDict[list(tmpDict.keys())[0]] # dict with header info
    Ncol1       = len(dict1['column'])             # add existing cols
    df1         = df.loc[:,0:(Ncol1-1)]              # df w. all info
    sr          = dict1['sampling rate']
    cols1       = cols[0:2]+[cols[x+1] for x in dict1['channels']]
    df1.set_axis(cols1, axis=1, inplace=True) #rename columns
    # begin1      =     dict1['time']    
    time_stamps.append(datetime.strptime(dict1['time'],'%H:%M:%S.%f'))
    
    # append zeros to fill gaps
    start_i = int((time_stamps[-1] - time_stamps[0]).total_seconds()*sr) 
    ecg     = pd.concat([ecg,pd.Series(np.zeros(start_i-len(ecg)))], ignore_index=True)
    ecg     = pd.concat([ecg,df1['CH1']], ignore_index=True)      # ecg

if f_ecg_flip:
    ecg=-ecg
t_ecg = np.arange(0,len(ecg))/sr
    
#print signal info (length, etc.)
print("\n\nFile: %s\n%d samples.\n%2.1f minutes long.\n\n" %(filename,len(ecg),len(ecg)/sr/60 ))

#%% 4.b Load Data (Lauren data as example)

# Just run this cell

Npath = 0;
paths = hf_find("%s_ECG+Events.mat" %Ssesh , Sfol)
paths.sort()

ECGstr = paths[Npath]
print("Loading %s" % (ECGstr))

# Load ECG
ECGdict = scipy.io.loadmat(ECGstr)   # loading in data from the .mat file
ecg     = ECGdict[Ssesh+'mffECG'][0]    # extracting ECG
sr      = ECGdict['PNSSamplingRate'][0][0] # extracting sampling rate
t_ecg = np.arange(0,len(ecg)/sr,1/sr)
event_labels = ECGdict['evt_ECI_TCPIP_55513'][0]
event_indexes= ECGdict['evt_ECI_TCPIP_55513'][1]

event_labels = [x[0] for x in event_labels]
event_indexes = [x[0][0].astype(int)-1 for x in event_indexes]

if (t_ecg[-1]-t_ecg[0])>3600:
    print("%s length: %1.3fs (%1.2fh)" % (Ssesh,t_ecg[-1]-t_ecg[0],(t_ecg[-1]-t_ecg[0])/3600))
else:
    print("%s length: %1.3fs" % (Ssesh,t_ecg[-1]-t_ecg[0]))

# del df_ecg # uncomment if you are on a slow computer for efficiency, important if it's a very large file


#%% 5.a Create an initial peak list
if f_child:
    filt1 = hf_ecg(ecg,sr=sr,freqfilt=[15])   
else:
    filt1 = hf_ecg(ecg,sr=sr,freqfilt=[0.5])
Peaks   = nk.ecg_findpeaks(np.concatenate((np.ones(int(sr))*filt1[0],filt1,np.ones(int(sr))*filt1[-1])), sampling_rate=sr, method="neurokit")
Peaks   = np.asarray(Peaks['ECG_R_Peaks'])
Peaks   = np.array([int(x-sr) for x in Peaks if x>sr and x<len(ecg)+sr])


Peaks = find_local_peaks(Peaks,ecg)
Peaks = remove_peaks(Peaks,sr)
print("%d peaks found" % len(Peaks))


#%% 5.b Load Existing Peaklist
ldStr = paths[Npath].split("\\", -1)
ldStr = ldStr[-1].split(".", 1)
ldStr = Sfol + ldStr[0] + Speak
Peaks=np.loadtxt(ldStr,dtype=int)
filt1=ecg
filt1 = hf_ecg(ecg,sr=sr,freqfilt=[15])  


#%% Buffer
#%% 
#%% PROCESSING DOWN HERE
#%% PROCESSING DOWN HERE
#%% PROCESSING DOWN HERE
#%%
#%% This is where you start to edit the list of peaks
#%% 6. spare - Set suspect index
ind_sus=0

#%% Buffer
#%% Buffer
#%% 6.a.i Search for bad peaks quickly (time gap)
DiffLim = 0.08 # Size of time gap to look for. Suggest using 0.2 as an initial check, then 0.1, 0.05, and 0.03. Many valid peaks will likely be included in the latter searches

PeakDiff = np.abs(np.diff(np.diff(t_ecg[Peaks])))
peak_ind=[x+1 for x,peak_ind in enumerate(PeakDiff) if PeakDiff[x]>DiffLim]
print(peak_ind)
print("There are "+str(len(peak_ind))+" bad peaks currently")


#%% 6.a.ii Plot all bad peaks 
for x in peak_ind:
    viz_ECG(ecg, filt1,t_ecg, Peaks, sr=sr,NZoom=[t_ecg[Peaks[max(x-2,0)]]-1.5,t_ecg[Peaks[min(x+2,len(Peaks)-1)]]+0.5])


#%% Buffer
#%% Buffer
#%% 6.b.i Change existing Peaks

ind_sus =  1102
shift   =      0   #positive = right, negative = left

Peaks[ind_sus] += shift
viz_wid = 0
t_buf = 1.3
viz_ECG(ecg, filt1,t_ecg, Peaks, sr=sr,NZoom=[t_ecg[Peaks[max(0,ind_sus-viz_wid)]]-t_buf,t_ecg[Peaks[min(ind_sus+viz_wid,len(Peaks)-1)]]+t_buf])
print("Changing Peak at index %d to time %2.3fs and ecg index %d, giving an original ecg value of %2.1f\nTime gap is +%3.3fs from previous peak and -%3.3fs from next peak" % (ind_sus, t_ecg[Peaks[ind_sus]], Peaks[ind_sus], ecg[Peaks[ind_sus]],t_ecg[Peaks[ind_sus]]-t_ecg[Peaks[max(0,ind_sus-1)]],t_ecg[Peaks[min(ind_sus+1,len(Peaks)-1)]]-t_ecg[Peaks[ind_sus]]))

#%% 6.b.ii. Find local maximum (saves having to find the exact location once a label is roughly on a peak)
Peaks = find_local_peaks(Peaks,ecg,width=0.004, f_twidth = True, sr=sr)
viz_ECG(ecg, filt1,t_ecg, Peaks, sr=sr,NZoom=[t_ecg[Peaks[max(0,ind_sus-viz_wid)]]-t_buf,t_ecg[Peaks[min(ind_sus+viz_wid,len(Peaks)-1)]]+t_buf])
print("Index %d at time %2.3fs has time gap is +%3.3fs from previous peak and -%3.3fs from next peak" % (ind_sus, t_ecg[Peaks[ind_sus]], t_ecg[Peaks[ind_sus]]-t_ecg[Peaks[max(0,ind_sus-1)]],t_ecg[Peaks[min(ind_sus+1,len(Peaks)-1)]]-t_ecg[Peaks[ind_sus]]))

#%% Buffer
#%% Buffer
#%% 6.c.i Visualise potential Insert (doesn't commit an insert, just visualises)

ind_sus  = 89

# time     = 1785.316  # insert at a particular time
time     = t_ecg[Peaks[ind_sus-1]]+0.45 # insert at a certain offset after previous peak
time     = t_ecg[Peaks[ind_sus]]-0.45 # insert at an offset before next peak

new_val  = np.where(t_ecg>=time)[0][0]
print('Visualising inserting Peak at index %d, time %2.3f and ecg index %d, giving an original ecg value of %2.1f\nTime gap is +%3.3fs from previous peak and -%3.3fs from next peak' % (ind_sus, t_ecg[new_val], new_val, ecg[new_val],(t_ecg[new_val]-t_ecg[Peaks[max(0,ind_sus-1)]]),(t_ecg[Peaks[ind_sus]]-t_ecg[new_val]) ))
viz_ECG(ecg, filt1, t_ecg, np.insert(Peaks,ind_sus,new_val), sr=sr,NZoom=[time-1,time+1])
#%% 6.c.ii Insert Peak (Commits the insert from 6.c.i)
Peaks = np.insert(Peaks,ind_sus,new_val)
Peaks = find_local_peaks(Peaks,ecg,width=0.004, f_twidth = True, sr=sr)
print('Inserting Peak. Peak index: %d, Time %2.3f, Pre gap: +%3.3fs, Post-gap: -%3.3fs, ECG index: %d, ECG value: %2.1f' % (ind_sus, time, t_ecg[Peaks[ind_sus]]-t_ecg[Peaks[max(0,ind_sus-1)]],t_ecg[Peaks[min(ind_sus+1,len(Peaks)-1)]]-t_ecg[Peaks[ind_sus]], new_val,ecg[new_val]))
viz_ECG(ecg, filt1,t_ecg, Peaks, sr=sr,NZoom=[time-1,time+1])

#%% Buffer
#%% Buffer
#%% 6.d.i Remove Peak

ind_sus = 841

print('Removing Peak at index %d, time %2.3f and ecg index %d' % (ind_sus, t_ecg[Peaks[ind_sus]], Peaks[ind_sus]))
Peaks = np.delete(Peaks,ind_sus)
#%% 6.d.ii Check the things are hunky-dory after insert/removal
print(np.arange(ind_sus-3,ind_sus+3))
print(t_ecg[Peaks[max(0,ind_sus-3):min(ind_sus+3,len(Peaks)-1)]])


#%% Buffer
#%% Buffer
#%% 7. All plots
peaks_width = 10 # How many peaks to include in each plot

for x in np.arange(0,np.ceil(len(Peaks)/peaks_width)):
    ind_sus = int(x)*peaks_width
    max_ind = min(ind_sus+peaks_width+1,len(Peaks)-1)
    viz_ECG(ecg, filt1,t_ecg, Peaks, sr=sr,NZoom=[t_ecg[Peaks[ind_sus]]-0.2,t_ecg[Peaks[max_ind]]+0.2])
    

#%% Buffer
#%% Buffer
#%% 8.a Clean Peaks List (Do before saving)
Peaks = np.unique(Peaks)
np.sort(Peaks)
Peaks = [int(x) for x in Peaks]
Peaks = find_local_peaks(Peaks,ecg,width=0.004, f_twidth = True, sr=sr)

#%% 8.b Save current values
svStr = paths[Npath].split("\\", -1)
svStr = svStr[-1].split(".", 1)
svStr = Sfol + svStr[0] + "_peaks.txt"
print("Saving peaks in %s" % svStr)
np.savetxt(svStr,Peaks)
