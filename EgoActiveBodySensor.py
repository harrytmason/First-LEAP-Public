# -*- coding: utf-8 -*-
"""
Body Sensor processing Pipeline for EgoActive device

Written by Dr Harry T. Mason, University of York
Additional work from Prof. Marina I. Knight and Dr. David R. Mullineaux 

The structure of the file goes like this:
    
    1. Load Libraries
    2. Load Functions
    3. Set the flags and filename for the session 
    4. Convert .dat files to .txt (if the .txt file doesn't already exist)
    5. Load ECGdata.txt and ACCdata.txt
    6. Find if a synchronization signal exists
    7. Reset ECG and ACC data time series
    8. Process ACC data
    9. Process ECG data (split into 4 parts)
    10. Convert ECG data into heart rate
    11. Process heart rate
    12. Run signal quality metrics on processed heart rate
    13. Save the output heart rate and accelerometer data
    
Users are required to put inputs for section 3, 9b
Otherwise the cell should be run without alteration
More advanced users can alter the settings throughout the script as they desire

"""
#%% 1. Load Libraries


import fnmatch
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import neurokit2 as nk     # The only package that doesn't come with default Python
import numpy as np
import os as os
import pandas as pd
import scipy
from   scipy.interpolate import interp1d
from   scipy.ndimage import median_filter
from   scipy.signal import find_peaks
import struct as struct

#%% 2. Load Functions

# A function to search folders for tiles which match a certain string
def hf_find(pattern, path): 
    result = []
    for root,dirs,  files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

# Plotting function
def viz_ECG_mini(ax,ecg_sig, peaks_new,time, NZoom=[], Title = "Filtered Signal",ylab="",xlab='Time (seconds)',col="plum"):
    if len(NZoom)==0:
        NZoom = [0,max(time)]
        ind_sta=0
        ind_end=len(time)
    else:
        try:    ind_sta = np.where(time<=NZoom[0])[0][-1]
        except: ind_sta=0
        try:    ind_end = np.where(time>=NZoom[1])[0][0]
        except: ind_end=len(time)
    plt.plot(time[ind_sta:ind_end],ecg_sig[ind_sta:ind_end], label = 'ECG signal', color=col,linewidth=2)
    if len(peaks_new)>0:
        plt.scatter(time[peaks_new],ecg_sig[peaks_new], color='black', marker='x',label = 'identified peaks',zorder=10)
#plt.legend()
    try:
        plt.ylim([min(ecg_sig[(time>NZoom[0]) & (time<NZoom[1])])-abs(max(ecg_sig[(time>NZoom[0]) & (time<NZoom[1])])-min(ecg_sig[(time>NZoom[0]) & (time<NZoom[1])]))*0.1,max(ecg_sig[(time>NZoom[0]) & (time<NZoom[1])])+abs(max(ecg_sig[(time>NZoom[0]) & (time<NZoom[1])])-min(ecg_sig[(time>NZoom[0]) & (time<NZoom[1])]))*0.1])
    except:
        # print("Warning: Limits of signal not found")
        plt.ylim([0,3.3])
    ax.set_xlim(NZoom)
    ax.set_title(Title,fontsize=20)
    ax.set_xlabel(xlab,fontsize=16)
    ax.set_ylabel(ylab,fontsize=16)
    

# Create a boolean array from changepoint
def create_boolean(bkps,array_len,zero_start=True):
    # NB: if bkps are not unique, then 2 repeated bkps will cancel out
    
    bool_vec = np.zeros(array_len)        # create array
    
    if not zero_start:      bool_vec +=1      # choose if startng at zero or 1
    if bkps[0]==0:          bkps = bkps[1:]   # ignore any leading zero on breakpoints
    if bkps[-1]==array_len: bkps = bkps[:-1]  # ignore the end breakpoint if just the length of the vector
    
    for x in bkps:
        if   bool_vec[x]==0: bool_vec[x:] = 1
        elif bool_vec[x]==1: bool_vec[x:] = 0
    #bool_vec = np.asarray([ int(x) for x in bool_vec]) #If I need it as integers
    return bool_vec

# ECG preprocessing function
def hf_ecg(ecg,sr=500,freqfilt=[0.5,20]):
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

# Peak correction function
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

# Function to clean up the heart rate signal by interpolating through values that fall outside a local median
def hrSQI(hr,sr=500, good_fact=1.3, Nfilt=31):
    # hr: the heart rate signal to evaluate
    # good_fact: the multiplicative factor defining the limits of a good signal
    # Nfilt: the width of the median filter (e.g. Nfilt = 3 would include x-1, x, x+1)
    # OUTPUTS
    # new_hr:    the filtered heart rate
    # qual_hr:   a 2-column matrix. The first column is the distance any bad hr measurement is from the boundary. The second column is the proportion of "bad" hr measurments within the filter
    
    hr_med = median_filter(hr,size=Nfilt,mode='nearest')
    hr_med_low = hr_med/good_fact
    hr_med_up  = hr_med*good_fact
    Nhr   = len(hr)
    
    good_hr = (hr>hr_med_low) & (hr<hr_med_up)

    qual_hr= np.zeros((Nhr,2))
    new_hr  = np.copy(hr)
    
    
    for ind,x in enumerate(hr):
        
        # Number of good hr_indices inside proposed median filter
        qual_hr[ind,1] = np.sum(good_hr[max(0,int(ind-Nfilt/2)):min(Nhr,int(ind+Nfilt/2+Nfilt%2))])/len(good_hr[max(0,int(ind-Nfilt/2)):min(Nhr,int(ind+Nfilt/2+Nfilt%2))])
        
        if not good_hr[ind]:
            # Distance beyond the boundary at this index
            qual_hr[ind,0]=max((hr_med_low[ind]-x)/hr_med_low[ind],(x-hr_med_up[ind])/hr_med_up[ind])
            
            # Either chose the best from 20 either side, or from filter width - since it's only finding a good value to linearly filter between, not as important
        
            start_i   = int(np.max([0,  ind-20])) # Dealing with the filter near the start of the signal
            end_i     = int(np.min([Nhr,ind+20])) # Dealing with the filter near the end of the signal
            
            ind_lr = np.concatenate((np.arange(start_i,ind),np.arange(ind+1,end_i))) # excluding the bad index
            
            ind_lr = ind_lr[good_hr[ind_lr]]             # excluding other bad indices around it
            if len(ind_lr)==1:                           # if only one good index remains, use that
                new_hr[ind] = hr[ind_lr]
            elif len(ind_lr)==0:                         # no good indices remain
                new_hr[ind] = hr[ind] # redundant, but clearly stating that a value is kept the same if nothing can replace it
            else:
                f=interp1d(ind_lr,hr[ind_lr],kind='slinear',fill_value="extrapolate") #linear interpolation
                new_hr[ind] = f(ind)
            
    return new_hr, qual_hr

# Function to clean mislabelled beats (e.g. early or late of the actual beat) in a heart rate signal 
def hrCleanBeats(hr,thresh1=15,thresh2=25):
    # hr: heart rate signal
    # new_hr : cleaned heart rate signal
    #
    # Find locations where:
    #       a) the difference is greater than thresh1 for 3 consecutive recordings, 
    #       b) greater than thresh2 for the middle recording and 
    #       c) the sign switches for all 3 beats
    # At these locations, lineraly interpolate the solution
    # age input (in absence of other 2) allows for a pre-established threshold to be used
    
    # Input handling
    if len(hr)<3: return hr
 
    hr_gtind = np.zeros(len(hr))   #hr greater than indices
    for x in np.arange(len(hr)-3):
        hr_gtind[x] = np.sum(np.abs(np.diff(hr[x:x+4]))>thresh1)==3 and np.abs(np.diff(hr[x+1:x+3]))>thresh2
    hr_signind = np.abs(np.convolve(np.sign(np.diff(hr)),[1,-1,1]))==3 #hr sign indices
    hr_ind = hr_gtind[:-1] * hr_signind[2:] #combining the two metrics
    
    hr_bad = np.sign(np.concatenate(([0,0],hr_ind[:-1]))+np.concatenate(([0],hr_ind)))
    
    # Create a copy of the heart rate ready to be cleaned 
    hr2 = np.copy(hr)
    
    for ind,x in enumerate(hr[:-1]):
        
        if hr_bad[ind]:
    
            start_i   = int(np.max([0,  ind-20]))     # Dealing with the filter near the start of the signal
            end_i     = int(np.min([len(hr),ind+20])) # Dealing with the filter near the end of the signal
            # Either chose the best from 20 either side, or from filter width - since it's only finding a good value to linearly filter between, not as important
              
            ind_lr = np.arange(start_i,end_i)        # indices from left and right
            ind_lr = ind_lr[(1-hr_bad[ind_lr])==1]   # excluding bad indices 
            if len(ind_lr)==1:                       # if only one good index remains, use that
                hr2[ind] = hr[ind_lr]
            elif len(ind_lr)==0:                     # no good indices remain
                hr2[ind] = hr[ind]                   # redundant, but clearly stating that a value is kept the same if nothing can replace it
            else:
                f=interp1d(ind_lr,hr[ind_lr],kind='slinear',fill_value="extrapolate") #linear interpolation
                hr2[ind] = f(ind)
    return hr2

# Function for Heart rate quality assessment (Signal Quality Index)
def sqi_clean(sqi,hr_raw,hr_filt,t_hr,sqi_thresh=0.75,Shr = ""):
    
    
    # First SQI calc
    sqi_hr_thresh = (sqi[:,1]>sqi_thresh)*1
    
    
    # 
    # Remove bad SQI regions (missing HR) (at start and after flip)
    #
    
    # Find places the time gap is longer than 2.5s
    ind_gap = (np.convolve(np.abs(np.diff(t_hr))>2.5,[1,1])>1)*1*np.arange(0,len(hr_raw)) 
    
    # Set to zero
    sqi_hr_thresh[ind_gap]=0
    
     
    # 
    # Flip small SQI regions
    #
    
    # Find all indexes
    qual_hr_cps = np.unique(np.concatenate(([0],np.where(np.abs(np.diff(sqi_hr_thresh))==1)[0]+1,[len(sqi_hr_thresh)-1])))
    
    # find all time gaps
    t_hr_cps = np.diff(t_hr[qual_hr_cps])
    
    while min(t_hr_cps)<5:
        
        # find minimum value index
        ind_min = np.argmin(t_hr_cps)
        
        # change SQI? (1- current value)?
        sqi_hr_thresh[qual_hr_cps[ind_min]:qual_hr_cps[ind_min+1]]=1-sqi_hr_thresh[qual_hr_cps[ind_min]:qual_hr_cps[ind_min+1]]
        
        #remove those two changepoints
        qual_hr_cps = np.delete(qual_hr_cps,[ind_min,ind_min+1])
        
        
        # Refresh indexes (in case 0/end were removed)
        qual_hr_cps = np.unique(np.concatenate(([0],np.where(np.abs(np.diff(sqi_hr_thresh))==1)[0]+1,[len(sqi_hr_thresh)-1])))
        
        # find all time gaps
        t_hr_cps = np.diff(t_hr[qual_hr_cps])
        
        
    # 
    # Slide SQI regions
    #
    #cycle through every qual changepoint (not start/end)
    # find which side is SQI up? (always slide out, since SQI threshold is >0.5)
    # slide for up to 15 points, the start/end, or to next cp - whichever is nearer
        # start and end indexes are in qual_hr_cps array, but done explicitly anyway
        
    # Find specific bad locations
    hr_bad = (hr_filt!=hr_raw)*1
    
    # Find all indexes
    qual_hr_cps = np.unique(np.concatenate(([0],np.where(np.abs(np.diff(sqi_hr_thresh))==1)[0]+1,[len(sqi_hr_thresh)-1])))
    
    # find all time gaps
    t_hr_cps = np.diff(t_hr[qual_hr_cps])
    
    for ind,cp in enumerate(qual_hr_cps[1:-1]):
        if sqi_hr_thresh[cp]==1: #slide left
            ind_slide_lim = max(0,cp-15,qual_hr_cps[ind]) 
            cp_new = np.copy(cp)
            for x in np.arange(cp-1,ind_slide_lim,-1):
                if hr_bad[x]==0:
                    sqi_hr_thresh[x]=1
                    cp_new-=1
                else:
                    ind_slide_end = max(x-4,ind_slide_lim)
                    if np.sum(1-hr_bad[ind_slide_end:x+1])>=3:
                        sqi_hr_thresh[ind_slide_end:x+1]=1
                        cp_new=ind_slide_end
                    else:
                        break
            else:
                continue
        elif sqi_hr_thresh[cp]==0: #slide left
            ind_slide_lim = min(len(sqi_hr_thresh),cp+15,qual_hr_cps[ind+2]) 
            cp_new = np.copy(cp)
            for x in np.arange(cp,ind_slide_lim,1):
                if hr_bad[x]==0:
                    sqi_hr_thresh[x]=1
                    cp_new+=1
                else:
                    ind_slide_end = min(x+3,ind_slide_lim)
                    if np.sum(1-hr_bad[x:ind_slide_end])>=2:
                        sqi_hr_thresh[x:ind_slide_end]=1
                        cp_new=ind_slide_end
                    else:
                        break
            else:
                continue
     
 
    # 
    # Remove bad SQI regions (majority bad)
    #
    # if SQI is 1 but the region is majority bad meausres, remove than index and set SQI to zero
    
    qual_hr_cps_copy = np.copy(qual_hr_cps)
    hr_bad = (hr_filt!=hr_raw)*1
    
    
    for ind,x in reversed(list(enumerate(qual_hr_cps_copy[:-1]))):
        if np.mean(hr_bad[x:qual_hr_cps_copy[ind+1]])>0.5 and np.mean(sqi_hr_thresh[x:qual_hr_cps_copy[ind+1]])==1:
            sqi_hr_thresh[x:qual_hr_cps_copy[ind+1]]=0
            qual_hr_cps = np.delete(qual_hr_cps,[ind,ind+1]) 
            qual_hr_cps = np.unique(np.concatenate(([0],qual_hr_cps,[len(sqi_hr_thresh)-1])))

    
    # 
    # Include good SQI regions (majority good)
    #
    # if SQI is 1 but: 
    #   the region is majority good measures - too generous, set to 75%?
    #   the region is under 15s in length
    # then: remove than index and set SQI to zero
    
    qual_hr_cps_copy = np.copy(qual_hr_cps)
    hr_bad = (hr_filt!=hr_raw)*1
    t_hr_cps = np.diff(t_hr[qual_hr_cps])
    
    for ind,x in reversed(list(enumerate(qual_hr_cps_copy[:-1]))):
        if np.mean(hr_bad[x:qual_hr_cps_copy[ind+1]])<0.25 and np.mean(sqi_hr_thresh[x:qual_hr_cps_copy[ind+1]])==0 and t_hr_cps[ind]<15:
            sqi_hr_thresh[x:qual_hr_cps_copy[ind+1]]=1
            qual_hr_cps = np.delete(qual_hr_cps,[ind,ind+1]) 
            qual_hr_cps = np.unique(np.concatenate(([0],qual_hr_cps,[len(sqi_hr_thresh)-1])))
     
    
    
    # 
    # Remove bad SQI regions (Continuous bad HR)
    #
    # Bad regions longer than 3.5s long
    
    # Find places the time gap is longer than 3.5s 
    hr_bad = (hr_filt!=hr_raw)*1
    bad_hr_cps = np.unique(np.concatenate(([0],np.where(np.abs(np.diff(hr_bad))==1)[0]+1,[len(hr_bad)-1])))
    t_hr_bad = np.diff(t_hr[bad_hr_cps])
    
    for ind,x in enumerate(bad_hr_cps[:-1]):
        if (t_hr_bad[ind])>3.5 and hr_bad[x]==1 and np.sum(sqi_hr_thresh[x:bad_hr_cps[ind+1]])!=0:
            sqi_hr_thresh[x:bad_hr_cps[ind+1]]=0

    
    # 
    # Remove bad SQI regions (missing HR)
    #
    # time gaps longer than 2.5s
    
    # Find places the time gap is longer than 2s 
    ind_gap = (np.convolve(np.abs(np.diff(t_hr))>2.5,[1,1])>1)*1*np.arange(0,len(hr_raw)) 
    
    # check zero index (which is sometimes caught incorrectly by the above)
    f_zero = (t_hr[1]-t_hr[0]<=2.5 and sqi_hr_thresh[0])*1
    
    # Set long gaps to zero
    sqi_hr_thresh[ind_gap]=0
    
    # reset zero
    sqi_hr_thresh[0]=f_zero
    
    
    # 
    # Flip small SQI regions down (to make sure no small good regions are left after above removals)
    #
    
    # Find all indexes
    qual_hr_cps = np.unique(np.concatenate(([0],np.where(np.abs(np.diff(sqi_hr_thresh))==1)[0]+1,[len(sqi_hr_thresh)-1])))
    
    # find all time gaps
    t_hr_cps = np.diff(t_hr[qual_hr_cps])
    
    # find remaining gaps longer than 5 (shouldn't be many, and since only flipping down don't need to go smallest -> biggest)
    ind_smol = np.where(t_hr_cps<5)[0]

    if len(ind_smol)>0:
        for x in ind_smol:
            
            if sqi_hr_thresh[qual_hr_cps[x]]==1:
                
                # change SQI 
                sqi_hr_thresh[qual_hr_cps[x]:qual_hr_cps[x+1]]=1-sqi_hr_thresh[qual_hr_cps[x]:qual_hr_cps[x+1]]
    
    
    return sqi_hr_thresh

#%% 3. Set the flags and filename for the session 
# Edit this cell with the appropriate strings and flags

# FILEPATHstr = "G:\\Shared drives\\First_Leap_II\\Data\\York\\11mo\\Home261\\Body_Sensor\\UOY48B\\" # folder containg the .dat file
# Bstr = "B002" # The recording to process

FILEPATHstr = "G:\\Shared drives\\First_Leap_II\\Data\\York\\13mo\\Home319\\Body_Sensor\\UOY73B\\" # folder containg the .dat file
Bstr = "B005" # The recording to process (B008 before)

FILEPATHstr = "G:\\Shared drives\\First_Leap_II\\Data\\York\\6mo\\Home316\\Body_Sensor\\UOY80B\\" # folder containg the .dat file
Bstr = "B005" # The recording to process (B008 before)

f_plt = 1     # Set to 1 to plot results, set to 0 for no plots
f_infant = 1  # Set to 1 if the participant is an infant, 0 if it is not 
f_save = 0    # Set to 1 to save the output hr and accelerometer data, 0 to not save

quality_threshold = 0.75 # Pick a value between 0 -> 1. This roughly represents the number of confident peaks within a window

#%% 4. Convert .dat files to .txt (if the .txt file doesn't already exist)
# Just run this cell

path = hf_find("%s-000*.DAT" % Bstr,FILEPATHstr)[0]

FILENAMEstr      = path[::-1].split('\\',1)[0][::-1] # e.g. B015-000.DAT
FILE_SESSION_num = int(FILENAMEstr[1:4])             # e.g. 000
FOLDER_BASEstr   = path[::-1].split('\\',1)[1][::-1]+"\\"
FOLDERstr        = FOLDER_BASEstr+"B"+'{:03d}'.format(FILE_SESSION_num)+"\\"
ENDstr           = FILENAMEstr.split("B"+'{:03d}'.format(FILE_SESSION_num)+"-000")[-1].split(".DAT")[0] # Extract anything tagged on after Bxxx-xxx

FILE_SESSION_idx = 000  #resets each time
lineNum = 0             #resets each time
ecg = [ 0, 0, 0, 0 ]    #resets each time

if not os.path.exists(FOLDERstr+"\\ECGdata.txt"): #If file already exists, don't process
    if not os.path.exists(FOLDERstr):
        os.mkdir(FOLDERstr)
        print("Creating "+FOLDERstr+" folder")
    fecg = open(FOLDERstr+ "ECGdata.txt", 'w')
    facc = open(FOLDERstr+ "ACCdata.txt", 'w')
    if ENDstr != "":
        fmeta = open(FOLDERstr+ "meta.txt", 'w')
        fmeta.write( ENDstr )
        fmeta.close()
    
    #% Save
    
    for FILE_SESSION_idx in range(0, 10000) :
    
      FILENAMEstr = FOLDER_BASEstr + "B" + '{:03d}'.format(FILE_SESSION_num) + '-' + '{:03d}'.format(FILE_SESSION_idx) + ENDstr + ".DAT"
    
      # print( "File " + FILENAMEstr, end=": " )
    
      # Open the LOG file and read the data
      try:
        fi = open( FILENAMEstr, 'rb')
      except FileNotFoundError :
        print( "Ended with %s" % FILENAMEstr)
        if FILE_SESSION_idx>0:
            fi.close()  
        facc.close()
        fecg.close()
        break
    
      # print( "Open, Reading ..." )
    
      while( df := fi.read(512) ) :
    

        ( id_V, ) = struct.unpack( "H", df[ 0 : 2 ] )
        device_ID = ( id_V >> 8 ) & 0x7F
        Vbatt = ( id_V & 0x7F ) * ( 2*3.30/127 )
      
        el = 0
        while( el < 85 ) :
          frameOffset = 2 + ( el * 6 )
          ( flags, ) = struct.unpack( "L", df[ frameOffset : (frameOffset+4)] )
      
          if ( flags & 0x40000000 ) :
            el += 1
            ( ts, ecg ) = struct.unpack( "LH", df[ frameOffset : (frameOffset + 6) ] )
      
            # Write to txt file "ECG_photostate ECG_time ECG_arr\n"
            fecg.write( '{:d} '.format( ( ecg >> 15 ) & 0x0001 ) )
            fecg.write( '{:0.3f} '.format( ( ts & 0x0FFFFFFF ) / 1000.0 ) )
            fecg.write( '{:0.3f} '.format( ( ( ecg & 0x0FFF ) * ( 3.3 / 1024 ) ) ) )
            fecg.write( '\n' )
      
          elif ( flags & 0x20000000 ) :
            el += 1
            ( ac_ts, acx ) = struct.unpack( "Lh", df[ frameOffset : (frameOffset + 6) ] )
      
          elif ( flags & 0x10000000 ) :
            el += 1
            ( acy, fl, acz ) = struct.unpack( "hHh", df[ frameOffset : (frameOffset + 6) ] )
      
            # Write to txt file "ACC_photostate ACC_time xdata ydata zdata\n"
            facc.write( '{:d} '.format( ( ecg >> 15 ) & 0x0001 ) )
            facc.write( '{:0.3f} '.format( ( ac_ts & 0x0FFFFFFF ) / 1000.0 ) )
            facc.write( '{:0.3f}'.format( acx * ( 2.000 / 32768 ) ) + " " + '{:0.3f}'.format( acy * ( 2.000 / 32768 ) ) + " " + '{:0.3f}'.format( acz * ( 2.000 / 32768 ) ) )
            facc.write( '\n' )
          else :
            print( "Bad Flag" )
    
      fi.close()
    else:
        print( "File " + FILENAMEstr + " already exists" )
    
    facc.close() # run this line if code is stopped halfway
    fecg.close() # run this line if code is stopped halfway


#%% 5. Load ECGdata.txt and ACCdata.txt

# Just run this cell
# The only exception - comment out the final two lines to delete the pandas data structure if required for performance reasons

ECGstr = FOLDERstr+"ECGdata.txt"
ACCstr = FOLDERstr+"ACCdata.txt"

print("Loading %s and %s" % (ECGstr, ACCstr))

# Load ECG
df_ecg = pd.read_csv(ECGstr,delimiter=' ',engine='python',header=None,names=['ecg_lum','t_ecg','ecg'],index_col=False)
df_ecg = df_ecg.dropna(axis=0)

while min(np.diff(df_ecg['t_ecg']))<-100: # Stitch data with bad time series indexing
    ind = np.where(np.diff(df_ecg['t_ecg'])<-100)[0][-1]
    print('Stitching together data at %d index, or %2.3fs'%(ind,df_ecg['t_ecg'][ind]))
    df_ecg.loc[np.arange(ind+1,len(df_ecg)),['t_ecg']]+= df_ecg['t_ecg'][ind]

df_ecg   = df_ecg.sort_values('t_ecg')

# Extract ECG from data structure
ecg      = np.asarray(df_ecg['ecg'])     # ECG values from 0->3.297
t_ecg    = np.asarray(df_ecg['t_ecg'])   # Time indexing for ECG
ecg_lum  = np.asarray(df_ecg['ecg_lum']) # The captured luminosity signal
sr_ecg   = 1/np.median(np.diff(t_ecg))

# Load accelerometer data
df_acc = pd.read_csv(ACCstr,delimiter=' ',engine='python',header=None,names=['acc_lum','t_acc','acc_x','acc_y','acc_z'])
df_acc = df_acc.dropna(axis=0)
df_acc = df_acc.sort_values('t_acc')

# Extract ACC from data structure
acc_x    = np.asarray(df_acc['acc_x'])           # Accelerometer data along sensor x-axis
acc_y    = np.asarray(df_acc['acc_y'])           # Accelerometer data along sensor y-axis
acc_z    = np.asarray(df_acc['acc_z'])           # Accelerometer data along sensor z-axis
acc      = (acc_x**2 + acc_y**2 + acc_z**2)**0.5 # Accelerometer magnitude data
t_acc    = np.asarray(df_acc['t_acc'])           # Time indexing for Accelerometer data
acc_lum  = np.asarray(df_acc['acc_lum'])         # Not used, but here for comparison
sr_acc   = 1/np.median(np.diff(t_acc))


if (t_ecg[-1]-t_ecg[0])>3600:
    print("%s length: %1.3fs (%1.2fh)" % (FOLDERstr,t_ecg[-1]-t_ecg[0],(t_ecg[-1]-t_ecg[0])/3600))
else:
    print("%s length: %1.3fs" % (FOLDERstr,t_ecg[-1]-t_ecg[0]))

# del df_ecg # uncomment if you are on a slow computer for efficiency, important if it's a very large file
# del df_acc # uncomment if you are on a slow computer for efficiency, important if it's a very large file



#%% 6. Find if a synchronization signal exists

# Just run this cell

if np.sum(np.diff(ecg_lum))==0:
    print("***No changes in luminosity signal found***")
else:
    
    #% Manipulating Sync

    sync = (ecg_lum-0.5)*2 #rescale to -1 -> +1

    sync_updn = np.where(np.abs(np.diff(sync))==2)[0] # find up and down points
    sync_bkps = np.zeros(len(sync_updn))    # wrapper for new updown points
    
    f_zero_start = not ecg_lum[1] #Did signal start switched on? (it ignores the first value, hence picking the second)
    t_sync = np.arange(np.floor(t_ecg[0]),t_ecg[-1],0.002) #could make to only apply to first 5 minutes
    
    sync_bkps = [int(np.abs(t_sync - t_ecg[sync_updn[x]]).argmin()) for x in  np.arange(len(sync_updn))] #find the nearest equivalent timepoint in t_sync to t_ecg[sync_updn]
    sync_bkps_plt = np.unique(np.concatenate((sync_bkps,np.array(sync_bkps)-1,[0,len(t_sync)-1]))).astype(int)
    
    if len(sync_bkps)==0:
        sync_vec =   t_sync*int(0)  
    else:
        sync_vec = (create_boolean(sync_bkps,len(t_sync),zero_start=f_zero_start)-0.5) * 2
    
    #% Plot full Luminosity signal
    if f_plt:
        plt.figure(figsize=(16,3))
        plt.plot(t_sync[sync_bkps_plt],sync_vec[sync_bkps_plt]/2+0.5)
        plt.title("Luminosity Signal\n%s" % FOLDERstr,fontsize=20)
        NZ,Splot = [t_ecg[0]-5,t_ecg[-1]+5],"Luminosity"
        plt.xlim(NZ)
        plt.xlabel("Time (seconds)",fontsize=14)
        plt.savefig(("%s\\%s" % (FOLDERstr,Splot)),bbox_inches='tight')
    
    sync_peaks=[]
        
    # Create artificial sync signal to match what we made (be 0 during variable bit section)
    sync_ref = (create_boolean([1000,1200,1400,1600,3800,4000,4200],4400,zero_start=1)-0.5) * 2
    for x in np.arange(1600,3600,200): #10-bit
        sync_ref[x:x+100]=0
        
    # Convolve to find best sync point
    sync_corr = np.correlate(sync_vec,sync_ref)
    sync_ind   = np.argmax(sync_corr)
    
    if abs(np.min(sync_corr))>np.max(sync_corr):
        sync_corr *=-1
        sync_vec  *=-1
    
    # Find all sync points
    sync_peaks = find_peaks(sync_corr*(abs(sync_corr)>2800))[0]
    sync_peak_times = t_sync[sync_peaks]
  
    if f_plt:
        # Plot correlation
        plt.figure(figsize=(12,3))
        plt.plot(t_sync[:len(sync_vec)-len(sync_ref)+1],sync_corr,label="Correlation between Luminosity and reference SYNC signal")
        plt.plot([t_sync[0],t_sync[-1]],[2800,2800],'g--',label="Threshold for detection")
        plt.legend(fontsize=12, loc="lower center", bbox_to_anchor=(0.5, -0.6, 0, 0))
        plt.title('Correlation with SYNC signal\n%s'%FOLDERstr,fontsize=16)
        plt.xlabel('Time (seconds)',fontsize=16)
        plt.savefig(("%s\\Luminosity_Correlation" % (FOLDERstr)),bbox_inches='tight')
        
    
    #% Iterate through Peaks?
    enc_list=[]
    sync_print=[]
    
    if len(sync_peaks)==0:
        print("No synchronization found for %s" % (FOLDERstr))
    
    for p in sync_peaks:
        bit_enc = 0
        date_enc = 0       
        hour_enc = 0
        count_enc = 0
         
        # decode the synchronization
        ind_off = 3600+p # start of encoding region
        for x in np.arange(10):
            if np.round(np.median(sync_vec[ind_off-200-200*x:ind_off-100-200*x]))==1:
                bit_enc            += 2**x
                if x<2:  count_enc += 2**x
                elif x<7: hour_enc += 2**(x-2)
                else:     date_enc += 2**(x-7)
             
        print("%s sync_point: %1.3fs" % (FOLDERstr,t_sync[p]))
        print("%s bit_encode: %1.03d, date-8 enc: %d, hour_enc: %d, counter_encode: %d " % (FOLDERstr,bit_enc, date_enc, hour_enc, count_enc))
        enc_list.append(bit_enc)
        
        sync_print.append("%s\t%1.3f\t\t\t\t%1.3f\t%1.3f\t\t\t%1.03d" % (FOLDERstr,t_ecg[-1]-t_ecg[0],t_sync[p],np.round((t_ecg[-1]-t_ecg[0])/5)*5,bit_enc))
        
            
        if f_plt:    
            plt.figure(figsize=(16,3))
            plt.plot(t_sync[sync_bkps_plt],sync_vec[sync_bkps_plt]/2+0.5)
            plt.plot([t_sync[p],t_sync[p]],[-0.1,1.1],'k--',linewidth=2,alpha=0.5)
            plt.plot(t_sync[0:len(sync_ref)]+t_sync[p]-t_sync[0],sync_ref/2+0.5)

            plt.title("Sync Encoding %1.03d at %1.3f\n%s" % (bit_enc,t_sync[p], FOLDERstr),fontsize=20)
            NZ,Splot = [t_sync[p]-2,t_sync[p]+10],"Sync_Start"
            plt.xlim(NZ)
            plt.xlabel("Time (seconds)",fontsize=14)
            plt.savefig(("%s\\%s_%1.03d" % (FOLDERstr,Splot,bit_enc)),bbox_inches='tight')

#%% 7. Reset ECG and ACC data time series
# Just run the cell (once) to set the first sync as the zero point of the time series (
# Change the "sync_index = x" line to select a later sync point

sync_index=0
try:
    t_sync_offset  = t_sync[sync_peaks[sync_index]]
except:
    t_sync_offset = 0
    print("No synchronization found")
t_ecg         -= t_sync_offset
t_acc         -= t_sync_offset      

print('Signal now starts at %1.3f and ends at %1.3f' % (t_ecg[0],t_ecg[-1]))   


#%% 8. Process ACC data 

# Just run this cell

if f_plt:
   
    
    #% Plot Acceleration?
    NZ,Splot = [t_acc[0]-5,t_acc[-1]+5],"Ego_Acc"
    
    plt.figure(figsize=(16,10))
    a0 = plt.subplot(4,1,1)
    viz_ECG_mini(a0, acc, [], t_acc,NZoom=NZ,col='brown', ylab="Accelerometer\nMagnitude",Title="York Accelerometer data\n%s" % FOLDERstr,xlab="")
    
    a1 = plt.subplot(4,1,2)
    viz_ECG_mini(a1, acc_x, [], t_acc,NZoom=NZ,col='brown', ylab="Axis 1",Title="",xlab="")
    a2 = plt.subplot(4,1,3)
    viz_ECG_mini(a2, acc_y, [], t_acc,NZoom=NZ,col='brown', ylab="Axis 2",Title="",xlab="")
    a3 = plt.subplot(4,1,4)
    viz_ECG_mini(a3, acc_z, [], t_acc,NZoom=NZ,col='brown', ylab="Axis 3",Title="",xlab="Time (Seconds)")
    
    plt.savefig(("%s\\%s" % (FOLDERstr,Splot)),bbox_inches='tight')

#%% 9.a Process ECG data (Visualise data)

# Just run this cell

# Plot full ECG 
if f_plt:
    plt.figure(figsize=(16,3))
    NZ=[t_ecg[0],t_ecg[-1]]
    ax1 = plt.subplot(111)
    viz_ECG_mini(ax1,ecg, [],t_ecg, NZoom=NZ, Title = "Full Raw ECG\n%s"% FOLDERstr)
    plt.savefig(("%s\\FullECG" % (FOLDERstr)),bbox_inches='tight')
    
    # Minutes
    for i1 in np.array([0,1,2,5,10,15,20,40,100]):
        if (i1*60)<t_ecg[-1]-t_ecg[0]+30:
            plt.figure(figsize=(16,3))
            ax1 = plt.subplot(111)
            NZ,Splot=[max(60*i1-30,0),60*i1+30]+np.floor(t_ecg[0]),("ECG_%im.png"%i1)
            viz_ECG_mini(ax1,ecg, [],t_ecg, NZoom=NZ, Title = "EgoActive BodySensor\n%s\nRaw ECG" %FOLDERstr) 
            plt.grid(True,axis='x')
            plt.savefig(("%s\\%s" % (FOLDERstr,Splot)),bbox_inches='tight')
     
    # Half Hours
    for i1 in np.arange(0.5,(t_ecg[-1]+30-t_ecg[0])/3600,0.5):
         plt.figure(figsize=(16,3))
         ax1 = plt.subplot(111)
         NZ,Splot=[3600*i1-30,3600*i1+30]+np.floor(t_ecg[0]),("ECG_%ih.png"%i1)
         viz_ECG_mini(ax1,ecg, [],t_ecg, NZoom=NZ, Title = "EgoActive BodySensor\n%s\nRaw ECG" %FOLDERstr) 
         
         plt.grid(True,axis='x')
         plt.savefig(("%s\\%s" % (FOLDERstr,Splot)),bbox_inches='tight')

#%% 9.b Process ECG data (decide whether to flip data)

# Change the flag in this cell if required
# IF YOU DO FLIP, YOU MAY WANT TO RERUN 9.a. TO REGENERATE THE ECG PLOTS

f_ecg_flip = 0  # Set to 1 to flip ECG, set to 0 to not flip it

if f_ecg_flip: 
    ecg=-ecg   # The flip!

#%% 9.c Process ECG data (Process ECG, detect peaks, process peaks)

# Just run this cell

# square wave detection
ind_minmax = (np.abs(ecg)<0.01)*1 + (np.abs(ecg)>3.28)*1
ind_minmaxmed = median_filter(ind_minmax,size=101,mode='mirror') 

# preprocessing
if f_infant:
    fband=15
else:
    fband=0.5 
ecg_nk2 = hf_ecg(ecg,sr=sr_ecg,freqfilt=[fband])

# peak detection
r_nk2   = nk.ecg_findpeaks(ecg_nk2, sampling_rate=sr_ecg, method="neurokit")
r_nk2   = np.asarray(r_nk2['ECG_R_Peaks'])
r_nk2   = np.array(list(set(r_nk2).difference(ind_minmaxmed*np.arange(len(ecg)))))
r_nk2.sort()
r_nk3   = np.copy(r_nk2) # Remove later
r_nk2   = find_local_peaks(r_nk2,ecg)
t_r_nk2 = t_ecg[r_nk2]

#%% 9.c.ii - remove later


# Plot full ECG
if f_plt:

     
    # Minutes
    for i1 in np.array([0,1,2,5,10,15,20,40,100]):
        
         NZ,Splot=[max(60*i1-30,0),60*i1+30]+np.floor(t_ecg[0]),("ECG_peaks_%im.png"%i1)
        
         plt.figure(figsize=(16,16))
         ax1 = plt.subplot(411)
         viz_ECG_mini(ax1,ecg, r_nk3,t_ecg, NZoom=NZ, ylab = "Raw ECG\nRaw Peaks", Title=FOLDERstr)
         plt.grid(True,axis='x')
         
         ax1 = plt.subplot(412)
         viz_ECG_mini(ax1,ecg_nk2, r_nk3,t_ecg, NZoom=NZ, ylab = "Processed ECG\nRaw Peaks", Title="")
         plt.grid(True,axis='x')
         
         ax1 = plt.subplot(413)
         viz_ECG_mini(ax1,ecg_nk2, r_nk2,t_ecg, NZoom=NZ, ylab = "Processed ECG\nCorrected Peaks", Title="")
         plt.grid(True,axis='x')
         
         ax1 = plt.subplot(414)
         viz_ECG_mini(ax1,ecg, r_nk2,t_ecg, NZoom=NZ, ylab = "Raw ECG\nCorrected Peaks", Title="")
         plt.grid(True,axis='x')

      

#%% 9.d Visualise ECG data with peaks 

# Just run this cell
# The same as 9a, but with peaks this time

# Plot full ECG
if f_plt:
    plt.figure(figsize=(16,3))
    NZ=[t_ecg[0],t_ecg[-1]]
    ax1 = plt.subplot(111)
    viz_ECG_mini(ax1,ecg,r_nk2 ,t_ecg, NZoom=NZ, Title = "Full Raw ECG\n%s with peaks"% FOLDERstr)
    plt.savefig(("%s\\FullECGpeaks" % (FOLDERstr)),bbox_inches='tight')
   
    # Minutes
    for i1 in np.array([0,1,2,5,10,15,20,40,100]):
        if (i1*60)<t_ecg[-1]-t_ecg[0]+30:
            plt.figure(figsize=(16,3))
            ax1 = plt.subplot(111)
            NZ,Splot=[max(60*i1-30,0),60*i1+30]+np.floor(t_ecg[0]),("ECG_peaks_%im.png"%i1)
            viz_ECG_mini(ax1,ecg, r_nk2,t_ecg, NZoom=NZ, Title = "EgoActive BodySensor\n%s\nRaw ECG with peaks" %FOLDERstr)
            plt.grid(True,axis='x')
            plt.savefig(("%s\\%s" % (FOLDERstr,Splot)),bbox_inches='tight')
     
    # Half Hours
    for i1 in np.arange(0.5,(t_ecg[-1]+30-t_ecg[0])/3600,0.5):
         plt.figure(figsize=(16,3))
         ax1 = plt.subplot(111)
         NZ,Splot=[3600*i1-30,3600*i1+30]+np.floor(t_ecg[0]),("ECG_peaks_%ih.png"%i1)
         viz_ECG_mini(ax1,ecg, r_nk2,t_ecg, NZoom=NZ, Title = "EgoActive BodySensor\n%s\nRaw ECG with peaks" %FOLDERstr)
         
         plt.grid(True,axis='x')
         plt.savefig(("%s\\%s" % (FOLDERstr,Splot)),bbox_inches='tight')

#%% 10. Convert ECG data into heart rate

# Just run this cell

hr_nk2=np.array([60/((r_nk2[x+1]-r_nk2[x])/(1/np.mean(np.diff(t_ecg[r_nk2[x]:r_nk2[x+1]])))) for x in np.arange(len(r_nk2)-1)]) # if sr gets wild

t_hr_nk2 = t_r_nk2[0:-1]+np.diff(t_r_nk2)/2


#%% 11. Process heart rate

# Just run this cell

hr_nk2_filt, qual_hr = hrSQI(hr_nk2,sr=sr_ecg,Nfilt=31,good_fact=1.3)
hr_nk2_filt2         = hrCleanBeats(hr_nk2_filt, thresh1=15,thresh2=25)

if f_plt:
    fig, ax = plt.subplots(figsize=(16,3))
    
    NZ, Ssav = [t_hr_nk2[0],t_hr_nk2[-1]], "HR"

    ax.plot(t_hr_nk2,hr_nk2_filt2, color='red',label='Processed HR',linewidth=0.5,marker='.')
    ax.plot(t_hr_nk2,hr_nk2,label='Raw HR', color='green',linewidth=0.5,alpha=0.65)
    
    ax.set_xlim(NZ)
    
    ax.set_title('HR Processing, EgoActiveBodySensor\n%s' % FOLDERstr,fontsize=20)
    ax.set_ylim([0, max([max(hr_nk2_filt2),200])])
    ax.set_ylabel('Heart Rate',fontsize=14)
    ax.set_xlabel('Time (seconds)',fontsize=14)
    plt.grid(True)
    plt.legend(loc = "upper right")
    plt.savefig(("%s\\%s" % (FOLDERstr,Ssav)),bbox_inches='tight')
    
    
    #% Full HR Plot loop (half hour) 
    if t_ecg[-1]-t_ecg[0]>1800: #only plot if signal is longer than half an hour
        for i1 in np.arange(np.floor(t_ecg[0]/1800),t_ecg[-1]/1800,1):
            plt.figure(figsize=(16,3))
            ax1 = plt.subplot(111)
            NZ,Splot=[1800*i1,1800*i1+1800],("HR_zoom_%1.1fh-%1.1fh.png"%(i1/2,(i1+1)/2))
            try:    ind_sta = np.where(t_hr_nk2<=NZ[0])[0][-1]
            except: ind_sta=0
            try:    ind_end = np.where(t_hr_nk2>=NZ[1])[0][0]
            except: ind_end=len(t_hr_nk2)
            ax1.plot(t_hr_nk2[ind_sta:ind_end],hr_nk2_filt2[ind_sta:ind_end], color='red',label='Processed HR',linewidth=2)
            ax1.plot(t_hr_nk2[ind_sta:ind_end],hr_nk2[      ind_sta:ind_end], label='Raw HR', color='green',alpha=0.65)
            ax1.set_xlim(NZ)
            ax1.set_title('HR Processing, EgoActiveBodySensor\n%s' % FOLDERstr,fontsize=20)
            ax1.set_xticks(np.arange(NZ[0],NZ[1]+1,20))
            ax1.set_xticklabels(ax1.get_xticks(), rotation = 90)
            ax1.set_ylim([0, max([max(hr_nk2_filt2),200])])
            # ax.set_ylim([149, 152])
            ax1.set_ylabel('Heart Rate',fontsize=14)
            ax1.set_xlabel('Time (seconds)',fontsize=14)
            plt.grid(True)
            plt.legend(loc = "upper right")
            plt.savefig(("%s\\%s" % (FOLDERstr,Splot)),bbox_inches='tight')



#%% 12. Run signal quality metrics on processed heart rate

# Just run this cell (can alter quality_threshold in section 3)

# This section is more user-dependent on the level of noise allowed in the data. 
# If a strictly correct heart rate is required (e.g. HRV analysis), set the tolerance to 1.
# If the roughly correct shape is acceptable, a lower threshold (e.g. 0.7) can suffice. 

hr_sqi_vec=sqi_clean(qual_hr,hr_nk2,hr_nk2_filt,t_hr_nk2,sqi_thresh=quality_threshold,Shr = FOLDERstr)
        
try:     ind_hr_sta = np.where(hr_sqi_vec==1)[0][0]
except:  ind_hr_sta = 0

t_hr_sta=t_hr_nk2[ind_hr_sta]

try:     ind_hr_end = np.where(hr_sqi_vec==1)[0][-1]
except:  ind_hr_end = len(hr_sqi_vec)-1

t_hr_end=t_hr_nk2[ind_hr_end]


if f_plt:
    
    plt.figure(figsize=(16,3))
    Splot='FullHR with SQI'
    ax1 = plt.subplot(111)
    ax1.plot(t_hr_nk2,hr_nk2_filt2, color='red',label='Filtered HR',linewidth=2)
    ax1.plot(t_hr_nk2,hr_nk2, label='HR Derived from Peaks', color='green',alpha=0.65)
    
    ax1.set_xlim([t_hr_nk2[0]-5,t_hr_nk2[-1]+5])
    ax1.set_title('Full HR with SQI metric, EgoActiveBodySensor\n%s' % FOLDERstr,fontsize=20)
    ax1.set_ylim([0, max([max(hr_nk2_filt2),200])])

    ax1.set_ylabel('HR',color='maroon',fontsize=14)
    ax1.set_xlabel('Time (seconds)',fontsize=14)
    plt.grid(True)
    
    ax1.tick_params(axis ='y', labelcolor = 'maroon')
 
    # Adding Twin Axes to plot using dataset_2
    ax2 = ax1.twinx()
    color = 'darkmagenta'
    ax2.set_ylabel('SQI', color = color,fontsize=14,rotation=270)
    ax2.plot(t_hr_nk2,hr_sqi_vec, label='SQI', color='darkmagenta',alpha=0.65, linewidth=2)
    ax2.tick_params(axis ='y',  labelcolor = color)
    ax2.set_yticks([0,1])
    plt.savefig(("%s\\%s" % (FOLDERstr,Splot)),bbox_inches='tight')
    
    # % Full HR Plot loop (half hour)
    t_loop = 300 # time width in seconds for each plot, recommended 1800 (a half hour) or lower
    t_ticks=   20 # spacing of the ticks, 20s is default
    
    
    for i1 in np.arange(np.floor(t_ecg[0]/t_loop),t_ecg[-1]/t_loop,1):
        
        NZ,Splot=[t_loop*i1,t_loop*i1+t_loop],("HR_SQI_zoom_%1.1fh-%1.1fh.png"%(i1/(3600/t_loop),(i1+1)/(3600/t_loop)))
        try:    ind_sta = np.where(t_hr_nk2<=NZ[0])[0][-1]
        except: ind_sta=0
        try:    ind_end = np.where(t_hr_nk2>=NZ[1])[0][0]
        except: ind_end=len(t_hr_nk2)
        
        plt.figure(figsize=(16,3))
        ax1 = plt.subplot(111)
        ax1.plot(t_hr_nk2[ind_sta:ind_end],hr_nk2_filt2[ind_sta:ind_end], color='red',label='Filtered HR',linewidth=2)
        ax1.plot(t_hr_nk2[ind_sta:ind_end],hr_nk2[      ind_sta:ind_end], label='HR Derived from Peaks', color='green',alpha=0.65)
        
        ax1.set_xlim(NZ)
        ax1.set_title('HR with SQI, EgoActiveBodySensor\n%s' % FOLDERstr,fontsize=20)
        ax1.set_ylim([0, max([max(hr_nk2_filt2),200])])
        ax1.set_xticks(np.arange(NZ[0],NZ[1]+1,t_ticks))
        ax1.set_xticklabels([])

        ax1.set_ylabel('HR',color='maroon',fontsize=14)
        ax1.set_xlabel('Time (seconds)',fontsize=14)
        plt.grid(True)
        
        ax1.tick_params(axis ='y', labelcolor = 'maroon')
     
        # Adding Twin Axes to plot using dataset_2
        ax2 = ax1.twinx()
        color = 'darkmagenta'
        ax2.set_ylabel('SQI', color = color,fontsize=14,rotation=270)
        ax2.plot(t_hr_nk2[ind_sta:ind_end],hr_sqi_vec[ind_sta:ind_end], label='SQI', color='darkmagenta',alpha=0.65, linewidth=2)
        ax2.tick_params(axis ='y',  labelcolor = color)
        ax2.set_yticks([0,1])
        plt.savefig(("%s\\%s" % (FOLDERstr,Splot)),bbox_inches='tight')
        
        
#%% 13. Save the output heart rate and accelerometer data  
      
#% Peak convertion for matlab indexing
r_nk2_mat = [x+1 for x in r_nk2 if x < len(ecg)]

if f_save: 
    scipy.io.savemat(FOLDERstr+"EgoAccHR.mat",
                      {'Subject_Folder'    :FOLDERstr,      # A unique subject identifier, I believe something akin to the filename
                       'ecg'               :ecg,            # the ecg signal
                       'ecg_time'          :t_ecg,          # the corresponding time index for the ecg
                       'ecg_peaks'         :r_nk2_mat,      # the index of peaks in the ECG
                       'acc'               :acc,            # the accelerometer magnitude data
                       'acc_time'          :t_acc,          # the corresponding time index for the accelerometer data
                       'heart_rate'        :hr_nk2_filt2,   # the heart rate signal (post processing)
                       'heart_rate_sqi_vec':hr_sqi_vec,     # the updated SQI measure, 0 when the signal is good, 1 when it is bad
                       'heart_rate_time'   :t_hr_nk2})      # the corresponding time index for the heart rate data (and the sqi and sqi_vec measures)
    print("Saving in "+FOLDERstr) 
else: print("Save flag not active, would have saved in "+FOLDERstr)

# Other options you may want (just uncomment, copy and past into the dictionary above)
    
# 'acc1'               :accX, # the first axis of acceleration data
# 'acc2'               :accY, # the second axis of acceleration data
# 'acc3'               :accZ, # the third axis of acceleration data
# 'ecg_processed'      :ecg_nk2, # The processed ECG
# 'heart_rate_raw'     :hr_nk2,  # The raw heart rate
# 'quality_threshold'  :quality_threshold, # The selected quality threshold
# 'ecg_flip_flag'      :f_ecg_flip, # whether the ecg was flipped
# 'infant_flag'        :f_infant, # whether you processed the data as infant data
# 't_offset'           :t_sync_offset, # the time offset caused by the synchronization signal 