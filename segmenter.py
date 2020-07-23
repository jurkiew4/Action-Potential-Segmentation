import numpy as np
import scipy
from scipy import signal
import h5py
import spikedec as sd

#Initialization of Data Arrays and parameters
t=np.linspace(0,10,200000);
y1=[];
y2=[];
GF=[];
BF=[];
GA=[];
BA=[];
AP=[];
FP=[];
k=200;
#Loads data from H5 file
a=h5py.File('20181204_2-30_PM_No_compound_M112818_01_AP_All_mwd.h5', 'r');
y=a[u'Data'][u'Recording_0'][u'AnalogStream'][u'Stream_0'][u'ChannelData'][range(0,287), :].tolist();

#FP decision step
#   -if FP range is greater than 1 mV, acceptable
#   -else unacceptable
y1[:]=(elem[:200000] for elem in y);
for x in y1:
    if max(abs(np.asarray(x)))<1000:
        BF.extend([y1.index(x)+1]);
        
    if max(abs(np.asarray(x)))>=1000:
        GF.extend([y1.index(x)+1]);
        FP.extend([x]);
        
y1=None;

#Initialization of AP filters
#   -b,c for noise removal
#   -b1, c1 for drift removal
b, c = signal.butter(10, 225./10000., 'lowpass', analog=False)
b1, c1=signal.butter(3, .5/10000., 'highpass', analog=False)

#AP decision algorithm
y2[:]=(elem[1400000:1600000] for elem in y);

for x in y2:
    d=0;    #Discriminator variable for detecting non-monotonicity
    R=signal.filtfilt(b, c, x);
    R=signal.filtfilt(b1,c1, R);
    
    z=np.gradient(R);
    z=z/max(z)
    
    if min(z)<-.4:     # tests for absolute significant negative derivative value
        d=1;
    L=sd.spikedec(R);
    #Scans neighborhood of local maxima in derivative for negative spikes 
    #   -if no negative spike, pulse is acceptable
    #   -if negative spike, pulse is unacceptable.
    
    #Signal is truncated at location of first unacceptable pulse
    #provided the AP trace contains a minimum of 5 beats before degrdation
    #If 4 or fewer acceptable beats, trace is discarded.
    for i in range(len(L)):
        p=z[range(max(L[i]-k, 0), min(L[i]+k,len(R)))];
        m=min(p);
        if m<-.05 and i+1>=6:
            d=1;
            AP.extend([R[range(L[i])]]);
            GA.extend([y2.index(x)+1]);
            break
        elif m<-.05 and i+1<=5:
            d=1;
            BA.extend([y2.index(x)+1]);
            break
    
    if d==0:
        GA.extend([y2.index(x)+1]);
        AP.extend([R]);  

y2=None;

APD30=[];
APD80=[];
DD=[];

#AP biomarker extraction steps
#Each pulse is scaled and shifted to have max val. 1, min val. 0 and max val
#occuring at t=0. APD at n% is measured as time until rescaled pulse reaches 1-(n/100)
#BCL is measured as time between successive maxima in derivative.
for x in AP:
    A=[];
    B=[];
    C=[];    
    L=sd.spikedec(x);
    t=np.linspace(0, len(x)/20000., num=len(x));
    
    for j in range(len(L)-1):
        V=x[range(L[j], int(np.floor((L[j]+L[j+1])/2.)))];
        n=np.argmax(V);
        V=V[range(n,len(V)-1)];
        l=min(V);
        m=np.argmin(V);
        V=V[range(min(len(V),m+1000))];
        T=t[range(1,len(V))]
        V=(V-l)/max(V);
        Vp=abs(V-.7);
        mi=np.argmin(Vp);
        A.extend([T[mi]])
        Vp=abs(V-.2);
        mi=np.argmin(Vp);
        B.extend([T[mi]])
        C.extend([t[L[j+1]]-t[L[j]]]);
    
    APD30.extend([A]);
    APD80.extend([B]);
    DD.extend([C]);