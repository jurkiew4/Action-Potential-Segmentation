import tkinter as tk
import tkinter.messagebox
from tkinter import messagebox as MessageBox
import csv
import os
import time
import numpy as np
import scipy
from scipy import signal
import h5py
import statistics as stat

def main():
    path=input("Please enter the full path to the H5 file folder:")
    os.chdir(path)

    top=tk.Tk(className="Select files to segment:")
    top.geometry("500x200")

    arr=os.listdir('.')
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    n=len(arr)
    scroll=tk.Scrollbar(top)
    scroll.pack(side=tk.RIGHT, fill = tk.Y)
    lb=tk.Listbox(top, yscrollcommand=scroll.set, selectmode='multiple')
    for x in range(0, n):
        lb.insert(x, arr[x])
        lb.pack(side = tk.TOP, fill=tk.BOTH, expand=True)

    scroll.config(command=lb.yview)

    

    def buttonCallBack():
        a=lb.curselection()
        for x in a:
           segmenter(path+'\\'+arr[x])
        MessageBox.showinfo("Segmentation complete!")
        
    B=tk.Button(top, text="Segment selected files", command=buttonCallBack)
    B.pack()


    top.mainloop()

def segmenter(fileName):
   
    start_time=time.time();

    #Initialization of Data Arrays and parameters
    #t=np.linspace(0,10,200000);
    y1=[];
    y2=[];
    k=200;
    #Loads data from H5 file
    a=h5py.File(fileName, 'r');
    y=a[u'Data'][u'Recording_0'][u'AnalogStream'][u'Stream_0'][u'ChannelData'][range(0,287), :]
    del(a)
    csvFile=open('testf.csv', 'w', newline='')
    filewrite=csv.writer(csvFile, delimiter=',')
    #FP decision step
    #   -if FP range is greater than 1 mV, acceptable
    #   -else unacceptable
    y1[:]=[elem[:200000] for elem in y];
    
    for x in y1:
        if max(abs(np.asarray(x)))>=1000:
            filewrite.writerow(x)
    csvFile.close()    
    y1=None;
    print ("FP time = "+str(time.time()-start_time))

    #Initialization of AP filters
    #   -b,c for noise removal
    #   -b1, c1 for drift removal
    b, c = signal.butter(10, 225./10000., 'lowpass', analog=False)
    b1, c1=signal.butter(3, .5/10000., 'highpass', analog=False)

    #AP decision algorithm
    y2[:]=[elem[1400000:1600000].tolist() for elem in y];
    csvFile=open('testa.csv', 'w', newline='')
    filewrite=csv.writer(csvFile, delimiter=',')
    start_time=time.time()

    for x in y2:
        d=0;    #Discriminator variable for detecting non-monotonicity
    
        #start_time=time.time()
        
        R=signal.filtfilt(b, c, x);
        R=signal.filtfilt(b1,c1, R);

    
        z=np.gradient(R);
        z=z/max(z)
    
        if min(z)<-.4:     # tests for absolute significant negative derivative value
            d=1;
            continue
        L=spikedec(z);
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
                filewrite.writerow(R[:L[i]-k])
                break
            elif m<-.05 and i+1<=5:
                d=1
                break
            else:
                pass
            
    
        if d==0:
            filewrite.writerow(R[:L[-1]])
   
    #print("time for 1 ap="+str(time.time()-start_time))
    csvFile.close()
    y2=None;
    print ("AP time = "+str(time.time()-start_time))

def spikedec(x):
#Function to locate depolarization spike in AP data trace.    
        
    lp=[];
    x=np.gradient(x)
    x=x/max(x)
    #Normalizes derivative to have max of 1.
    sigma=stat.stdev(x);
    
    for i in range(1000):
        l=np.argmax(x);
        #Spike must have magnitude greater than .15 and greater than 3 times 
        #the standard deviation of the derivative.
        if max(x)<.15 or max(x)<3*sigma:
            break
        lpp=np.argmax(x[range(max(1,l-1000), min(l+1000,len(x)))]);
        
        x[max(1,l-1000):min(l+1000,len(x))]=np.zeros(min(l+1000,len(x))-max(1,l-1000));
        lp.extend([lpp+max(l-1000,1)]);
    
    lp.sort()
    #returns list of depolarization locations in ascending timestamp order.
    return lp

main()
