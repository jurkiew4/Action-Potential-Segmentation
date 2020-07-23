def spikedec(x):
#Function to locate depolarization spike in AP data trace.    
    import numpy as np
    import statistics as stat
        
    lp=[];
    x=np.gradient(x)/max(np.gradient(x));
    #Normalizes derivative to have max of 1.
    sigma=stat.stdev(x);
    
    for i in range(1000):
        l=np.argmax(x);
        #Spike must have magnitude greater than .15 and greater than 3 times 
        #the standard deviation of the derivative.
        if max(x)<.15 or max(x)<3*sigma:
            break
        lpp=np.argmax(x[range(max(1,l-1000), min(l+1000,len(x)))]);
        for j in range(max(1,l-1000), min(l+1000,len(x))):
            x[j]=0;
        lp.extend([lpp+max(l-1000,1)]);
    
    lp.sort()
    #returns list of depolarization locations in ascending timestamp order.
    return lp