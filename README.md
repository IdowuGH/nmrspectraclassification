```python
# Metabolic Profile of Human Pituitary Tumours. A data driven approach to classifying human pituitary tumours
# Sample data involved the 1D 1H-NMR spectra for n-samples of Human Pituitary Tumours. n = 134

```


```python
#Packages for the analysis:

from scipy.signal import find_peaks
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import scipy.signal as ssig
import scipy.signal as sig # import the complete module of scipy.signal with alias sig
import scipy.stats as ss
from scipy.stats import zscore
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
%matplotlib inline
from metabolabpy.nmr import nmrDataSet
from scipy.stats import ranksums
from scipy.spatial.distance import euclidean 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
import seaborn as sbn
import sklearn.metrics as skm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
nd = nmrDataSet.NmrDataSet()
dc = nd.load('karavitaki_allSpectra7_tsaScaling.mlpy')
```


```python
nd.nmrdat[0][1].title
```


```python
# Extraction of nmr spectra from the nmr spectroscopy data signal for the n-sample = 134
```


```python
rol = [] # initializes an empty list
nspc = len(nd.nmrdat[0])# Calculate the number of Spectra of in The nd.nmrdat[0]
for i in range(134):# Start a loop that iterates over the of value from 0 to nspc
    rol.append(nd.nmrdat[0][i].spc[0].real)#Accesses the 'i'th spectrum and retrieves the real part of the first data point and append to the rol list
        
rol = np.array(rol)
#rol # the array values of all the 150 samples
```


```python
# Random assessment of individual pituitary Sample Spectrum. n-Sample = 13(pituitary tumour number 13) 

nspc = len(nd.nmrdat[0])# Calculates the number of Spectra in the nd.nmrdat[0] object and assigns it 
for samp in range(nspc): # Starts a loop that iterates over the range of values from 0 to nspc -1'. 
    samp = nd.nmrdat[0][13].spc[0].real # assigned the real part of the first data point of the spectrum at index 13 
np.info(samp) # Call the np.info() function to display information about the variable Samp
plt.plot(samp)# Plot the vaues in 'samp'
```

    class:  ndarray
    shape:  (131072,)
    strides:  (16,)
    itemsize:  8
    aligned:  True
    contiguous:  False
    fortran:  False
    data pointer: 0x1be0fcfe060
    byteorder:  little
    byteswap:  False
    type: float64
    




    [<matplotlib.lines.Line2D at 0x1be0fa21fa0>]




    
![png](output_5_2.png)
    



```python
# Extrected and stacked spectra for the 134 pituitary tumour samples

rold = []# initializes an empty list ' rold'
np_ts = len(nd.nmrdat[0][i].spc[0])#Calculate the lenght of the nd.nmrdat[0][i].spc and assigns it to the variable np_ts
ns_pc = len(nd.nmrdat[0])# Calculate the lenght of the nd.nmrdat[0] and assign it to the variable ns_pc
rold = np.array([[]]) # initializes a 2D Numpypy rold with array rold with an empty array 
rold = np.resize(rold, (ns_pc, np_ts))# Resizes the rold array to have the dimension(ns_pc, np_ts)
for i in range (134):# start a loop that iterate  over the range of value from 0 to ns_pc -1
    rold[i] = (nd.nmrdat[0][i].spc[0])# assigns the data from the nd.nmrdat[0][i].spc[0] array to the i-th row of the rold array - it populate the rold array with the spectra data
plt.plot(rold)
```

    C:\Users\bosmi\AppData\Local\Temp\ipykernel_219768\1723261217.py:9: ComplexWarning: Casting complex values to real discards the imaginary part
      rold[i] = (nd.nmrdat[0][i].spc[0])# assigns the data from the nd.nmrdat[0][i].spc[0] array to the i-th row of the rold array - it populate the rold array with the spectra data
    




    [<matplotlib.lines.Line2D at 0x2df400d39a0>,
     <matplotlib.lines.Line2D at 0x2df400d3a00>,
     <matplotlib.lines.Line2D at 0x2df400d3a30>,
     <matplotlib.lines.Line2D at 0x2df400d3b20>,
     <matplotlib.lines.Line2D at 0x2df400d3c10>,
     <matplotlib.lines.Line2D at 0x2df400d3d00>,
     <matplotlib.lines.Line2D at 0x2df400d3df0>,
     <matplotlib.lines.Line2D at 0x2df400d3ee0>,
     <matplotlib.lines.Line2D at 0x2df400d3fd0>,
     <matplotlib.lines.Line2D at 0x2df400e2100>,
     <matplotlib.lines.Line2D at 0x2df400d39d0>,
     <matplotlib.lines.Line2D at 0x2df400e21f0>,
     <matplotlib.lines.Line2D at 0x2df400e22e0>,
     <matplotlib.lines.Line2D at 0x2df400e2490>,
     <matplotlib.lines.Line2D at 0x2df400e2580>,
     <matplotlib.lines.Line2D at 0x2df400e2670>,
     <matplotlib.lines.Line2D at 0x2df400e2760>,
     <matplotlib.lines.Line2D at 0x2df400e2850>,
     <matplotlib.lines.Line2D at 0x2df400e2940>,
     <matplotlib.lines.Line2D at 0x2df400e2a30>,
     <matplotlib.lines.Line2D at 0x2df400e2b20>,
     <matplotlib.lines.Line2D at 0x2df400e2c10>,
     <matplotlib.lines.Line2D at 0x2df400e2d00>,
     <matplotlib.lines.Line2D at 0x2df400e2df0>,
     <matplotlib.lines.Line2D at 0x2df400e2ee0>,
     <matplotlib.lines.Line2D at 0x2df400e2fd0>,
     <matplotlib.lines.Line2D at 0x2df400ea100>,
     <matplotlib.lines.Line2D at 0x2df400ea1f0>,
     <matplotlib.lines.Line2D at 0x2df400ea2e0>,
     <matplotlib.lines.Line2D at 0x2df400ea3d0>,
     <matplotlib.lines.Line2D at 0x2df400ea4f0>,
     <matplotlib.lines.Line2D at 0x2df400ea5e0>,
     <matplotlib.lines.Line2D at 0x2df400ea6d0>,
     <matplotlib.lines.Line2D at 0x2df400ea7c0>,
     <matplotlib.lines.Line2D at 0x2df400ea8b0>,
     <matplotlib.lines.Line2D at 0x2df400ea9a0>,
     <matplotlib.lines.Line2D at 0x2df400eaa90>,
     <matplotlib.lines.Line2D at 0x2df400eab80>,
     <matplotlib.lines.Line2D at 0x2df400eac70>,
     <matplotlib.lines.Line2D at 0x2df400ead60>,
     <matplotlib.lines.Line2D at 0x2df400eae50>,
     <matplotlib.lines.Line2D at 0x2df400eaf40>,
     <matplotlib.lines.Line2D at 0x2df400f3070>,
     <matplotlib.lines.Line2D at 0x2df400f3160>,
     <matplotlib.lines.Line2D at 0x2df400f3250>,
     <matplotlib.lines.Line2D at 0x2df400f3340>,
     <matplotlib.lines.Line2D at 0x2df400f3430>,
     <matplotlib.lines.Line2D at 0x2df400f3520>,
     <matplotlib.lines.Line2D at 0x2df400f3610>,
     <matplotlib.lines.Line2D at 0x2df400f3700>,
     <matplotlib.lines.Line2D at 0x2df400f37f0>,
     <matplotlib.lines.Line2D at 0x2df400f38e0>,
     <matplotlib.lines.Line2D at 0x2df400f39d0>,
     <matplotlib.lines.Line2D at 0x2df400f3ac0>,
     <matplotlib.lines.Line2D at 0x2df400f3bb0>,
     <matplotlib.lines.Line2D at 0x2df400f3ca0>,
     <matplotlib.lines.Line2D at 0x2df400f3d90>,
     <matplotlib.lines.Line2D at 0x2df400f3e80>,
     <matplotlib.lines.Line2D at 0x2df400f3f70>,
     <matplotlib.lines.Line2D at 0x2df400fa0a0>,
     <matplotlib.lines.Line2D at 0x2df400fa190>,
     <matplotlib.lines.Line2D at 0x2df400fa280>,
     <matplotlib.lines.Line2D at 0x2df400fa370>,
     <matplotlib.lines.Line2D at 0x2df400fa460>,
     <matplotlib.lines.Line2D at 0x2df400fa550>,
     <matplotlib.lines.Line2D at 0x2df400fa640>,
     <matplotlib.lines.Line2D at 0x2df400fa730>,
     <matplotlib.lines.Line2D at 0x2df400fa820>,
     <matplotlib.lines.Line2D at 0x2df400fa910>,
     <matplotlib.lines.Line2D at 0x2df400faa00>,
     <matplotlib.lines.Line2D at 0x2df400faaf0>,
     <matplotlib.lines.Line2D at 0x2df400fabe0>,
     <matplotlib.lines.Line2D at 0x2df400facd0>,
     <matplotlib.lines.Line2D at 0x2df400fadc0>,
     <matplotlib.lines.Line2D at 0x2df400faeb0>,
     <matplotlib.lines.Line2D at 0x2df400fafa0>,
     <matplotlib.lines.Line2D at 0x2df401010d0>,
     <matplotlib.lines.Line2D at 0x2df401011c0>,
     <matplotlib.lines.Line2D at 0x2df401012b0>,
     <matplotlib.lines.Line2D at 0x2df401013a0>,
     <matplotlib.lines.Line2D at 0x2df40101490>,
     <matplotlib.lines.Line2D at 0x2df40101580>,
     <matplotlib.lines.Line2D at 0x2df40101670>,
     <matplotlib.lines.Line2D at 0x2df40101760>,
     <matplotlib.lines.Line2D at 0x2df40101850>,
     <matplotlib.lines.Line2D at 0x2df40101940>,
     <matplotlib.lines.Line2D at 0x2df40101a30>,
     <matplotlib.lines.Line2D at 0x2df40101b20>,
     <matplotlib.lines.Line2D at 0x2df40101c10>,
     <matplotlib.lines.Line2D at 0x2df40101d00>,
     <matplotlib.lines.Line2D at 0x2df40101df0>,
     <matplotlib.lines.Line2D at 0x2df40101ee0>,
     <matplotlib.lines.Line2D at 0x2df40101fd0>,
     <matplotlib.lines.Line2D at 0x2df4010a100>,
     <matplotlib.lines.Line2D at 0x2df4010a1f0>,
     <matplotlib.lines.Line2D at 0x2df4010a2e0>,
     <matplotlib.lines.Line2D at 0x2df4010a3d0>,
     <matplotlib.lines.Line2D at 0x2df4010a4c0>,
     <matplotlib.lines.Line2D at 0x2df4010a5b0>,
     <matplotlib.lines.Line2D at 0x2df4010a6a0>,
     <matplotlib.lines.Line2D at 0x2df4010a790>,
     <matplotlib.lines.Line2D at 0x2df4010a880>,
     <matplotlib.lines.Line2D at 0x2df4010a970>,
     <matplotlib.lines.Line2D at 0x2df4010aa60>,
     <matplotlib.lines.Line2D at 0x2df4010ab50>,
     <matplotlib.lines.Line2D at 0x2df4010ac40>,
     <matplotlib.lines.Line2D at 0x2df40038fd0>,
     <matplotlib.lines.Line2D at 0x2df4003e220>,
     <matplotlib.lines.Line2D at 0x2df4003e310>,
     <matplotlib.lines.Line2D at 0x2df4003e040>,
     <matplotlib.lines.Line2D at 0x2df40029580>,
     <matplotlib.lines.Line2D at 0x2df4010af10>,
     <matplotlib.lines.Line2D at 0x2df40111040>,
     <matplotlib.lines.Line2D at 0x2df40111130>,
     <matplotlib.lines.Line2D at 0x2df40111220>,
     <matplotlib.lines.Line2D at 0x2df40111310>,
     <matplotlib.lines.Line2D at 0x2df40111400>,
     <matplotlib.lines.Line2D at 0x2df401114f0>,
     <matplotlib.lines.Line2D at 0x2df401115e0>,
     <matplotlib.lines.Line2D at 0x2df401116d0>,
     <matplotlib.lines.Line2D at 0x2df401117c0>,
     <matplotlib.lines.Line2D at 0x2df401118b0>,
     <matplotlib.lines.Line2D at 0x2df401119a0>,
     <matplotlib.lines.Line2D at 0x2df40111a90>,
     <matplotlib.lines.Line2D at 0x2df40111b80>,
     <matplotlib.lines.Line2D at 0x2df40111c70>,
     <matplotlib.lines.Line2D at 0x2df40111d60>,
     <matplotlib.lines.Line2D at 0x2df40111e50>,
     <matplotlib.lines.Line2D at 0x2df40111f40>,
     <matplotlib.lines.Line2D at 0x2df40118070>,
     <matplotlib.lines.Line2D at 0x2df40118160>,
     <matplotlib.lines.Line2D at 0x2df40118250>,
     <matplotlib.lines.Line2D at 0x2df40118340>,
     <matplotlib.lines.Line2D at 0x2df40118430>,
     <matplotlib.lines.Line2D at 0x2df40118520>,
     <matplotlib.lines.Line2D at 0x2df40118610>,
     <matplotlib.lines.Line2D at 0x2df40118700>,
     <matplotlib.lines.Line2D at 0x2df401187f0>,
     <matplotlib.lines.Line2D at 0x2df401188e0>,
     <matplotlib.lines.Line2D at 0x2df401189d0>,
     <matplotlib.lines.Line2D at 0x2df40118ac0>,
     <matplotlib.lines.Line2D at 0x2df40118bb0>,
     <matplotlib.lines.Line2D at 0x2df40118ca0>,
     <matplotlib.lines.Line2D at 0x2df40118d90>,
     <matplotlib.lines.Line2D at 0x2df40118e80>,
     <matplotlib.lines.Line2D at 0x2df40118f70>,
     <matplotlib.lines.Line2D at 0x2df401210a0>,
     <matplotlib.lines.Line2D at 0x2df40121190>,
     <matplotlib.lines.Line2D at 0x2df40121280>,
     <matplotlib.lines.Line2D at 0x2df40121370>,
     <matplotlib.lines.Line2D at 0x2df40121460>,
     <matplotlib.lines.Line2D at 0x2df40121550>,
     <matplotlib.lines.Line2D at 0x2df40121640>,
     <matplotlib.lines.Line2D at 0x2df40121730>,
     <matplotlib.lines.Line2D at 0x2df40121820>,
     <matplotlib.lines.Line2D at 0x2df40121910>,
     <matplotlib.lines.Line2D at 0x2df40121a00>,
     <matplotlib.lines.Line2D at 0x2df40121af0>,
     <matplotlib.lines.Line2D at 0x2df40121be0>,
     <matplotlib.lines.Line2D at 0x2df40121cd0>,
     <matplotlib.lines.Line2D at 0x2df40121dc0>,
     <matplotlib.lines.Line2D at 0x2df40121eb0>,
     <matplotlib.lines.Line2D at 0x2df40121fa0>,
     <matplotlib.lines.Line2D at 0x2df401280d0>,
     <matplotlib.lines.Line2D at 0x2df401281c0>,
     <matplotlib.lines.Line2D at 0x2df401282b0>,
     <matplotlib.lines.Line2D at 0x2df401283a0>,
     <matplotlib.lines.Line2D at 0x2df40128490>,
     <matplotlib.lines.Line2D at 0x2df40128580>,
     <matplotlib.lines.Line2D at 0x2df40128670>,
     <matplotlib.lines.Line2D at 0x2df40128760>,
     <matplotlib.lines.Line2D at 0x2df40128850>,
     <matplotlib.lines.Line2D at 0x2df40128940>,
     <matplotlib.lines.Line2D at 0x2df40128a30>,
     <matplotlib.lines.Line2D at 0x2df40128b20>,
     <matplotlib.lines.Line2D at 0x2df40128c10>,
     <matplotlib.lines.Line2D at 0x2df40128d00>,
     <matplotlib.lines.Line2D at 0x2df40128df0>,
     <matplotlib.lines.Line2D at 0x2df40128ee0>,
     <matplotlib.lines.Line2D at 0x2df40128fd0>,
     <matplotlib.lines.Line2D at 0x2df42ad0100>,
     <matplotlib.lines.Line2D at 0x2df42ad01f0>,
     <matplotlib.lines.Line2D at 0x2df42ad02e0>,
     <matplotlib.lines.Line2D at 0x2df42ad03d0>,
     <matplotlib.lines.Line2D at 0x2df42ad04c0>,
     <matplotlib.lines.Line2D at 0x2df42ad05b0>,
     <matplotlib.lines.Line2D at 0x2df42ad06a0>,
     <matplotlib.lines.Line2D at 0x2df42ad0790>,
     <matplotlib.lines.Line2D at 0x2df42ad0880>,
     <matplotlib.lines.Line2D at 0x2df42ad0970>,
     <matplotlib.lines.Line2D at 0x2df42ad0a60>,
     <matplotlib.lines.Line2D at 0x2df42ad0b50>,
     <matplotlib.lines.Line2D at 0x2df42ad0c40>,
     <matplotlib.lines.Line2D at 0x2df42ad0d30>,
     <matplotlib.lines.Line2D at 0x2df42ad0e20>,
     <matplotlib.lines.Line2D at 0x2df42ad0f10>,
     <matplotlib.lines.Line2D at 0x2df42ad9040>,
     <matplotlib.lines.Line2D at 0x2df42ad9130>,
     <matplotlib.lines.Line2D at 0x2df42ad9220>,
     <matplotlib.lines.Line2D at 0x2df42ad9310>,
     <matplotlib.lines.Line2D at 0x2df42ad9400>,
     <matplotlib.lines.Line2D at 0x2df42ad94f0>,
     <matplotlib.lines.Line2D at 0x2df42ad95e0>,
     <matplotlib.lines.Line2D at 0x2df42ad96d0>,
     <matplotlib.lines.Line2D at 0x2df42ad97c0>,
     <matplotlib.lines.Line2D at 0x2df42ad98b0>,
     <matplotlib.lines.Line2D at 0x2df42ad99a0>,
     <matplotlib.lines.Line2D at 0x2df42ad9a90>,
     <matplotlib.lines.Line2D at 0x2df42ad9b80>,
     <matplotlib.lines.Line2D at 0x2df42ad9c70>,
     <matplotlib.lines.Line2D at 0x2df42ad9d60>,
     <matplotlib.lines.Line2D at 0x2df42ad9e50>,
     <matplotlib.lines.Line2D at 0x2df42ad9f40>,
     <matplotlib.lines.Line2D at 0x2df42ae0070>,
     <matplotlib.lines.Line2D at 0x2df42ae0160>,
     <matplotlib.lines.Line2D at 0x2df42ae0250>,
     <matplotlib.lines.Line2D at 0x2df42ae0340>,
     <matplotlib.lines.Line2D at 0x2df42ae0430>,
     <matplotlib.lines.Line2D at 0x2df42ae0520>,
     <matplotlib.lines.Line2D at 0x2df42ae0610>,
     <matplotlib.lines.Line2D at 0x2df42ae0700>,
     <matplotlib.lines.Line2D at 0x2df42ae07f0>,
     <matplotlib.lines.Line2D at 0x2df42ae08e0>,
     <matplotlib.lines.Line2D at 0x2df42ae09d0>,
     <matplotlib.lines.Line2D at 0x2df42ae0ac0>,
     <matplotlib.lines.Line2D at 0x2df42ae0bb0>,
     <matplotlib.lines.Line2D at 0x2df42ae0ca0>,
     <matplotlib.lines.Line2D at 0x2df42ae0d90>,
     <matplotlib.lines.Line2D at 0x2df42ae0e80>,
     <matplotlib.lines.Line2D at 0x2df42ae0f70>,
     <matplotlib.lines.Line2D at 0x2df42ae70a0>,
     <matplotlib.lines.Line2D at 0x2df42ae7190>,
     <matplotlib.lines.Line2D at 0x2df42ae7280>,
     <matplotlib.lines.Line2D at 0x2df42ae7370>,
     <matplotlib.lines.Line2D at 0x2df42ae7460>,
     <matplotlib.lines.Line2D at 0x2df42ae7550>,
     <matplotlib.lines.Line2D at 0x2df42ae7640>,
     <matplotlib.lines.Line2D at 0x2df42ae7730>,
     <matplotlib.lines.Line2D at 0x2df42ae7820>,
     <matplotlib.lines.Line2D at 0x2df42ae7910>,
     <matplotlib.lines.Line2D at 0x2df42ae7a00>,
     <matplotlib.lines.Line2D at 0x2df42ae7af0>,
     <matplotlib.lines.Line2D at 0x2df42ae7be0>,
     <matplotlib.lines.Line2D at 0x2df42ae7cd0>,
     <matplotlib.lines.Line2D at 0x2df42ae7dc0>,
     <matplotlib.lines.Line2D at 0x2df42ae7eb0>,
     <matplotlib.lines.Line2D at 0x2df42ae7fa0>,
     <matplotlib.lines.Line2D at 0x2df42af10d0>,
     <matplotlib.lines.Line2D at 0x2df42af11c0>,
     <matplotlib.lines.Line2D at 0x2df42af12b0>,
     <matplotlib.lines.Line2D at 0x2df42af13a0>,
     <matplotlib.lines.Line2D at 0x2df42af1490>,
     <matplotlib.lines.Line2D at 0x2df42af1580>,
     <matplotlib.lines.Line2D at 0x2df42af1670>,
     <matplotlib.lines.Line2D at 0x2df42af1760>,
     <matplotlib.lines.Line2D at 0x2df42af1850>,
     <matplotlib.lines.Line2D at 0x2df42af1940>,
     <matplotlib.lines.Line2D at 0x2df42af1a30>,
     <matplotlib.lines.Line2D at 0x2df42af1b20>,
     <matplotlib.lines.Line2D at 0x2df42af1c10>,
     <matplotlib.lines.Line2D at 0x2df42af1d00>,
     <matplotlib.lines.Line2D at 0x2df42af1df0>,
     <matplotlib.lines.Line2D at 0x2df42af1ee0>,
     <matplotlib.lines.Line2D at 0x2df42af1fd0>,
     <matplotlib.lines.Line2D at 0x2df42af9100>,
     <matplotlib.lines.Line2D at 0x2df42af91f0>,
     <matplotlib.lines.Line2D at 0x2df42af92e0>,
     <matplotlib.lines.Line2D at 0x2df42af93d0>,
     <matplotlib.lines.Line2D at 0x2df42af94c0>,
     <matplotlib.lines.Line2D at 0x2df42af95b0>,
     <matplotlib.lines.Line2D at 0x2df42af96a0>,
     <matplotlib.lines.Line2D at 0x2df42af9790>,
     <matplotlib.lines.Line2D at 0x2df42af9880>,
     <matplotlib.lines.Line2D at 0x2df42af9970>,
     <matplotlib.lines.Line2D at 0x2df42af9a60>,
     <matplotlib.lines.Line2D at 0x2df42af9b50>,
     <matplotlib.lines.Line2D at 0x2df42af9c40>,
     <matplotlib.lines.Line2D at 0x2df42af9d30>,
     <matplotlib.lines.Line2D at 0x2df42af9e20>,
     <matplotlib.lines.Line2D at 0x2df42af9f10>,
     <matplotlib.lines.Line2D at 0x2df42b01040>,
     <matplotlib.lines.Line2D at 0x2df42b01130>,
     <matplotlib.lines.Line2D at 0x2df42b01220>,
     <matplotlib.lines.Line2D at 0x2df42b01310>,
     <matplotlib.lines.Line2D at 0x2df42b01400>,
     <matplotlib.lines.Line2D at 0x2df42b014f0>,
     <matplotlib.lines.Line2D at 0x2df42b015e0>,
     <matplotlib.lines.Line2D at 0x2df42b016d0>,
     <matplotlib.lines.Line2D at 0x2df42b017c0>,
     <matplotlib.lines.Line2D at 0x2df42b018b0>,
     <matplotlib.lines.Line2D at 0x2df42b019a0>,
     <matplotlib.lines.Line2D at 0x2df42b01a90>,
     <matplotlib.lines.Line2D at 0x2df42b01b80>,
     <matplotlib.lines.Line2D at 0x2df42b01c70>,
     <matplotlib.lines.Line2D at 0x2df42b01d60>,
     <matplotlib.lines.Line2D at 0x2df42b01e50>,
     <matplotlib.lines.Line2D at 0x2df42b01f40>,
     <matplotlib.lines.Line2D at 0x2df42b09070>,
     <matplotlib.lines.Line2D at 0x2df42b09160>,
     <matplotlib.lines.Line2D at 0x2df42b09250>,
     <matplotlib.lines.Line2D at 0x2df42b09340>,
     <matplotlib.lines.Line2D at 0x2df42b09430>,
     <matplotlib.lines.Line2D at 0x2df42b09520>,
     <matplotlib.lines.Line2D at 0x2df42b09610>,
     <matplotlib.lines.Line2D at 0x2df42b09700>,
     <matplotlib.lines.Line2D at 0x2df42b097f0>,
     <matplotlib.lines.Line2D at 0x2df42b098e0>,
     <matplotlib.lines.Line2D at 0x2df42b099d0>,
     <matplotlib.lines.Line2D at 0x2df42b09ac0>,
     <matplotlib.lines.Line2D at 0x2df42b09bb0>,
     <matplotlib.lines.Line2D at 0x2df42b09ca0>,
     <matplotlib.lines.Line2D at 0x2df42b09d90>,
     <matplotlib.lines.Line2D at 0x2df42b09e80>,
     <matplotlib.lines.Line2D at 0x2df42b09f70>,
     <matplotlib.lines.Line2D at 0x2df42b110a0>,
     <matplotlib.lines.Line2D at 0x2df42b11190>,
     <matplotlib.lines.Line2D at 0x2df42b11280>,
     <matplotlib.lines.Line2D at 0x2df42b11370>,
     <matplotlib.lines.Line2D at 0x2df42b11460>,
     <matplotlib.lines.Line2D at 0x2df42b11550>,
     <matplotlib.lines.Line2D at 0x2df42b11640>,
     <matplotlib.lines.Line2D at 0x2df42b11730>,
     <matplotlib.lines.Line2D at 0x2df42b11820>,
     <matplotlib.lines.Line2D at 0x2df42b11910>,
     <matplotlib.lines.Line2D at 0x2df42b11a00>,
     <matplotlib.lines.Line2D at 0x2df42b11af0>,
     <matplotlib.lines.Line2D at 0x2df42b11be0>,
     <matplotlib.lines.Line2D at 0x2df42b11cd0>,
     <matplotlib.lines.Line2D at 0x2df42b11dc0>,
     <matplotlib.lines.Line2D at 0x2df42b11eb0>,
     <matplotlib.lines.Line2D at 0x2df42b11fa0>,
     <matplotlib.lines.Line2D at 0x2df42b190d0>,
     <matplotlib.lines.Line2D at 0x2df42b191c0>,
     <matplotlib.lines.Line2D at 0x2df42b192b0>,
     <matplotlib.lines.Line2D at 0x2df42b193a0>,
     <matplotlib.lines.Line2D at 0x2df42b19490>,
     <matplotlib.lines.Line2D at 0x2df42b19580>,
     <matplotlib.lines.Line2D at 0x2df42b19670>,
     <matplotlib.lines.Line2D at 0x2df42b19760>,
     <matplotlib.lines.Line2D at 0x2df42b19850>,
     <matplotlib.lines.Line2D at 0x2df42b19940>,
     <matplotlib.lines.Line2D at 0x2df42b19a30>,
     <matplotlib.lines.Line2D at 0x2df42b19b20>,
     <matplotlib.lines.Line2D at 0x2df42b19c10>,
     <matplotlib.lines.Line2D at 0x2df42b19d00>,
     <matplotlib.lines.Line2D at 0x2df42b19df0>,
     <matplotlib.lines.Line2D at 0x2df42b19ee0>,
     <matplotlib.lines.Line2D at 0x2df42b19fd0>,
     <matplotlib.lines.Line2D at 0x2df42b21100>,
     <matplotlib.lines.Line2D at 0x2df42b211f0>,
     <matplotlib.lines.Line2D at 0x2df42b212e0>,
     <matplotlib.lines.Line2D at 0x2df42b213d0>,
     <matplotlib.lines.Line2D at 0x2df42b214c0>,
     <matplotlib.lines.Line2D at 0x2df42b215b0>,
     <matplotlib.lines.Line2D at 0x2df42b216a0>,
     <matplotlib.lines.Line2D at 0x2df42b21790>,
     <matplotlib.lines.Line2D at 0x2df42b21880>,
     <matplotlib.lines.Line2D at 0x2df42b21970>,
     <matplotlib.lines.Line2D at 0x2df42b21a60>,
     <matplotlib.lines.Line2D at 0x2df42b21b50>,
     <matplotlib.lines.Line2D at 0x2df42b21c40>,
     <matplotlib.lines.Line2D at 0x2df42b21d30>,
     <matplotlib.lines.Line2D at 0x2df42b21e20>,
     <matplotlib.lines.Line2D at 0x2df42b21f10>,
     <matplotlib.lines.Line2D at 0x2df42b29040>,
     <matplotlib.lines.Line2D at 0x2df42b29130>,
     <matplotlib.lines.Line2D at 0x2df42b29220>,
     <matplotlib.lines.Line2D at 0x2df42b29310>,
     <matplotlib.lines.Line2D at 0x2df42b29400>,
     <matplotlib.lines.Line2D at 0x2df42b29520>,
     <matplotlib.lines.Line2D at 0x2df42b29610>,
     <matplotlib.lines.Line2D at 0x2df42b29700>,
     <matplotlib.lines.Line2D at 0x2df42b297f0>,
     <matplotlib.lines.Line2D at 0x2df42b298e0>,
     <matplotlib.lines.Line2D at 0x2df42b299d0>,
     <matplotlib.lines.Line2D at 0x2df42b29ac0>,
     <matplotlib.lines.Line2D at 0x2df42b29bb0>,
     <matplotlib.lines.Line2D at 0x2df42b29ca0>,
     <matplotlib.lines.Line2D at 0x2df42b29d90>,
     <matplotlib.lines.Line2D at 0x2df42b29e80>,
     <matplotlib.lines.Line2D at 0x2df42b29f70>,
     <matplotlib.lines.Line2D at 0x2df42b300a0>,
     <matplotlib.lines.Line2D at 0x2df42b30190>,
     <matplotlib.lines.Line2D at 0x2df42b30280>,
     <matplotlib.lines.Line2D at 0x2df42b30370>,
     <matplotlib.lines.Line2D at 0x2df42b30460>,
     <matplotlib.lines.Line2D at 0x2df42b30550>,
     <matplotlib.lines.Line2D at 0x2df42b30640>,
     <matplotlib.lines.Line2D at 0x2df42b30730>,
     <matplotlib.lines.Line2D at 0x2df42b30820>,
     <matplotlib.lines.Line2D at 0x2df42b30910>,
     <matplotlib.lines.Line2D at 0x2df42b30a00>,
     <matplotlib.lines.Line2D at 0x2df42b30af0>,
     <matplotlib.lines.Line2D at 0x2df42b30be0>,
     <matplotlib.lines.Line2D at 0x2df42b30cd0>,
     <matplotlib.lines.Line2D at 0x2df42b30dc0>,
     <matplotlib.lines.Line2D at 0x2df42b30eb0>,
     <matplotlib.lines.Line2D at 0x2df42b30fa0>,
     <matplotlib.lines.Line2D at 0x2df42b390d0>,
     <matplotlib.lines.Line2D at 0x2df42b391c0>,
     <matplotlib.lines.Line2D at 0x2df42b392b0>,
     <matplotlib.lines.Line2D at 0x2df42b393a0>,
     <matplotlib.lines.Line2D at 0x2df42b39490>,
     <matplotlib.lines.Line2D at 0x2df42b39580>,
     <matplotlib.lines.Line2D at 0x2df42b39670>,
     <matplotlib.lines.Line2D at 0x2df42b39760>,
     <matplotlib.lines.Line2D at 0x2df42b39850>,
     <matplotlib.lines.Line2D at 0x2df42b39940>,
     <matplotlib.lines.Line2D at 0x2df42b39a30>,
     <matplotlib.lines.Line2D at 0x2df42b39b20>,
     <matplotlib.lines.Line2D at 0x2df42b39c10>,
     <matplotlib.lines.Line2D at 0x2df42b39d00>,
     <matplotlib.lines.Line2D at 0x2df42b39df0>,
     <matplotlib.lines.Line2D at 0x2df42b39ee0>,
     <matplotlib.lines.Line2D at 0x2df42b39fd0>,
     <matplotlib.lines.Line2D at 0x2df42b40100>,
     <matplotlib.lines.Line2D at 0x2df42b401f0>,
     <matplotlib.lines.Line2D at 0x2df42b402e0>,
     <matplotlib.lines.Line2D at 0x2df42b403d0>,
     <matplotlib.lines.Line2D at 0x2df42b404c0>,
     <matplotlib.lines.Line2D at 0x2df42b405b0>,
     <matplotlib.lines.Line2D at 0x2df42b406a0>,
     <matplotlib.lines.Line2D at 0x2df42b40790>,
     <matplotlib.lines.Line2D at 0x2df42b40880>,
     <matplotlib.lines.Line2D at 0x2df42b40970>,
     <matplotlib.lines.Line2D at 0x2df42b40a60>,
     <matplotlib.lines.Line2D at 0x2df42b40b50>,
     <matplotlib.lines.Line2D at 0x2df42b40c40>,
     <matplotlib.lines.Line2D at 0x2df42b40d30>,
     <matplotlib.lines.Line2D at 0x2df42b40e20>,
     <matplotlib.lines.Line2D at 0x2df42b40f10>,
     <matplotlib.lines.Line2D at 0x2df42b48040>,
     <matplotlib.lines.Line2D at 0x2df42b48130>,
     <matplotlib.lines.Line2D at 0x2df42b48220>,
     <matplotlib.lines.Line2D at 0x2df42b48310>,
     <matplotlib.lines.Line2D at 0x2df42b48400>,
     <matplotlib.lines.Line2D at 0x2df42b484f0>,
     <matplotlib.lines.Line2D at 0x2df42b485e0>,
     <matplotlib.lines.Line2D at 0x2df42b486d0>,
     <matplotlib.lines.Line2D at 0x2df42b487c0>,
     <matplotlib.lines.Line2D at 0x2df42b488b0>,
     <matplotlib.lines.Line2D at 0x2df42b489a0>,
     <matplotlib.lines.Line2D at 0x2df42b48a90>,
     <matplotlib.lines.Line2D at 0x2df42b48b80>,
     <matplotlib.lines.Line2D at 0x2df42b48c70>,
     <matplotlib.lines.Line2D at 0x2df42b48d60>,
     <matplotlib.lines.Line2D at 0x2df42b48e50>,
     <matplotlib.lines.Line2D at 0x2df42b48f40>,
     <matplotlib.lines.Line2D at 0x2df42b51070>,
     <matplotlib.lines.Line2D at 0x2df42b51160>,
     <matplotlib.lines.Line2D at 0x2df42b51250>,
     <matplotlib.lines.Line2D at 0x2df42b51340>,
     <matplotlib.lines.Line2D at 0x2df42b51430>,
     <matplotlib.lines.Line2D at 0x2df42b51520>,
     <matplotlib.lines.Line2D at 0x2df42b51610>,
     <matplotlib.lines.Line2D at 0x2df42b51700>,
     <matplotlib.lines.Line2D at 0x2df42b517f0>,
     <matplotlib.lines.Line2D at 0x2df42b518e0>,
     <matplotlib.lines.Line2D at 0x2df42b519d0>,
     <matplotlib.lines.Line2D at 0x2df42b51ac0>,
     <matplotlib.lines.Line2D at 0x2df42b51bb0>,
     <matplotlib.lines.Line2D at 0x2df42b51ca0>,
     <matplotlib.lines.Line2D at 0x2df42b51d90>,
     <matplotlib.lines.Line2D at 0x2df42b51e80>,
     <matplotlib.lines.Line2D at 0x2df42b51f70>,
     <matplotlib.lines.Line2D at 0x2df42b580a0>,
     <matplotlib.lines.Line2D at 0x2df42b58190>,
     <matplotlib.lines.Line2D at 0x2df42b58280>,
     <matplotlib.lines.Line2D at 0x2df42b58370>,
     <matplotlib.lines.Line2D at 0x2df42b58460>,
     <matplotlib.lines.Line2D at 0x2df42b58550>,
     <matplotlib.lines.Line2D at 0x2df42b58640>,
     <matplotlib.lines.Line2D at 0x2df42b58730>,
     <matplotlib.lines.Line2D at 0x2df42b58820>,
     <matplotlib.lines.Line2D at 0x2df42b58910>,
     <matplotlib.lines.Line2D at 0x2df42b58a00>,
     <matplotlib.lines.Line2D at 0x2df42b58af0>,
     <matplotlib.lines.Line2D at 0x2df42b58be0>,
     <matplotlib.lines.Line2D at 0x2df42b58cd0>,
     <matplotlib.lines.Line2D at 0x2df42b58dc0>,
     <matplotlib.lines.Line2D at 0x2df42b58eb0>,
     <matplotlib.lines.Line2D at 0x2df42b58fa0>,
     <matplotlib.lines.Line2D at 0x2df42b5f0d0>,
     <matplotlib.lines.Line2D at 0x2df42b5f1c0>,
     <matplotlib.lines.Line2D at 0x2df42b5f2b0>,
     <matplotlib.lines.Line2D at 0x2df42b5f3a0>,
     <matplotlib.lines.Line2D at 0x2df42b5f490>,
     <matplotlib.lines.Line2D at 0x2df42b5f580>,
     <matplotlib.lines.Line2D at 0x2df42b5f670>,
     <matplotlib.lines.Line2D at 0x2df42b5f760>,
     <matplotlib.lines.Line2D at 0x2df42b5f850>,
     <matplotlib.lines.Line2D at 0x2df42b5f940>,
     <matplotlib.lines.Line2D at 0x2df42b5fa30>,
     <matplotlib.lines.Line2D at 0x2df42b5fb20>,
     <matplotlib.lines.Line2D at 0x2df42b5fc10>,
     <matplotlib.lines.Line2D at 0x2df42b5fd00>,
     <matplotlib.lines.Line2D at 0x2df42b5fdf0>,
     <matplotlib.lines.Line2D at 0x2df42b5fee0>,
     <matplotlib.lines.Line2D at 0x2df42b5ffd0>,
     <matplotlib.lines.Line2D at 0x2df42b69100>,
     <matplotlib.lines.Line2D at 0x2df42b691f0>,
     <matplotlib.lines.Line2D at 0x2df42b692e0>,
     <matplotlib.lines.Line2D at 0x2df42b693d0>,
     <matplotlib.lines.Line2D at 0x2df42b694c0>,
     <matplotlib.lines.Line2D at 0x2df42b695b0>,
     <matplotlib.lines.Line2D at 0x2df42b696a0>,
     <matplotlib.lines.Line2D at 0x2df42b69790>,
     <matplotlib.lines.Line2D at 0x2df42b69880>,
     <matplotlib.lines.Line2D at 0x2df42b69970>,
     <matplotlib.lines.Line2D at 0x2df42b69a60>,
     <matplotlib.lines.Line2D at 0x2df42b69b50>,
     <matplotlib.lines.Line2D at 0x2df42b69c40>,
     <matplotlib.lines.Line2D at 0x2df42b69d30>,
     <matplotlib.lines.Line2D at 0x2df42b69e20>,
     <matplotlib.lines.Line2D at 0x2df42b69f10>,
     <matplotlib.lines.Line2D at 0x2df42b71040>,
     <matplotlib.lines.Line2D at 0x2df42b71130>,
     <matplotlib.lines.Line2D at 0x2df42b71220>,
     <matplotlib.lines.Line2D at 0x2df42b71310>,
     <matplotlib.lines.Line2D at 0x2df42b71400>,
     <matplotlib.lines.Line2D at 0x2df42b714f0>,
     <matplotlib.lines.Line2D at 0x2df42b715e0>,
     <matplotlib.lines.Line2D at 0x2df42b716d0>,
     <matplotlib.lines.Line2D at 0x2df42b717c0>,
     <matplotlib.lines.Line2D at 0x2df42b718b0>,
     <matplotlib.lines.Line2D at 0x2df42b719a0>,
     <matplotlib.lines.Line2D at 0x2df42b71a90>,
     <matplotlib.lines.Line2D at 0x2df42b71b80>,
     <matplotlib.lines.Line2D at 0x2df42b71c70>,
     <matplotlib.lines.Line2D at 0x2df42b71d60>,
     <matplotlib.lines.Line2D at 0x2df42b71e50>,
     <matplotlib.lines.Line2D at 0x2df42b71f40>,
     <matplotlib.lines.Line2D at 0x2df42b79070>,
     <matplotlib.lines.Line2D at 0x2df42b79160>,
     <matplotlib.lines.Line2D at 0x2df42b79250>,
     <matplotlib.lines.Line2D at 0x2df42b79340>,
     <matplotlib.lines.Line2D at 0x2df42b79430>,
     <matplotlib.lines.Line2D at 0x2df42b79520>,
     <matplotlib.lines.Line2D at 0x2df42b79610>,
     <matplotlib.lines.Line2D at 0x2df42b79700>,
     <matplotlib.lines.Line2D at 0x2df42b797f0>,
     <matplotlib.lines.Line2D at 0x2df42b798e0>,
     <matplotlib.lines.Line2D at 0x2df42b799d0>,
     <matplotlib.lines.Line2D at 0x2df42b79ac0>,
     <matplotlib.lines.Line2D at 0x2df42b79bb0>,
     <matplotlib.lines.Line2D at 0x2df42b79ca0>,
     <matplotlib.lines.Line2D at 0x2df42b79d90>,
     <matplotlib.lines.Line2D at 0x2df42b79e80>,
     <matplotlib.lines.Line2D at 0x2df42b79f70>,
     <matplotlib.lines.Line2D at 0x2df42b810a0>,
     <matplotlib.lines.Line2D at 0x2df42b81190>,
     <matplotlib.lines.Line2D at 0x2df42b81280>,
     <matplotlib.lines.Line2D at 0x2df42b81370>,
     <matplotlib.lines.Line2D at 0x2df42b81460>,
     <matplotlib.lines.Line2D at 0x2df42b81550>,
     <matplotlib.lines.Line2D at 0x2df42b81640>,
     <matplotlib.lines.Line2D at 0x2df42b81730>,
     <matplotlib.lines.Line2D at 0x2df42b81820>,
     <matplotlib.lines.Line2D at 0x2df42b81910>,
     <matplotlib.lines.Line2D at 0x2df42b81a00>,
     <matplotlib.lines.Line2D at 0x2df42b81af0>,
     <matplotlib.lines.Line2D at 0x2df42b81be0>,
     <matplotlib.lines.Line2D at 0x2df42b81cd0>,
     <matplotlib.lines.Line2D at 0x2df42b81dc0>,
     <matplotlib.lines.Line2D at 0x2df42b81eb0>,
     <matplotlib.lines.Line2D at 0x2df42b81fa0>,
     <matplotlib.lines.Line2D at 0x2df42b890d0>,
     <matplotlib.lines.Line2D at 0x2df42b891c0>,
     <matplotlib.lines.Line2D at 0x2df42b892b0>,
     <matplotlib.lines.Line2D at 0x2df42b893a0>,
     <matplotlib.lines.Line2D at 0x2df42b89490>,
     <matplotlib.lines.Line2D at 0x2df42b89580>,
     <matplotlib.lines.Line2D at 0x2df42b89670>,
     <matplotlib.lines.Line2D at 0x2df42b89760>,
     <matplotlib.lines.Line2D at 0x2df42b89850>,
     <matplotlib.lines.Line2D at 0x2df42b89940>,
     <matplotlib.lines.Line2D at 0x2df42b89a30>,
     <matplotlib.lines.Line2D at 0x2df42b89b20>,
     <matplotlib.lines.Line2D at 0x2df42b89c10>,
     <matplotlib.lines.Line2D at 0x2df42b89d00>,
     <matplotlib.lines.Line2D at 0x2df42b89df0>,
     <matplotlib.lines.Line2D at 0x2df42b89ee0>,
     <matplotlib.lines.Line2D at 0x2df42b89fd0>,
     <matplotlib.lines.Line2D at 0x2df42b92100>,
     <matplotlib.lines.Line2D at 0x2df42b921f0>,
     <matplotlib.lines.Line2D at 0x2df42b922e0>,
     <matplotlib.lines.Line2D at 0x2df42b923d0>,
     <matplotlib.lines.Line2D at 0x2df42b924c0>,
     <matplotlib.lines.Line2D at 0x2df42b925b0>,
     <matplotlib.lines.Line2D at 0x2df42b926a0>,
     <matplotlib.lines.Line2D at 0x2df42b92790>,
     <matplotlib.lines.Line2D at 0x2df42b92880>,
     <matplotlib.lines.Line2D at 0x2df42b92970>,
     <matplotlib.lines.Line2D at 0x2df42b92a60>,
     <matplotlib.lines.Line2D at 0x2df42b92b50>,
     <matplotlib.lines.Line2D at 0x2df42b92c40>,
     <matplotlib.lines.Line2D at 0x2df42b92d30>,
     <matplotlib.lines.Line2D at 0x2df42b92e20>,
     <matplotlib.lines.Line2D at 0x2df42b92f10>,
     <matplotlib.lines.Line2D at 0x2df42b99040>,
     <matplotlib.lines.Line2D at 0x2df42b99130>,
     <matplotlib.lines.Line2D at 0x2df42b99220>,
     <matplotlib.lines.Line2D at 0x2df42b99310>,
     <matplotlib.lines.Line2D at 0x2df42b99400>,
     <matplotlib.lines.Line2D at 0x2df42b994f0>,
     <matplotlib.lines.Line2D at 0x2df42b995e0>,
     <matplotlib.lines.Line2D at 0x2df42b996d0>,
     <matplotlib.lines.Line2D at 0x2df42b997c0>,
     <matplotlib.lines.Line2D at 0x2df42b998b0>,
     <matplotlib.lines.Line2D at 0x2df42b999a0>,
     <matplotlib.lines.Line2D at 0x2df42b99a90>,
     <matplotlib.lines.Line2D at 0x2df42b99b80>,
     <matplotlib.lines.Line2D at 0x2df42b99c70>,
     <matplotlib.lines.Line2D at 0x2df42b99d60>,
     <matplotlib.lines.Line2D at 0x2df42b99e50>,
     <matplotlib.lines.Line2D at 0x2df42b99f40>,
     <matplotlib.lines.Line2D at 0x2df42ba1070>,
     <matplotlib.lines.Line2D at 0x2df42ba1160>,
     <matplotlib.lines.Line2D at 0x2df42ba1250>,
     <matplotlib.lines.Line2D at 0x2df42ba1340>,
     <matplotlib.lines.Line2D at 0x2df42ba1430>,
     <matplotlib.lines.Line2D at 0x2df42ba1520>,
     <matplotlib.lines.Line2D at 0x2df42ba1610>,
     <matplotlib.lines.Line2D at 0x2df42ba1700>,
     <matplotlib.lines.Line2D at 0x2df42ba17f0>,
     <matplotlib.lines.Line2D at 0x2df42ba18e0>,
     <matplotlib.lines.Line2D at 0x2df42ba19d0>,
     <matplotlib.lines.Line2D at 0x2df42ba1ac0>,
     <matplotlib.lines.Line2D at 0x2df42ba1bb0>,
     <matplotlib.lines.Line2D at 0x2df42ba1ca0>,
     <matplotlib.lines.Line2D at 0x2df42ba1d90>,
     <matplotlib.lines.Line2D at 0x2df42ba1e80>,
     <matplotlib.lines.Line2D at 0x2df42ba1f70>,
     <matplotlib.lines.Line2D at 0x2df42ba90a0>,
     <matplotlib.lines.Line2D at 0x2df42ba9190>,
     <matplotlib.lines.Line2D at 0x2df42ba9280>,
     <matplotlib.lines.Line2D at 0x2df42ba9370>,
     <matplotlib.lines.Line2D at 0x2df42ba9460>,
     <matplotlib.lines.Line2D at 0x2df42ba9550>,
     <matplotlib.lines.Line2D at 0x2df42ba9640>,
     <matplotlib.lines.Line2D at 0x2df42ba9730>,
     <matplotlib.lines.Line2D at 0x2df42ba9820>,
     <matplotlib.lines.Line2D at 0x2df42ba9910>,
     <matplotlib.lines.Line2D at 0x2df42ba9a00>,
     <matplotlib.lines.Line2D at 0x2df42ba9af0>,
     <matplotlib.lines.Line2D at 0x2df42ba9be0>,
     <matplotlib.lines.Line2D at 0x2df42ba9cd0>,
     <matplotlib.lines.Line2D at 0x2df42ba9dc0>,
     <matplotlib.lines.Line2D at 0x2df42ba9eb0>,
     <matplotlib.lines.Line2D at 0x2df42ba9fa0>,
     <matplotlib.lines.Line2D at 0x2df42bb10d0>,
     <matplotlib.lines.Line2D at 0x2df42bb11c0>,
     <matplotlib.lines.Line2D at 0x2df42bb12b0>,
     <matplotlib.lines.Line2D at 0x2df42bb13a0>,
     <matplotlib.lines.Line2D at 0x2df42bb1490>,
     <matplotlib.lines.Line2D at 0x2df42bb1580>,
     <matplotlib.lines.Line2D at 0x2df42bb1670>,
     <matplotlib.lines.Line2D at 0x2df42bb1760>,
     <matplotlib.lines.Line2D at 0x2df42bb1850>,
     <matplotlib.lines.Line2D at 0x2df42bb1940>,
     <matplotlib.lines.Line2D at 0x2df42bb1a30>,
     <matplotlib.lines.Line2D at 0x2df42bb1b20>,
     <matplotlib.lines.Line2D at 0x2df42bb1c10>,
     <matplotlib.lines.Line2D at 0x2df42bb1d00>,
     <matplotlib.lines.Line2D at 0x2df42bb1df0>,
     <matplotlib.lines.Line2D at 0x2df42bb1ee0>,
     <matplotlib.lines.Line2D at 0x2df42bb1fd0>,
     <matplotlib.lines.Line2D at 0x2df42bb8100>,
     <matplotlib.lines.Line2D at 0x2df42bb81f0>,
     <matplotlib.lines.Line2D at 0x2df42bb82e0>,
     <matplotlib.lines.Line2D at 0x2df42bb83d0>,
     <matplotlib.lines.Line2D at 0x2df42bb84c0>,
     <matplotlib.lines.Line2D at 0x2df42bb85b0>,
     <matplotlib.lines.Line2D at 0x2df42bb86a0>,
     <matplotlib.lines.Line2D at 0x2df42bb8790>,
     <matplotlib.lines.Line2D at 0x2df42bb8880>,
     <matplotlib.lines.Line2D at 0x2df42bb8970>,
     <matplotlib.lines.Line2D at 0x2df42bb8a60>,
     <matplotlib.lines.Line2D at 0x2df42bb8b50>,
     <matplotlib.lines.Line2D at 0x2df42bb8c40>,
     <matplotlib.lines.Line2D at 0x2df42bb8d30>,
     <matplotlib.lines.Line2D at 0x2df42bb8e20>,
     <matplotlib.lines.Line2D at 0x2df42bb8f10>,
     <matplotlib.lines.Line2D at 0x2df42bc1040>,
     <matplotlib.lines.Line2D at 0x2df42bc1130>,
     <matplotlib.lines.Line2D at 0x2df42bc1220>,
     <matplotlib.lines.Line2D at 0x2df42bc1310>,
     <matplotlib.lines.Line2D at 0x2df42bc1400>,
     <matplotlib.lines.Line2D at 0x2df42bc14f0>,
     <matplotlib.lines.Line2D at 0x2df42bc15e0>,
     <matplotlib.lines.Line2D at 0x2df42bc16d0>,
     <matplotlib.lines.Line2D at 0x2df42bc17c0>,
     <matplotlib.lines.Line2D at 0x2df42bc18b0>,
     <matplotlib.lines.Line2D at 0x2df42bc19a0>,
     <matplotlib.lines.Line2D at 0x2df42bc1a90>,
     <matplotlib.lines.Line2D at 0x2df42bc1b80>,
     <matplotlib.lines.Line2D at 0x2df42bc1c70>,
     <matplotlib.lines.Line2D at 0x2df42bc1d60>,
     <matplotlib.lines.Line2D at 0x2df42bc1e50>,
     <matplotlib.lines.Line2D at 0x2df42bc1f40>,
     <matplotlib.lines.Line2D at 0x2df42bc9070>,
     <matplotlib.lines.Line2D at 0x2df42bc9160>,
     <matplotlib.lines.Line2D at 0x2df42bc9250>,
     <matplotlib.lines.Line2D at 0x2df42bc9340>,
     <matplotlib.lines.Line2D at 0x2df42bc9430>,
     <matplotlib.lines.Line2D at 0x2df42bc9520>,
     <matplotlib.lines.Line2D at 0x2df42bc9610>,
     <matplotlib.lines.Line2D at 0x2df42bc9700>,
     <matplotlib.lines.Line2D at 0x2df42bc97f0>,
     <matplotlib.lines.Line2D at 0x2df42bc98e0>,
     <matplotlib.lines.Line2D at 0x2df42bc99d0>,
     <matplotlib.lines.Line2D at 0x2df42bc9ac0>,
     <matplotlib.lines.Line2D at 0x2df42bc9bb0>,
     <matplotlib.lines.Line2D at 0x2df42bc9ca0>,
     <matplotlib.lines.Line2D at 0x2df42bc9d90>,
     <matplotlib.lines.Line2D at 0x2df42bc9e80>,
     <matplotlib.lines.Line2D at 0x2df42bc9f70>,
     <matplotlib.lines.Line2D at 0x2df42bcf0a0>,
     <matplotlib.lines.Line2D at 0x2df42bcf190>,
     <matplotlib.lines.Line2D at 0x2df42bcf280>,
     <matplotlib.lines.Line2D at 0x2df42bcf370>,
     <matplotlib.lines.Line2D at 0x2df42bcf460>,
     <matplotlib.lines.Line2D at 0x2df42bcf550>,
     <matplotlib.lines.Line2D at 0x2df42bcf640>,
     <matplotlib.lines.Line2D at 0x2df42bcf730>,
     <matplotlib.lines.Line2D at 0x2df42bcf820>,
     <matplotlib.lines.Line2D at 0x2df42bcf910>,
     <matplotlib.lines.Line2D at 0x2df42bcfa00>,
     <matplotlib.lines.Line2D at 0x2df42bcfaf0>,
     <matplotlib.lines.Line2D at 0x2df42bcfbe0>,
     <matplotlib.lines.Line2D at 0x2df42bcfcd0>,
     <matplotlib.lines.Line2D at 0x2df42bcfdc0>,
     <matplotlib.lines.Line2D at 0x2df42bcfeb0>,
     <matplotlib.lines.Line2D at 0x2df42bcffa0>,
     <matplotlib.lines.Line2D at 0x2df42bd80d0>,
     <matplotlib.lines.Line2D at 0x2df42bd81c0>,
     <matplotlib.lines.Line2D at 0x2df42bd82b0>,
     <matplotlib.lines.Line2D at 0x2df42bd83a0>,
     <matplotlib.lines.Line2D at 0x2df42bd8490>,
     <matplotlib.lines.Line2D at 0x2df42bd8580>,
     <matplotlib.lines.Line2D at 0x2df42bd8670>,
     <matplotlib.lines.Line2D at 0x2df42bd8760>,
     <matplotlib.lines.Line2D at 0x2df42bd8850>,
     <matplotlib.lines.Line2D at 0x2df42bd8940>,
     <matplotlib.lines.Line2D at 0x2df42bd8a30>,
     <matplotlib.lines.Line2D at 0x2df42bd8b20>,
     <matplotlib.lines.Line2D at 0x2df42bd8c10>,
     <matplotlib.lines.Line2D at 0x2df42bd8d00>,
     <matplotlib.lines.Line2D at 0x2df42bd8df0>,
     <matplotlib.lines.Line2D at 0x2df42bd8ee0>,
     <matplotlib.lines.Line2D at 0x2df42bd8fd0>,
     <matplotlib.lines.Line2D at 0x2df42be1100>,
     <matplotlib.lines.Line2D at 0x2df42be11f0>,
     <matplotlib.lines.Line2D at 0x2df42be12e0>,
     <matplotlib.lines.Line2D at 0x2df42be13d0>,
     <matplotlib.lines.Line2D at 0x2df42be14c0>,
     <matplotlib.lines.Line2D at 0x2df42be15b0>,
     <matplotlib.lines.Line2D at 0x2df42be16a0>,
     <matplotlib.lines.Line2D at 0x2df42be1790>,
     <matplotlib.lines.Line2D at 0x2df42be1880>,
     <matplotlib.lines.Line2D at 0x2df42be1970>,
     <matplotlib.lines.Line2D at 0x2df42be1a60>,
     <matplotlib.lines.Line2D at 0x2df42be1b50>,
     <matplotlib.lines.Line2D at 0x2df42be1c40>,
     <matplotlib.lines.Line2D at 0x2df42be1d30>,
     <matplotlib.lines.Line2D at 0x2df42be1e20>,
     <matplotlib.lines.Line2D at 0x2df42be1f10>,
     <matplotlib.lines.Line2D at 0x2df42be9040>,
     <matplotlib.lines.Line2D at 0x2df42be9130>,
     <matplotlib.lines.Line2D at 0x2df42be9220>,
     <matplotlib.lines.Line2D at 0x2df42be9310>,
     <matplotlib.lines.Line2D at 0x2df42be9400>,
     <matplotlib.lines.Line2D at 0x2df42be94f0>,
     <matplotlib.lines.Line2D at 0x2df42be95e0>,
     <matplotlib.lines.Line2D at 0x2df42be96d0>,
     <matplotlib.lines.Line2D at 0x2df42be97c0>,
     <matplotlib.lines.Line2D at 0x2df42be98b0>,
     <matplotlib.lines.Line2D at 0x2df42be99a0>,
     <matplotlib.lines.Line2D at 0x2df42be9a90>,
     <matplotlib.lines.Line2D at 0x2df42be9b80>,
     <matplotlib.lines.Line2D at 0x2df42be9c70>,
     <matplotlib.lines.Line2D at 0x2df42be9d60>,
     <matplotlib.lines.Line2D at 0x2df42be9e50>,
     <matplotlib.lines.Line2D at 0x2df42be9f40>,
     <matplotlib.lines.Line2D at 0x2df42bf2070>,
     <matplotlib.lines.Line2D at 0x2df42bf2160>,
     <matplotlib.lines.Line2D at 0x2df42bf2250>,
     <matplotlib.lines.Line2D at 0x2df42bf2340>,
     <matplotlib.lines.Line2D at 0x2df42bf2430>,
     <matplotlib.lines.Line2D at 0x2df42bf2520>,
     <matplotlib.lines.Line2D at 0x2df42bf2610>,
     <matplotlib.lines.Line2D at 0x2df42bf2700>,
     <matplotlib.lines.Line2D at 0x2df42bf27f0>,
     <matplotlib.lines.Line2D at 0x2df42bf28e0>,
     <matplotlib.lines.Line2D at 0x2df42bf29d0>,
     <matplotlib.lines.Line2D at 0x2df42bf2ac0>,
     <matplotlib.lines.Line2D at 0x2df42bf2bb0>,
     <matplotlib.lines.Line2D at 0x2df42bf2cd0>,
     <matplotlib.lines.Line2D at 0x2df42bf2dc0>,
     <matplotlib.lines.Line2D at 0x2df42bf2eb0>,
     <matplotlib.lines.Line2D at 0x2df42bf2fa0>,
     <matplotlib.lines.Line2D at 0x2df42bf90d0>,
     <matplotlib.lines.Line2D at 0x2df42bf91c0>,
     <matplotlib.lines.Line2D at 0x2df42bf92b0>,
     <matplotlib.lines.Line2D at 0x2df42bf93a0>,
     <matplotlib.lines.Line2D at 0x2df42bf9490>,
     <matplotlib.lines.Line2D at 0x2df42bf9580>,
     <matplotlib.lines.Line2D at 0x2df42bf9670>,
     <matplotlib.lines.Line2D at 0x2df42bf9760>,
     <matplotlib.lines.Line2D at 0x2df42bf9850>,
     <matplotlib.lines.Line2D at 0x2df42bf9940>,
     <matplotlib.lines.Line2D at 0x2df42bf9a30>,
     <matplotlib.lines.Line2D at 0x2df42bf9b20>,
     <matplotlib.lines.Line2D at 0x2df42bf9c10>,
     <matplotlib.lines.Line2D at 0x2df42bf9d00>,
     <matplotlib.lines.Line2D at 0x2df42bf9df0>,
     <matplotlib.lines.Line2D at 0x2df42bf9ee0>,
     <matplotlib.lines.Line2D at 0x2df42bf9fd0>,
     <matplotlib.lines.Line2D at 0x2df42c01100>,
     <matplotlib.lines.Line2D at 0x2df42c011f0>,
     <matplotlib.lines.Line2D at 0x2df42c012e0>,
     <matplotlib.lines.Line2D at 0x2df42c013d0>,
     <matplotlib.lines.Line2D at 0x2df42c014c0>,
     <matplotlib.lines.Line2D at 0x2df42c015b0>,
     <matplotlib.lines.Line2D at 0x2df42c016a0>,
     <matplotlib.lines.Line2D at 0x2df42c01790>,
     <matplotlib.lines.Line2D at 0x2df42c01880>,
     <matplotlib.lines.Line2D at 0x2df42c01970>,
     <matplotlib.lines.Line2D at 0x2df42c01a60>,
     <matplotlib.lines.Line2D at 0x2df42c01b50>,
     <matplotlib.lines.Line2D at 0x2df42c01c40>,
     <matplotlib.lines.Line2D at 0x2df42c01d30>,
     <matplotlib.lines.Line2D at 0x2df42c01e20>,
     <matplotlib.lines.Line2D at 0x2df42c01f10>,
     <matplotlib.lines.Line2D at 0x2df42c09040>,
     <matplotlib.lines.Line2D at 0x2df42c09130>,
     <matplotlib.lines.Line2D at 0x2df42c09220>,
     <matplotlib.lines.Line2D at 0x2df42c09310>,
     <matplotlib.lines.Line2D at 0x2df42c09400>,
     <matplotlib.lines.Line2D at 0x2df42c094f0>,
     <matplotlib.lines.Line2D at 0x2df42c095e0>,
     <matplotlib.lines.Line2D at 0x2df42c096d0>,
     <matplotlib.lines.Line2D at 0x2df42c097c0>,
     <matplotlib.lines.Line2D at 0x2df42c098b0>,
     <matplotlib.lines.Line2D at 0x2df42c099a0>,
     <matplotlib.lines.Line2D at 0x2df42c09a90>,
     <matplotlib.lines.Line2D at 0x2df42c09b80>,
     <matplotlib.lines.Line2D at 0x2df42c09c70>,
     <matplotlib.lines.Line2D at 0x2df42c09d60>,
     <matplotlib.lines.Line2D at 0x2df42c09e50>,
     <matplotlib.lines.Line2D at 0x2df42c09f40>,
     <matplotlib.lines.Line2D at 0x2df42c11070>,
     <matplotlib.lines.Line2D at 0x2df42c11160>,
     <matplotlib.lines.Line2D at 0x2df42c11250>,
     <matplotlib.lines.Line2D at 0x2df42c11340>,
     <matplotlib.lines.Line2D at 0x2df42c11430>,
     <matplotlib.lines.Line2D at 0x2df42c11520>,
     <matplotlib.lines.Line2D at 0x2df42c11610>,
     <matplotlib.lines.Line2D at 0x2df42c11700>,
     <matplotlib.lines.Line2D at 0x2df42c117f0>,
     <matplotlib.lines.Line2D at 0x2df42c118e0>,
     <matplotlib.lines.Line2D at 0x2df42c119d0>,
     <matplotlib.lines.Line2D at 0x2df42c11ac0>,
     <matplotlib.lines.Line2D at 0x2df42c11bb0>,
     <matplotlib.lines.Line2D at 0x2df42c11ca0>,
     <matplotlib.lines.Line2D at 0x2df42c11d90>,
     <matplotlib.lines.Line2D at 0x2df42c11e80>,
     <matplotlib.lines.Line2D at 0x2df42c11f70>,
     <matplotlib.lines.Line2D at 0x2df42c1a0a0>,
     <matplotlib.lines.Line2D at 0x2df42c1a190>,
     <matplotlib.lines.Line2D at 0x2df42c1a280>,
     <matplotlib.lines.Line2D at 0x2df42c1a370>,
     <matplotlib.lines.Line2D at 0x2df42c1a460>,
     <matplotlib.lines.Line2D at 0x2df42c1a550>,
     <matplotlib.lines.Line2D at 0x2df42c1a640>,
     <matplotlib.lines.Line2D at 0x2df42c1a730>,
     <matplotlib.lines.Line2D at 0x2df42c1a820>,
     <matplotlib.lines.Line2D at 0x2df42c1a910>,
     <matplotlib.lines.Line2D at 0x2df42c1aa00>,
     <matplotlib.lines.Line2D at 0x2df42c1aaf0>,
     <matplotlib.lines.Line2D at 0x2df42c1abe0>,
     <matplotlib.lines.Line2D at 0x2df42c1acd0>,
     <matplotlib.lines.Line2D at 0x2df42c1adc0>,
     <matplotlib.lines.Line2D at 0x2df42c1aeb0>,
     <matplotlib.lines.Line2D at 0x2df42c1afa0>,
     <matplotlib.lines.Line2D at 0x2df42c220d0>,
     <matplotlib.lines.Line2D at 0x2df42c221c0>,
     <matplotlib.lines.Line2D at 0x2df42c222b0>,
     <matplotlib.lines.Line2D at 0x2df42c223a0>,
     <matplotlib.lines.Line2D at 0x2df42c22490>,
     <matplotlib.lines.Line2D at 0x2df42c22580>,
     <matplotlib.lines.Line2D at 0x2df42c22670>,
     <matplotlib.lines.Line2D at 0x2df42c22760>,
     <matplotlib.lines.Line2D at 0x2df42c22850>,
     <matplotlib.lines.Line2D at 0x2df42c22940>,
     <matplotlib.lines.Line2D at 0x2df42c22a30>,
     <matplotlib.lines.Line2D at 0x2df42c22b20>,
     <matplotlib.lines.Line2D at 0x2df42c22c10>,
     <matplotlib.lines.Line2D at 0x2df42c22d00>,
     <matplotlib.lines.Line2D at 0x2df42c22df0>,
     <matplotlib.lines.Line2D at 0x2df42c22ee0>,
     <matplotlib.lines.Line2D at 0x2df42c22fd0>,
     <matplotlib.lines.Line2D at 0x2df42c29100>,
     <matplotlib.lines.Line2D at 0x2df42c291f0>,
     <matplotlib.lines.Line2D at 0x2df42c292e0>,
     <matplotlib.lines.Line2D at 0x2df42c293d0>,
     <matplotlib.lines.Line2D at 0x2df42c294c0>,
     <matplotlib.lines.Line2D at 0x2df42c295b0>,
     <matplotlib.lines.Line2D at 0x2df42c296a0>,
     <matplotlib.lines.Line2D at 0x2df42c29790>,
     <matplotlib.lines.Line2D at 0x2df42c29880>,
     <matplotlib.lines.Line2D at 0x2df42c29970>,
     <matplotlib.lines.Line2D at 0x2df42c29a60>,
     <matplotlib.lines.Line2D at 0x2df42c29b50>,
     <matplotlib.lines.Line2D at 0x2df42c29c40>,
     <matplotlib.lines.Line2D at 0x2df42c29d30>,
     <matplotlib.lines.Line2D at 0x2df42c29e20>,
     <matplotlib.lines.Line2D at 0x2df42c29f10>,
     <matplotlib.lines.Line2D at 0x2df42c30040>,
     <matplotlib.lines.Line2D at 0x2df42c30130>,
     <matplotlib.lines.Line2D at 0x2df42c30220>,
     <matplotlib.lines.Line2D at 0x2df42c30310>,
     <matplotlib.lines.Line2D at 0x2df42c30400>,
     <matplotlib.lines.Line2D at 0x2df42c304f0>,
     <matplotlib.lines.Line2D at 0x2df42c305e0>,
     <matplotlib.lines.Line2D at 0x2df42c306d0>,
     <matplotlib.lines.Line2D at 0x2df42c307c0>,
     <matplotlib.lines.Line2D at 0x2df42c308b0>,
     <matplotlib.lines.Line2D at 0x2df42c309a0>,
     <matplotlib.lines.Line2D at 0x2df42c30a90>,
     <matplotlib.lines.Line2D at 0x2df42c30b80>,
     <matplotlib.lines.Line2D at 0x2df42c30c70>,
     <matplotlib.lines.Line2D at 0x2df42c30d60>,
     <matplotlib.lines.Line2D at 0x2df42c30e50>,
     <matplotlib.lines.Line2D at 0x2df42c30f40>,
     <matplotlib.lines.Line2D at 0x2df42c39070>,
     <matplotlib.lines.Line2D at 0x2df42c39160>,
     <matplotlib.lines.Line2D at 0x2df42c39250>,
     <matplotlib.lines.Line2D at 0x2df42c39340>,
     <matplotlib.lines.Line2D at 0x2df42c39430>,
     <matplotlib.lines.Line2D at 0x2df42c39520>,
     <matplotlib.lines.Line2D at 0x2df42c39610>,
     <matplotlib.lines.Line2D at 0x2df42c39700>,
     <matplotlib.lines.Line2D at 0x2df42c397f0>,
     <matplotlib.lines.Line2D at 0x2df42c398e0>,
     <matplotlib.lines.Line2D at 0x2df42c399d0>,
     <matplotlib.lines.Line2D at 0x2df42c39ac0>,
     <matplotlib.lines.Line2D at 0x2df42c39bb0>,
     <matplotlib.lines.Line2D at 0x2df42c39ca0>,
     <matplotlib.lines.Line2D at 0x2df42c39d90>,
     <matplotlib.lines.Line2D at 0x2df42c39e80>,
     <matplotlib.lines.Line2D at 0x2df42c39f70>,
     <matplotlib.lines.Line2D at 0x2df42c410a0>,
     <matplotlib.lines.Line2D at 0x2df42c41190>,
     <matplotlib.lines.Line2D at 0x2df42c41280>,
     <matplotlib.lines.Line2D at 0x2df42c41370>,
     <matplotlib.lines.Line2D at 0x2df42c41460>,
     <matplotlib.lines.Line2D at 0x2df42c41550>,
     <matplotlib.lines.Line2D at 0x2df42c41640>,
     <matplotlib.lines.Line2D at 0x2df42c41730>,
     <matplotlib.lines.Line2D at 0x2df42c41820>,
     <matplotlib.lines.Line2D at 0x2df42c41910>,
     <matplotlib.lines.Line2D at 0x2df42c41a00>,
     <matplotlib.lines.Line2D at 0x2df42c41af0>,
     <matplotlib.lines.Line2D at 0x2df42c41be0>,
     <matplotlib.lines.Line2D at 0x2df42c41cd0>,
     <matplotlib.lines.Line2D at 0x2df42c41dc0>,
     <matplotlib.lines.Line2D at 0x2df42c41eb0>,
     <matplotlib.lines.Line2D at 0x2df42c41fa0>,
     <matplotlib.lines.Line2D at 0x2df42c490d0>,
     <matplotlib.lines.Line2D at 0x2df42c491c0>,
     <matplotlib.lines.Line2D at 0x2df42c492b0>,
     <matplotlib.lines.Line2D at 0x2df42c493a0>,
     <matplotlib.lines.Line2D at 0x2df42c49490>,
     <matplotlib.lines.Line2D at 0x2df42c49580>,
     <matplotlib.lines.Line2D at 0x2df42c49670>,
     <matplotlib.lines.Line2D at 0x2df42c49760>,
     <matplotlib.lines.Line2D at 0x2df42c49850>,
     <matplotlib.lines.Line2D at 0x2df42c49940>,
     <matplotlib.lines.Line2D at 0x2df42c49a30>,
     <matplotlib.lines.Line2D at 0x2df42c49b20>,
     <matplotlib.lines.Line2D at 0x2df42c49c10>,
     <matplotlib.lines.Line2D at 0x2df42c49d00>,
     <matplotlib.lines.Line2D at 0x2df42c49df0>,
     <matplotlib.lines.Line2D at 0x2df42c49ee0>,
     <matplotlib.lines.Line2D at 0x2df42c49fd0>,
     <matplotlib.lines.Line2D at 0x2df42c51100>,
     <matplotlib.lines.Line2D at 0x2df42c511f0>,
     <matplotlib.lines.Line2D at 0x2df42c512e0>,
     <matplotlib.lines.Line2D at 0x2df42c513d0>,
     <matplotlib.lines.Line2D at 0x2df42c514c0>,
     <matplotlib.lines.Line2D at 0x2df42c515b0>,
     <matplotlib.lines.Line2D at 0x2df42c516a0>,
     <matplotlib.lines.Line2D at 0x2df42c51790>,
     <matplotlib.lines.Line2D at 0x2df42c51880>,
     <matplotlib.lines.Line2D at 0x2df42c51970>,
     <matplotlib.lines.Line2D at 0x2df42c51a60>,
     <matplotlib.lines.Line2D at 0x2df42c51b50>,
     <matplotlib.lines.Line2D at 0x2df42c51c40>,
     <matplotlib.lines.Line2D at 0x2df42c51d30>,
     ...]




    
![png](output_6_2.png)
    



```python
# Stacking of the 134 sample across 131,000 data points
plt.plot(rold.transpose())
```




    [<matplotlib.lines.Line2D at 0x2e000a6ebb0>,
     <matplotlib.lines.Line2D at 0x2e000a6ec10>,
     <matplotlib.lines.Line2D at 0x2e000a6ec40>,
     <matplotlib.lines.Line2D at 0x2e000a6ed30>,
     <matplotlib.lines.Line2D at 0x2e000a6ee20>,
     <matplotlib.lines.Line2D at 0x2e000a6ef10>,
     <matplotlib.lines.Line2D at 0x2e000a83040>,
     <matplotlib.lines.Line2D at 0x2e000a83130>,
     <matplotlib.lines.Line2D at 0x2e000a83220>,
     <matplotlib.lines.Line2D at 0x2e000a83310>,
     <matplotlib.lines.Line2D at 0x2e000a6ebe0>,
     <matplotlib.lines.Line2D at 0x2e000a83400>,
     <matplotlib.lines.Line2D at 0x2e000a834f0>,
     <matplotlib.lines.Line2D at 0x2e000a836a0>,
     <matplotlib.lines.Line2D at 0x2e000a83790>,
     <matplotlib.lines.Line2D at 0x2e000a83880>,
     <matplotlib.lines.Line2D at 0x2e000a83970>,
     <matplotlib.lines.Line2D at 0x2e000a83a60>,
     <matplotlib.lines.Line2D at 0x2e000a83b50>,
     <matplotlib.lines.Line2D at 0x2e000a83c40>,
     <matplotlib.lines.Line2D at 0x2e000a83d30>,
     <matplotlib.lines.Line2D at 0x2e000a83e20>,
     <matplotlib.lines.Line2D at 0x2e000a83f10>,
     <matplotlib.lines.Line2D at 0x2e000a8a040>,
     <matplotlib.lines.Line2D at 0x2e000a8a130>,
     <matplotlib.lines.Line2D at 0x2e000a8a220>,
     <matplotlib.lines.Line2D at 0x2e000a8a310>,
     <matplotlib.lines.Line2D at 0x2e000a8a400>,
     <matplotlib.lines.Line2D at 0x2e000a8a4f0>,
     <matplotlib.lines.Line2D at 0x2e000a8a5e0>,
     <matplotlib.lines.Line2D at 0x2e000a8a6d0>,
     <matplotlib.lines.Line2D at 0x2e000a8a7c0>,
     <matplotlib.lines.Line2D at 0x2e000a8a8b0>,
     <matplotlib.lines.Line2D at 0x2e000a8a9a0>,
     <matplotlib.lines.Line2D at 0x2e000a8aa90>,
     <matplotlib.lines.Line2D at 0x2e000a8ab80>,
     <matplotlib.lines.Line2D at 0x2e000a8ac70>,
     <matplotlib.lines.Line2D at 0x2e000a8ad60>,
     <matplotlib.lines.Line2D at 0x2e000a8ae50>,
     <matplotlib.lines.Line2D at 0x2e000a8af40>,
     <matplotlib.lines.Line2D at 0x2e000b16070>,
     <matplotlib.lines.Line2D at 0x2e000b16160>,
     <matplotlib.lines.Line2D at 0x2e000b16250>,
     <matplotlib.lines.Line2D at 0x2e000b16340>,
     <matplotlib.lines.Line2D at 0x2e000b16430>,
     <matplotlib.lines.Line2D at 0x2e000b16520>,
     <matplotlib.lines.Line2D at 0x2e000b16610>,
     <matplotlib.lines.Line2D at 0x2e000b16700>,
     <matplotlib.lines.Line2D at 0x2e000b167f0>,
     <matplotlib.lines.Line2D at 0x2e000b168e0>,
     <matplotlib.lines.Line2D at 0x2e000b169d0>,
     <matplotlib.lines.Line2D at 0x2e000b16ac0>,
     <matplotlib.lines.Line2D at 0x2e000b16bb0>,
     <matplotlib.lines.Line2D at 0x2e000b16ca0>,
     <matplotlib.lines.Line2D at 0x2e000b16d90>,
     <matplotlib.lines.Line2D at 0x2e000b16e80>,
     <matplotlib.lines.Line2D at 0x2e000b16f70>,
     <matplotlib.lines.Line2D at 0x2e000b230a0>,
     <matplotlib.lines.Line2D at 0x2e000b23190>,
     <matplotlib.lines.Line2D at 0x2e000b23280>,
     <matplotlib.lines.Line2D at 0x2e000b23370>,
     <matplotlib.lines.Line2D at 0x2e000b23460>,
     <matplotlib.lines.Line2D at 0x2e000b23550>,
     <matplotlib.lines.Line2D at 0x2e000b23640>,
     <matplotlib.lines.Line2D at 0x2e000b23730>,
     <matplotlib.lines.Line2D at 0x2e000b23820>,
     <matplotlib.lines.Line2D at 0x2e000b23910>,
     <matplotlib.lines.Line2D at 0x2e000b23a00>,
     <matplotlib.lines.Line2D at 0x2e000b23af0>,
     <matplotlib.lines.Line2D at 0x2e000b23be0>,
     <matplotlib.lines.Line2D at 0x2e000b23cd0>,
     <matplotlib.lines.Line2D at 0x2e000b23dc0>,
     <matplotlib.lines.Line2D at 0x2e000b23eb0>,
     <matplotlib.lines.Line2D at 0x2e000b23fa0>,
     <matplotlib.lines.Line2D at 0x2e000b280d0>,
     <matplotlib.lines.Line2D at 0x2e000b281c0>,
     <matplotlib.lines.Line2D at 0x2e000b282b0>,
     <matplotlib.lines.Line2D at 0x2e000b283a0>,
     <matplotlib.lines.Line2D at 0x2e000b28490>,
     <matplotlib.lines.Line2D at 0x2e000b28580>,
     <matplotlib.lines.Line2D at 0x2e000b28670>,
     <matplotlib.lines.Line2D at 0x2e000b28760>,
     <matplotlib.lines.Line2D at 0x2e000b28850>,
     <matplotlib.lines.Line2D at 0x2e000b28940>,
     <matplotlib.lines.Line2D at 0x2e000b28a30>,
     <matplotlib.lines.Line2D at 0x2e000b28b20>,
     <matplotlib.lines.Line2D at 0x2e000b28c10>,
     <matplotlib.lines.Line2D at 0x2e000b28d00>,
     <matplotlib.lines.Line2D at 0x2e000b28df0>,
     <matplotlib.lines.Line2D at 0x2e000b28ee0>,
     <matplotlib.lines.Line2D at 0x2e000b28fd0>,
     <matplotlib.lines.Line2D at 0x2e000b30100>,
     <matplotlib.lines.Line2D at 0x2e000b301f0>,
     <matplotlib.lines.Line2D at 0x2e000b302e0>,
     <matplotlib.lines.Line2D at 0x2e000b303d0>,
     <matplotlib.lines.Line2D at 0x2e000b304c0>,
     <matplotlib.lines.Line2D at 0x2e000b305b0>,
     <matplotlib.lines.Line2D at 0x2e000b306a0>,
     <matplotlib.lines.Line2D at 0x2e000b30790>,
     <matplotlib.lines.Line2D at 0x2e000b30880>,
     <matplotlib.lines.Line2D at 0x2e000b30970>,
     <matplotlib.lines.Line2D at 0x2e000b30a60>,
     <matplotlib.lines.Line2D at 0x2e000b30b50>,
     <matplotlib.lines.Line2D at 0x2e000b30c40>,
     <matplotlib.lines.Line2D at 0x2e000b30d30>,
     <matplotlib.lines.Line2D at 0x2e000b30e20>,
     <matplotlib.lines.Line2D at 0x2e000b30f10>,
     <matplotlib.lines.Line2D at 0x2e000b3f040>,
     <matplotlib.lines.Line2D at 0x2e000b3f130>,
     <matplotlib.lines.Line2D at 0x2e000b3f220>,
     <matplotlib.lines.Line2D at 0x2e000b3f310>,
     <matplotlib.lines.Line2D at 0x2e000b3f400>,
     <matplotlib.lines.Line2D at 0x2e000b3f4f0>,
     <matplotlib.lines.Line2D at 0x2e000b3f5e0>,
     <matplotlib.lines.Line2D at 0x2e000b3f6d0>,
     <matplotlib.lines.Line2D at 0x2e000b3f7c0>,
     <matplotlib.lines.Line2D at 0x2e000b3f8b0>,
     <matplotlib.lines.Line2D at 0x2e000b3f9a0>,
     <matplotlib.lines.Line2D at 0x2e000b3fa90>,
     <matplotlib.lines.Line2D at 0x2e000b3fb80>,
     <matplotlib.lines.Line2D at 0x2e000b3fc70>,
     <matplotlib.lines.Line2D at 0x2e000b3fd60>,
     <matplotlib.lines.Line2D at 0x2e000b3fe50>,
     <matplotlib.lines.Line2D at 0x2e000b3ff40>,
     <matplotlib.lines.Line2D at 0x2e000b46070>,
     <matplotlib.lines.Line2D at 0x2e000b46160>,
     <matplotlib.lines.Line2D at 0x2e000b46250>,
     <matplotlib.lines.Line2D at 0x2e000b46340>,
     <matplotlib.lines.Line2D at 0x2e000b46430>,
     <matplotlib.lines.Line2D at 0x2e000b46520>,
     <matplotlib.lines.Line2D at 0x2e000b46610>,
     <matplotlib.lines.Line2D at 0x2e000b46700>,
     <matplotlib.lines.Line2D at 0x2e000b467f0>,
     <matplotlib.lines.Line2D at 0x2e000b468e0>,
     <matplotlib.lines.Line2D at 0x2e000b469d0>,
     <matplotlib.lines.Line2D at 0x2e000b46ac0>,
     <matplotlib.lines.Line2D at 0x2e000b46bb0>,
     <matplotlib.lines.Line2D at 0x2e000b46ca0>,
     <matplotlib.lines.Line2D at 0x2e000b46d90>,
     <matplotlib.lines.Line2D at 0x2e000b46e80>,
     <matplotlib.lines.Line2D at 0x2e000b46f70>,
     <matplotlib.lines.Line2D at 0x2e000b4c0a0>,
     <matplotlib.lines.Line2D at 0x2e000b4c190>,
     <matplotlib.lines.Line2D at 0x2e000b4c280>,
     <matplotlib.lines.Line2D at 0x2e000b4c370>,
     <matplotlib.lines.Line2D at 0x2e000b4c460>,
     <matplotlib.lines.Line2D at 0x2e000b4c550>,
     <matplotlib.lines.Line2D at 0x2e000b4c640>,
     <matplotlib.lines.Line2D at 0x2e000b4c730>,
     <matplotlib.lines.Line2D at 0x2e000b4c820>,
     <matplotlib.lines.Line2D at 0x2e000b4c910>]




    
![png](output_7_1.png)
    



```python
#The implementation of the standard deviation with z-score and missing values imputation.
#It selects 20 columns/ppm(s) with highest standard deviation from the 134 stacked pituitary tumours samples

roly = pd.DataFrame(rol)
zz_scored = ss.zscore(roly, axis=0)
zz_scores = np.array(zz_scored)

z_thresh = 2.575
out_mask = np.abs(zz_scored) > z_thresh
roly[out_mask] = np.nan

imputery = SimpleImputer(strategy ='mean')
roly = imputery.fit_transform(roly)

stdvec7 = np.nanstd(roly, axis=0)
localmaxidx7, _ = ssig.find_peaks(stdvec7)
idxsorted_stdvec7 = np.flipud(np.argsort(stdvec7[localmaxidx7]))

num_maxima_to_display = 20
print(stdvec7[localmaxidx7[idxsorted_stdvec7[:num_maxima_to_display]]])
print(localmaxidx7[idxsorted_stdvec7[:num_maxima_to_display]])

plt.plot(stdvec7) #plot std. dev.
plt.plot(localmaxidx7[idxsorted_stdvec7[0:20]], stdvec7[localmaxidx7[idxsorted_stdvec7[0:20]]], color='red',
     marker='o', markersize=3, linestyle='none') #plot std. dev. local maxima
plt.xlabel("ppm")
plt.ylabel("std. dev. of spectra over samples")
plt.show()

```

    [5.61242295e+08 5.45868697e+08 3.95869279e+08 3.57237143e+08
     3.45192127e+08 3.42602443e+08 3.40700352e+08 3.38335006e+08
     2.13775636e+08 1.68960898e+08 1.62480846e+08 1.55035152e+08
     1.49333370e+08 1.49082966e+08 1.47710038e+08 1.45889976e+08
     1.44773550e+08 1.43215076e+08 1.41883121e+08 1.38270160e+08]
    [88181 88257 79046 89200 75859 75856 89268 89266 84354 73600 74518 75800
     75547 82602 73186 84093 75807 89196 70019 69943]
    


    
![png](output_8_1.png)
    



```python
#the iterative correlation values between the highest standard deviation column and the subsequent next 20 columns
ass_stdvec7 = stdvec7[localmaxidx7[idxsorted_stdvec7[0:20]]]
srt_std7 = localmaxidx7[idxsorted_stdvec7[0:20]]
cortto = []# initialize an empty list to store the correlation values.
for itht in range(0, len(srt_std7)):# Loop over the range of column indices starting from 1 to the last column index.
        ithc3 = srt_std7[0]# get the column index of the first column
        ithc4 = srt_std7[itht]# Get the column index of the column being considered in loop
        clms3t = roly[:, ithc3].reshape(-1, 1)#obtain the data-info from the first column
        clms4t = roly[:, ithc4].reshape(-1, 1)# data-info from the current column
        cor_mats = np.corrcoef(clms3t, clms4t, rowvar=False)#execute the correlation between the columns
        cortto.append(cor_mats[0, 1:])# save the results of the correlation in the empty list cort.    
plt.figure(figsize=(20, 12))   
plt.plot(range(0, len(srt_std7)), cortto)# plot the stored correlation value in cort against the column index 
plt.xlabel('Column')# Give the x-axis a label name column
plt.ylabel('Correlation')# Give the y-axis the label name correlation
plt.title(' iterative corre. of first column with the next one')# label the plot title
plt.xticks(range(len(srt_std7)), srt_std7, rotation=90)
plt.show()# show the plot
```


    
![png](output_9_0.png)
    



```python

```


```python
# Ranked correlation and correlation matrix of the column with hihest standard deviations

corttz =[]
for ith1 in range(0, len(srt_std7)):# Loop over the range of column indices starting from
        ithc3 = srt_std7[0]# get the column index of the first column
        ithc4 = srt_std7[ith1]# Get the column index of the column being considered in loop
        clms3s = roly[:, ithc3].reshape(-1, 1)#obtain the data-info from the first column
        clms4s = roly[:, ithc4].reshape(-1, 1)# data-info from the current column
        clms3_zz = zscore(clms3s)
        clms4_zz = zscore(clms4s)
        cor_maats = np.corrcoef(clms3_zz, clms4_zz, rowvar=False)
        cor_mats = np.corrcoef(clms3s, clms4s, rowvar=False)#execute the correlation between the columns
        corttz.append(cor_mats[0, 1])# save the results of the correlation in the empty list cort
        
sorted_inn = np.argsort(corttz)[::-1]
srt_std7_srn = np.array(corttz)[sorted_inn]

plt.figure(figsize=(12, 10))
plt.bar(range(1, len(srt_std7) + 1), srt_std7_srn)
plt.xlabel('Colunm')
plt.ylabel('correlarion Coefficient')
plt.title('Ranked Correlation Coefficient with First Column')
plt.xticks(range(1, len(srt_std7) +1), srt_std7[sorted_inn], rotation=90)
plt.show()

fig, ad = plt.subplots(figsize=(20, 12))#give the plot window specification
cor_mat1s = np.corrcoef(roly[:, srt_std7], rowvar=False)#execute the correlation coefficient between the first and the subsequent columns
sbn.heatmap(cor_mat1s, cmap='coolwarm', annot=True, fmt='.2f', square=True)#create heatmap matrix plot
    
plt.title('Correlation Matrix')# state the title of the plot
plt.xlabel('columns')#give the x-axis a name column 
plt.ylabel('Correlation')#give the y-axix a name correlation
plt.xticks(range(len(srt_std7)), srt_std7, rotation=90)#Display the labey from the data-srt_std for x-axis
plt.yticks(range(len(srt_std7)), srt_std7, rotation=0)#Display the labey from the data-srt_std for y-axis
    

```


    
![png](output_11_0.png)
    





    ([<matplotlib.axis.YTick at 0x1be3dc0f400>,
      <matplotlib.axis.YTick at 0x1be3dc0fee0>,
      <matplotlib.axis.YTick at 0x1be3e422ac0>,
      <matplotlib.axis.YTick at 0x1be3e3ddb80>,
      <matplotlib.axis.YTick at 0x1be3d22c460>,
      <matplotlib.axis.YTick at 0x1be3d88dee0>,
      <matplotlib.axis.YTick at 0x1be3d88daf0>,
      <matplotlib.axis.YTick at 0x1be3d5d1c70>,
      <matplotlib.axis.YTick at 0x1be3e483cd0>,
      <matplotlib.axis.YTick at 0x1be41c47640>,
      <matplotlib.axis.YTick at 0x1be3d6cca00>,
      <matplotlib.axis.YTick at 0x1be41c477c0>,
      <matplotlib.axis.YTick at 0x1be3d58cbe0>,
      <matplotlib.axis.YTick at 0x1be3e280250>,
      <matplotlib.axis.YTick at 0x1be3d8a6e50>,
      <matplotlib.axis.YTick at 0x1be3d684370>,
      <matplotlib.axis.YTick at 0x1be3d8a6850>,
      <matplotlib.axis.YTick at 0x1be3d5b98b0>,
      <matplotlib.axis.YTick at 0x1be3d5b9b80>,
      <matplotlib.axis.YTick at 0x1be3e3fe610>],
     [Text(0, 0, '88181'),
      Text(0, 1, '88257'),
      Text(0, 2, '79046'),
      Text(0, 3, '89200'),
      Text(0, 4, '75859'),
      Text(0, 5, '75856'),
      Text(0, 6, '89268'),
      Text(0, 7, '89266'),
      Text(0, 8, '84354'),
      Text(0, 9, '73600'),
      Text(0, 10, '74518'),
      Text(0, 11, '75800'),
      Text(0, 12, '75547'),
      Text(0, 13, '82602'),
      Text(0, 14, '73186'),
      Text(0, 15, '84093'),
      Text(0, 16, '75807'),
      Text(0, 17, '89196'),
      Text(0, 18, '70019'),
      Text(0, 19, '69943')])




    
![png](output_11_2.png)
    



```python
# The pair peaks with maximum silhouette score cluster estimation

def optimized_log(x, c=1):
   return np.log(x + c)

def meta_data(nd, property =''):
    if len(property) == 0:
        return []
    
    n_spc = len(nd.nmrdat[nd.s])
    data_vector = []
    for k in range(134):
        title = nd.nmrdat[nd.s][k].title
        try:
            try:
                idx1 = title.index(property)
            except:
                idx1 = title.index(property.replace(' ', '_'))
                
            sub_str = title[idx1:]
            idx2 = sub_str.index(':')
            idx3 = sub_str.index('\n')
            data_vector.append(sub_str[idx2 + 1:idx3].strip())
        except:
            print(f'property not found! (exp: {k + 1})')
            data_vector.append('not found')
        
        
    return data_vector

tumour_type = 'tumour'
tumour_class = meta_data(nd, 'tumour')
tumour_class = [ 0 if value == 'not found' else value for value in tumour_class]

e = pd.DataFrame({'Tumour_Class':tumour_class})
pd.set_option('display.max_row', None)
pd.set_option('display.max_columns', None)

# the pair peak with maximum silhouette score
ihc1 = 89267
ihc2 = 72046

coc1 = roly[:, ihc1].reshape(-1, 1)
coc2 = roly[:, ihc2].reshape(-1, 1)
datao = np.hstack((coc1, coc2))

silhouette_src = []
for k in range(2, 16):
    kmop = KMeans(n_clusters=k)
    kmop.fit(datao)
    labb = kmop.labels_
    
    sil_op = silhouette_score(datao, labb)
    silhouette_src.append(sil_op)
    
op_kk = np.argmax(silhouette_src) +2

print(f'Optimal Number of Clusters (k): {op_kk}')
print(f'Maximum Silhouette Score: {max(silhouette_src):.4f}')

opt_kmop = KMeans(n_clusters = op_kk)
opt_kmop.fit(datao)
opt_labb = opt_kmop.labels_

cluster_ind1 = np.where(opt_labb ==0)[0]
cluster_ind2 = np.where(opt_labb ==1)[0]

cluster_cl1 = e['Tumour_Class'].iloc[cluster_ind1]
cluster_cl2 = e['Tumour_Class'].iloc[cluster_ind2]

# calculate and display pecentage of tumour cluss belonging to cluter 1 and cluster 2
tot_cta = e['Tumour_Class'].value_counts()
clust1a_per = cluster_cl1.value_counts()
clust2a_per = cluster_cl2.value_counts()

clust1a_percent = (clust1a_per/tot_cta)* 100
clust2a_percent = (clust2a_per/tot_cta)* 100

cluster_val1 = datao[cluster_ind1]
cluster_val2 = datao[cluster_ind2]

print('Percent for ecah Tumor Cluster 1:', clust1a_percent)
print('Percent for each Tumour Cluster 2:', clust2a_percent)

print('Cluster 1 Class:', cluster_cl1.tolist())
print('Cluster 1 Values:', cluster_val1.tolist())

print('Cluster 2 Class:', cluster_cl2.tolist())
print('Cluster 2 Values:', cluster_val2.tolist())


plt.figure(figsize=(10, 8))
plt.scatter(coc1, coc2, c=opt_labb, cmap ='viridis', edgecolor ='k', alpha=0.7)
plt.xlabel(f'Column{ihc1}')
plt.ylabel(f'Column{ihc2}')
plt.title(f'scatter Plot for The pair of columns{ihc1} and {ihc2}\nOptimal k: {op_kk}(Silhouette score:{max(silhouette_src):.4f}')
plt.show()

clust_class1 =  e['Tumour_Class'].iloc[cluster_ind1]
clust_class2 = e['Tumour_Class'].iloc[cluster_ind2]


clust1_tumour_c = clust_class1.value_counts()
clust2_tumour_c = clust_class2.value_counts()


print('Cluster 1 Tumour counts:')
print(clust1_tumour_c)


print('\nCluster 2 Tumour Counts:')
print(clust2_tumour_c)

centd1 = np.mean(cluster_val1, axis=0)
centd2 = np.mean(cluster_val2, axis=0)

centr_df = pd.DataFrame({'Cluster':['Cluster 1', 'Cluster 2'],
                        'Centroid': [centd1, centd2]})


df_clust_cls1 = pd.DataFrame({'Cluster 1': clust_class1})
df_clust_cls2 = pd.DataFrame({'Cluster 2': clust_class2})

plt.figure(figsize=(10, 8))

plt.subplot(1, 3, 1)


plt.boxplot([cluster_val1[:, 0], cluster_val2[:, 0]], labels=['Cluster 1', 'Cluster 2'])

plt.xlabel('Cluster')
plt.ylabel('Centroid Value')
plt.title('Box plot of the Tumour Class vs Centroid Values for Cluster 1 and Cluster 2')

plt.show() 
```

    C:\Users\bosmi\anaconda3\envs\metabolabpy\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\bosmi\anaconda3\envs\metabolabpy\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\bosmi\anaconda3\envs\metabolabpy\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\bosmi\anaconda3\envs\metabolabpy\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\bosmi\anaconda3\envs\metabolabpy\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\bosmi\anaconda3\envs\metabolabpy\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\bosmi\anaconda3\envs\metabolabpy\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\bosmi\anaconda3\envs\metabolabpy\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\bosmi\anaconda3\envs\metabolabpy\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\bosmi\anaconda3\envs\metabolabpy\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\bosmi\anaconda3\envs\metabolabpy\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\bosmi\anaconda3\envs\metabolabpy\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\bosmi\anaconda3\envs\metabolabpy\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\bosmi\anaconda3\envs\metabolabpy\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\bosmi\anaconda3\envs\metabolabpy\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    

    Optimal Number of Clusters (k): 2
    Maximum Silhouette Score: 0.9043
    Percent for ecah Tumor Cluster 1: Tumour_Class
    0    100.000000
    1    100.000000
    2     92.307692
    3    100.000000
    4     80.000000
    5     80.952381
    6     66.666667
    7     78.571429
    Name: count, dtype: float64
    Percent for each Tumour Cluster 2: Tumour_Class
    0          NaN
    1          NaN
    2     7.692308
    3          NaN
    4    20.000000
    5    19.047619
    6    33.333333
    7    21.428571
    Name: count, dtype: float64
    Cluster 1 Class: ['2', '3', '3', '5', '2', '0', '2', '1', '2', '1', '1', '4', '2', '2', '7', '2', '2', '5', '7', '5', '1', '2', '2', '5', '2', '2', '2', '6', '5', '6', '4', '2', '3', '2', '7', '2', '7', '2', '2', '2', '7', '1', '7', '5', '1', '5', '2', '3', '7', '2', '2', '2', '7', '1', '0', '1', '2', '3', '2', '1', '5', '7', '2', '1', '2', '2', '5', '2', '1', '2', '2', '2', '2', '2', '1', '5', '2', '1', '3', '5', '2', '5', '2', '1', '1', '5', '4', '2', '2', '3', '1', '6', '2', '2', '6', '0', '3', '1', '2', '2', '0', '3', '4', '2', '5', '2', '3', '5', '1', '5', '7', '2', '2', '5', '1', '2', '3', '1', '1', '7']
    Cluster 1 Values: [[13863971.8050107, 15189628.799498798], [16442099.125383694, 32822967.350307394], [18222025.50091495, 20322851.62561161], [17271603.66368365, 21457206.41458716], [5764291.352393599, 15134191.044425717], [8711573.312902832, 17259475.56537524], [7103980.468533734, 16454208.180280566], [10406032.096843775, 19237692.68082935], [7766165.742448874, 11031761.4006237], [8217088.675597327, 16279597.574824637], [10826266.972089075, 16006640.586486408], [8268869.248840679, 16623987.935883617], [9371155.214088298, 20485160.63266485], [8300112.734685306, 21126914.84239656], [11012593.253135057, 21652725.166085716], [10284799.294949574, 17630718.820643157], [5994381.29680126, 15977666.267184502], [6830090.157830605, 19934266.390236523], [7137511.371295761, 20537026.755032226], [5944945.826476522, 10907769.108844843], [7832368.824374115, 25920797.470273674], [24439486.51572211, 12632073.657548899], [12119001.749450168, 14369378.45723858], [6891524.876436359, 16366681.466040773], [8791795.155071277, 15510735.642941628], [9991546.150994029, 14579711.547024887], [8949237.551783476, 25049732.637208182], [7033484.376081921, 20429324.459671076], [8469680.846775033, 17019787.662609342], [5717359.565642273, 20100947.025880877], [53002781.08757884, 33111035.88426525], [16125447.783797162, 19323658.971193925], [19487026.373218294, 14719111.08500646], [7916339.016108703, 9975990.90271814], [8521849.27528112, 20826821.133142874], [6724828.031862988, 21066327.771295298], [10518231.395976651, 35591826.06482039], [6620685.773570173, 12311015.42264579], [7428670.720994984, 16725470.557233714], [7982071.2722167, 10906551.660794405], [7501560.15189448, 15496524.298620556], [13196802.673122142, 17158567.794642963], [12627121.765125394, 14579224.13311564], [13783343.451124528, 13982993.193696925], [13871068.6256408, 25766924.359874867], [8637931.333192624, 24648172.44155433], [9314546.094307331, 17488355.795614257], [6948995.982980858, 15285563.231401902], [6812175.717468705, 13754210.056437116], [4789900.991494768, 19057537.17802656], [5648831.354142778, 18065187.63203572], [13051996.541797882, 15607815.569689557], [28334755.915225327, 19687623.79632211], [14909179.509679519, 17042043.824445944], [6685560.435873204, 12424674.466900188], [4880668.650111585, 20089574.06661828], [7504042.235792685, 14569497.278654823], [13843963.517348668, 17019787.662609342], [21374356.706083685, 25836993.90973647], [13297002.95210762, 29246169.259367563], [8330413.217999031, 19946745.893908683], [21934291.482378844, 16437020.590992684], [8310223.237953813, 24165456.986376997], [67442632.57156684, 17019787.662609342], [8278212.419587369, 18039007.00822752], [11377598.314968077, 21545579.064431585], [7635364.447027369, 10984288.053729367], [6417628.157622672, 13225358.162774844], [7784677.8125746455, 12774193.421440754], [8670892.958390469, 14347058.568163186], [7396884.145457107, 14347927.178028643], [15106745.668606522, 11539967.927964827], [12684434.874736508, 16048946.93724507], [13128163.365510782, 16511719.708859013], [14593786.194527024, 18635360.26837118], [12139299.077830793, 25062265.765749622], [5773625.86724279, 19986193.901652735], [7332389.792383912, 17128291.308913723], [9209438.434563553, 17631579.174396813], [14090950.301086186, 13664532.722202005], [13722066.100092696, 23352055.092241317], [7990478.553743461, 20594886.82323348], [11860929.765277477, 12066631.095307192], [10676781.892662466, 13190085.234658409], [10153387.450283192, 18091772.67240347], [13229693.091063844, 11789570.70180269], [171459499.5324959, 28225717.125737146], [8243270.729361766, 15011691.295839835], [36644184.26995285, 27779491.705909897], [14758628.428526634, 25021109.113004018], [9795310.599266618, 12631548.491785157], [11681568.712479332, 13830743.254547706], [16409559.091994097, 14047046.835526658], [6937498.463637665, 13318934.313720306], [28826994.176191296, 17019787.662609342], [77546024.81672685, 19863980.095458783], [32078904.142128382, 21787443.19013739], [9730967.885310743, 17307628.84180137], [7625607.912549093, 11704745.282198347], [7981703.508634636, 11744551.145035762], [193599525.3237176, 17019787.662609342], [14026852.244617283, 15342392.509223219], [17698481.271948896, 13743925.205591766], [9430896.894709157, 12739092.656680811], [16003261.245377451, 15116566.914371075], [16761364.501468522, 13107616.230799178], [7796325.169983464, 13037442.185857028], [15372129.408543468, 15247828.016374476], [28701772.35065322, 14765469.860999858], [18596806.148133058, 15075933.427999362], [12043784.335001696, 18329111.395773456], [133410873.02772246, 29235499.005711675], [387701462.9591123, 14145376.474909393], [425970131.17821395, 11505019.840067286], [133410873.02772246, 10268842.347064482], [339896911.7895915, 17640959.946768645], [133410873.02772246, 18007032.797510542], [488894559.5531835, 14245761.07613577], [133410873.02772246, 7000750.236457533], [133410873.02772246, 15527960.354595855]]
    Cluster 2 Class: ['7', '7', '5', '5', '7', '2', '5', '2', '6', '6', '2', '4', '5', '2']
    Cluster 2 Values: [[1001472035.8759032, 9675602.619644044], [1325596859.6986055, 18459787.75079407], [844187163.4318571, 6111752.772070049], [830478466.9210824, 15822724.382129045], [1276613648.175096, 16428454.271167034], [576848373.0928175, 10161188.813514678], [1062585024.4621333, 17281173.586640947], [669563959.5366851, 10218616.301592574], [816479050.0897835, 13768077.865047172], [999288001.2366556, 12594358.695012797], [592790644.8946917, 9308544.54358248], [1543440919.4739118, 16021185.026508696], [1024601910.5011135, 7618571.136243448], [1209254809.2858615, 19241564.030465964]]
    


    
![png](output_12_2.png)
    


    Cluster 1 Tumour counts:
    Tumour_Class
    2    48
    1    21
    5    17
    3    11
    7    11
    0     4
    4     4
    6     4
    Name: count, dtype: int64
    
    Cluster 2 Tumour Counts:
    Tumour_Class
    5    4
    2    4
    7    3
    6    2
    4    1
    Name: count, dtype: int64
    


    
![png](output_12_4.png)
    



```python
# Calculate distances to centroids for Cluster 1
distan_clus1 = np.sqrt(np.sum((cluster_val1 - centd1) ** 2, axis=1))

# Calculate distances to centroids for Cluster 2
distan_clus2 = np.sqrt(np.sum((cluster_val2 - centd2) ** 2, axis=1))

# Create DataFrames for each cluster with distances and cluster labels
df_clust_cls1 = pd.DataFrame({'Cluster Label': cluster_cl1, 'Distance to Centroid': distan_clus1})
df_clust_cls2 = pd.DataFrame({'Cluster Label': cluster_cl2, 'Distance to Centroid': distan_clus2})

# Plot the box plot
plt.figure(figsize=(10, 8))
sbn.boxplot(data=pd.concat([df_clust_cls1, df_clust_cls2]), x='Cluster Label', y='Distance to Centroid')
plt.xlabel('Cluster Label')
plt.ylabel('Distance to Centroid')
plt.title('Box Plot of Distance to Centroid for Clusters')
plt.show()
```


    
![png](output_13_0.png)
    



```python
distances_cluster1 = [euclidean(point, centd1) for point in cluster_val1]
distances_cluster2 = [euclidean(point, centd2) for point in cluster_val2]

# Create DataFrames for each cluster with distances and cluster labels
df_clust_cls1 = pd.DataFrame({'Cluster': 'Cluster 1', 'Tumor Class': cluster_cl1, 'Distance to Centroid': distances_cluster1})
df_clust_cls2 = pd.DataFrame({'Cluster': 'Cluster 2', 'Tumor Class': cluster_cl2, 'Distance to Centroid': distances_cluster2})

# Concatenate the DataFrames
df_combined = pd.concat([df_clust_cls1, df_clust_cls2])

# Plot the box plot
plt.figure(figsize=(10, 8))
sbn.boxplot(data=df_combined, x='Cluster', y='Distance to Centroid', hue='Tumor Class')
plt.xlabel('Cluster')
plt.ylabel('Distance to Centroid')
plt.title('Box Plot of Distance to Centroid for Clusters')
plt.show()
```


    
![png](output_14_0.png)
    



```python
# Calculate distances for each data point to centroids
distances = []
for point in datao:
    distance1 = np.linalg.norm(point - centd1)
    distance2 = np.linalg.norm(point - centd2)
    distances.append((distance1, distance2))

# Create a DataFrame for distances
df_distances = pd.DataFrame(distances, columns=['Distance to Centroid 1', 'Distance to Centroid 2'])

# Add Tumor Class column from the original DataFrame
df_distances['Tumor Class'] = e['Tumour_Class']

# Calculate p-values for each tumor class
p_values = []
for tumor_class in np.unique(df_distances['Tumor Class']):
    distances_class = df_distances.loc[df_distances['Tumor Class'] == tumor_class]
    _, p_value = ranksums(distances_class['Distance to Centroid 1'], distances_class['Distance to Centroid 2'])
    p_values.append((tumor_class, p_value))

# Display the p-values
for tumor_class, p_value in p_values:
    print(f'Tumor Class: {tumor_class}, p-value: {p_value:.4f}')
```

    Tumor Class: 0, p-value: 0.0209
    Tumor Class: 1, p-value: 0.0000
    Tumor Class: 2, p-value: 0.0000
    Tumor Class: 3, p-value: 0.0001
    Tumor Class: 4, p-value: 0.1172
    Tumor Class: 5, p-value: 0.0000
    Tumor Class: 6, p-value: 0.1093
    Tumor Class: 7, p-value: 0.0115
    


```python
# Calculating the distances from the centroid 1
plt.figure(figsize=(10, 8))
sbn.boxplot(data=df_distances, x='Tumor Class', y='Distance to Centroid 1')
plt.xlabel('Tumor Class')
plt.ylabel('Distance to Centroid 1')
plt.title('Box Plot of Distances to Centroid 1 for Tumor Classes')
plt.xticks(rotation=45)
plt.show()

#Calculating the distances from centroid 2
plt.figure(figsize=(10, 8))
sbn.boxplot(data=df_distances, x='Tumor Class', y='Distance to Centroid 2')
plt.xlabel('Tumor Class')
plt.ylabel('Distance to Centroid 2')
plt.title('Box Plot of Distances to Centroid 2 for Tumor Classes')
plt.xticks(rotation=45)
plt.show()
```


    
![png](output_16_0.png)
    



    
![png](output_16_1.png)
    



```python
#Clculation of the adjusted p Value
p_values = [
    (0, 0.0209),    # Replace nan with 1.0 for calculation
    (1, 0.0001),    # Replace nan with 1.0 for calculation
    (2, 0.0001),
    (3, 0.0001),    # Replace nan with 1.0 for calculation
    (4, 0.1172),
    (5, 0.0001),
    (6, 0.1093),
    (7, 0.0115)
]
# Number of tests
num_tests = len(p_values)

# Bonferroni corrected significance level
alpha_corrected = 0.05

# Calculate adjusted p-values based on Bonferroni correction
adjusted_p_values = [(tumor_class, min(1, p_value * num_tests)) for tumor_class, p_value in p_values]

# Compare adjusted p-values to corrected significance level and display results
for tumor_class, adjusted_p_value in adjusted_p_values:
    if adjusted_p_value < alpha_corrected:
        print(f"Tumor Class {tumor_class}: Significant (adjusted p-value = {adjusted_p_value:.4f})")
    else:
        print(f"Tumor Class {tumor_class}: Not significant (adjusted p-value = {adjusted_p_value:.4f})")
```

    Tumor Class 0: Not significant (adjusted p-value = 0.1672)
    Tumor Class 1: Significant (adjusted p-value = 0.0008)
    Tumor Class 2: Significant (adjusted p-value = 0.0008)
    Tumor Class 3: Significant (adjusted p-value = 0.0008)
    Tumor Class 4: Not significant (adjusted p-value = 0.9376)
    Tumor Class 5: Significant (adjusted p-value = 0.0008)
    Tumor Class 6: Not significant (adjusted p-value = 0.8744)
    Tumor Class 7: Not significant (adjusted p-value = 0.0920)
    


```python
#This Indicates that tumour classes 1, 2, 3, and 5 corresponding to Acromegaly, Gonadotroph, Cushin's and Null Cell have dinstint characteristics metabolic profile

```
