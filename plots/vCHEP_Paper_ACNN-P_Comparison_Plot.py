
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import (MultipleLocator, 
                               FormatStrFormatter, 
                               AutoMinorLocator) 

## The data values correspond to the following asymmetry values:
## 0.0,0.5,1.5,2.0,2.5,3.0,5.0,7.5

##
## PERTURBATIVE MODELS
##
# Model A (P)
# BM_ACN_4i4_P_6L
x = [90.7,92.2,93.7,94.0,94.2,94.4,94.8,95.0]
y = [.061,.094,.140,.158,.173,.189,.237,.280]
g1 = (x,y)

# Model E (P)
# ACN_4i4_P_6L_1S_BN_RC1
x = [90.6,92.3,93.9,94.2,94.4,94.6,94.9,95.0]
y = [.064,.102,.146,.164,.177,.193,.247,.296]
g2 = (x,y)

# Model W (P)
# ACN_4i4_P_8L_3S
x = [91.1,92.8,94.1,94.3,94.5,94.7,95.0,95.2]
y = [.061,.093,.138,.156,.168,.185,.224,.264]
g3 = (x,y)

# set data parameters
data = (g1, g2, g3)
groups = ("BM_ACN_4i4_P_6L", "ACN_4i4_P_6L_1S_BN_RC1", 
        "ACN_4i4_P_8L_3S")
colors = ("black", "blue", "red")
markers = ("D", "s", "o")

# set size of figure
fig = plt.figure(num=1, figsize=(7,7), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)
# set x-axis ticks to have 0 decimal places
ax.xaxis.set_major_formatter(FormatStrFormatter('% 1.0f')) 

# loop thorugh data parameters and plot points on plot (scatter)
for data, color, mi, group in zip(data, colors, markers, groups):
    x, y = data
    ax.plot(x,y,c=color,marker=mi,linestyle='none',markersize=9,label=group)
    ax.legend(loc='upper left', fontsize=12)

##
## NON-PERTURBATIVE MODELS
##
# Model A
# BM_ACN_2i4_6L
x = [90.9,92.3,93.6,93.9,94.1,94.2,94.5,94.7]
y = [.071,.101,.149,.168,.185,.200,.247,.301]
g1 = (x,y)

# Model E
# ACN_2i4_6L_1S_BN_RC1
x = [90.7,92.4,93.8,94.2,94.4,94.5,94.8,94.9]
y = [.067,.101,.157,.183,.201,.219,.278,.337]
g2 = (x,y)

# Model W
# ACN_2i4_8L_3S
x = [91.0,92.6,93.8,94.1,94.3,94.5,94.8,95.0]
y = [0.095,0.097,0.141,0.162,0.176,0.191,0.244,0.289]
g3 = (x,y)

# set data parameters
data = (g1, g2, g3)
groups = ("BM_ACN_2i4_6L", "ACN_2i4_6L_1S_BN_RC1", 
        "ACN_2i4_8L_3S")
colors = ("black", "blue", "red")
markers = ("D", "s", "o")

# loop thorugh data parameters and plot points on plot (scatter) (same plot as above)
for data, color, mi, group in zip(data, colors, markers, groups):
    x, y = data
    ax.plot(x,y,c=color,marker=mi,fillstyle='none',linestyle='none',markersize=9,label=group)
    ax.legend(loc='upper left', fontsize=12)

# adjust font size of ticks
ax.tick_params(axis='both', which='major', labelsize=14)

#plt.grid(which='major', axis='both', alpha=.1)
# set limits to x and y axes
plt.xlim(92,96)
plt.ylim(.04,.35)
# add x and y labels
plt.xlabel("Efficiency (avg. over 10 epochs)", fontsize=16)
plt.ylabel("False Positive Rate (avg. over 10 epochs)", fontsize=16)
plt.tight_layout()
plt.show()