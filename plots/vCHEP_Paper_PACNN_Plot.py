
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import (MultipleLocator, 
                               FormatStrFormatter, 
                               AutoMinorLocator) 
import mplhep as hep

## The data values correspond to the following asymmetry values:
## 0.0,0.5,1.5,2.0,2.5,3.0,5.0,7.5,9.0*
## *9.0 is present only in ACN_4i4_8L_DenseNet_BN

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

# Model X (P)
# ACN_4i4_P_10L_4S_BN
x = [89.9,92.3,94.0,94.3,94.6,94.7,95.2,95.3]
y = [.043,.073,.122,.140,.154,.166,.205,.243]
g4 = (x,y)

# Model Y (P)
# ACN_4i4_8L_DenseNet_BN
x = [90.6,92.5,93.9,94.3,94.5,94.7,95.2,95.4,95.4]
y = [.048,.076,.112,.126,.138,.151,.183,.213,.232]
g5 = (x,y)

# set data parameters
data = (g1, g2, g3, g4, g5)
groups = ("BM_ACN_4i4_P_6L (10966)", "ACN_4i4_P_6L_1S_BN_RC1 (11975)", 
        "ACN_4i4_P_8L_3S (18719)", "ACN_4i4_P_10L_4S_BN (19646)", 
        "ACN_4i4_8L_DenseNet_BN (41983)")
colors = ("black", "blue", "red", "green", "purple")
markers = ("D", "s", "o", "v", "^")

# set style to CERN standard
plt.style.use(hep.style.ROOT)
# set size of figure
fig = plt.figure(num=1, figsize=(12,8), dpi=80, facecolor='w', edgecolor='k')
# set x-axis ticks to have 0 decimal places
ax = fig.add_subplot(111)
# set x-axis ticks to have 0 decimal places
ax.xaxis.set_major_formatter(FormatStrFormatter('% 1.1f')) 

# loop thorugh data parameters and plot points on plot (scatter)
for data, color, mi, group in zip(data, colors, markers, groups):
    x, y = data
    ax.plot(x, y, alpha=0.8, c=color, marker=mi, linestyle='none', markersize=13, label=group)
    plt.legend(loc=2, fontsize=23)

# adjust font size of ticks
ax.tick_params(axis='both', which='major', labelsize=23, pad=8)

#plt.grid(which='major', axis='both', alpha=.1)
# set limits to x and y axes
plt.xlim(92,96)
plt.ylim(.04,.35)
# Toy MC Simulation disclaimer
plt.text(.97,0.07,'Toy MC Simulation',horizontalalignment='right',
     verticalalignment='top', transform = ax.transAxes, fontsize=18)
# add x and y labels
plt.xlabel("Efficiency (%)", fontsize=23, horizontalalignment='center',
    verticalalignment='center', labelpad=20)
plt.ylabel("False Positive Rate (per event)", fontsize=23, horizontalalignment='center',
    verticalalignment='center', labelpad=20)
plt.tight_layout()
plt.show()
