
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## The data values correspond to the following asymmetry values:
## 0.0,0.5,1.5,2.0,2.5,3.0,5.0,7.5

# Model A
# BM_ACN_2i4_6L
x = [.909,.923,.936,.939,.941,.942,.945,.947]
y = [.071,.101,.149,.168,.185,.200,.247,.301]
g1 = (x,y)

# Model C
# ACN_2i4_6L_1S
x = [.906,.923,.937,.940,.942,.943,.947,.949]
y = [.064,.098,.156,.174,.190,.207,.263,.325]
g2 = (x,y)

# Model E
# ACN_2i4_6L_1S_BN_RC1
x = [.907,.924,.938,.942,.944,.945,.948,.949]
y = [.067,.101,.157,.183,.201,.219,.278,.337]
g3 = (x,y)

# Model G
# ACN_2i4_6L_RC2
x = [.906,.921,.934,.937,.940,.941,.945,.947]
y = [.068,.099,.154,.170,.198,.207,.278,.346]
g4 = (x,y)

# Model I
# ACN_2i4_6L_RC3
x = [.895,.912,.927,.930,.933,.935,.940,.942]
y = [.059,.089,.141,.159,.181,.199,.257,.326]
g5 = (x,y)

# Model L
# ACN_2i4_6L_RC4
x = [.902,.918,.932,.935,.938,.940,.943,.945]
y = [.065,.095,.149,.172,.194,.214,.273,.346]
g6 = (x,y)

# Model W
# ACN_2i4_8L_3S
x = [0.910,0.926,0.938,0.941,0.943,0.945,0.948,0.950]
y = [0.095,0.097,0.141,0.162,0.176,0.191,0.244,0.289]
g7 = (x,y)


data = (g1, g2, g3, g4, g5, g6, g7)

groups = ("BM_ACN_2i4_6L", "ACN_2i4_6L_1S", 
        "ACN_2i4_6L_1S_BN_RC1", "ACN_2i4_6L_RC2", 
        "ACN_2i4_6L_RC3", "ACN_2i4_6L_RC4", "ACN_2i4_8L_3S")

colors = ("black", "blue", "red", "green", "purple", "magenta", "orange")
markers = ("D", "s", "o", "v", "^", "P", "X")
fig = plt.figure()
ax = fig.add_subplot(111)

for data, color, mi, group in zip(data, colors, markers, groups):
    x, y = data
    ax.scatter(x, y, alpha=0.8, c=color, marker=mi, s=30, label=group)
    plt.legend(loc=2)

# Toy MC Simulation disclaimer
plt.text(0.1,0.5,'Toy MC Simulation',horizontalalignment='left',
     verticalalignment='center', transform = ax.transAxes)
plt.xlabel("Efficiency (%)")
plt.ylabel("False Positive Rate (per event)")
plt.title('Varied Loss Function Coefficient of AllCNN Models')
plt.tight_layout()
plt.show()
