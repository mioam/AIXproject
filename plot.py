import torch
import matplotlib.pyplot as plt
import numpy as np

NAME = 'rand'
hist = torch.load(f'_hist{NAME}.pt')

print(len(hist))
n = 100
s = [[np.zeros(n) for x in range(2)] for y in range(2)]
T = 0
F = 0
for x in hist:
    if x[0] >= n:
        t = n - 1
    else:
        t = x[0]
    s[x[1]][x[2]][t] += 1
    if x[1] == x[2]:
        T += 1
    else:
        F += 1

print(T, F)

plt.plot(s[0][0]/(s[0][0] + s[1][0]),alpha=0.7,label='00/(00+10)')
plt.plot(s[1][1]/(s[0][1] + s[1][1]),alpha=0.7,label='11/(01+11)')

plt.ylim(0,1)
plt.legend()
plt.savefig(f'a{NAME}.png', dpi=500)
plt.cla()

recall = []
precision = []
for i in range(n):
    TP = s[0][0][:i].sum()
    FP = s[0][1][:i].sum()
    FN = F - FP
    TN = T - TP
    # print(TP, FP, TN, FN)
    recall.append(0 if TP == 0 else TP/(TP+FN))
    precision.append(0 if TP == 0 else TP/(TP+FP))
plt.plot(recall,alpha=0.7,label='recall')
plt.plot(precision,alpha=0.7,label='precision')

plt.ylim(0,1)
plt.legend()
plt.savefig(f'b{NAME}.png', dpi=500)
plt.cla() 

