#!/usr/bin/python

import LSTM_g
import csv
from collections import deque
import math

#the network architecture used in the first Distracted Sequence Recall experiment
#numInputs is 11 instead of 10 because the last "input unit" is a bias unit

# inp = 3d*1v -> sq = 0.060   - cl
# inp = 4d*1v -> sq = 0.055  /

# inp = 4d*2v -> sq = 0.053  - cl (hi-lo)

# inp = 4d*3v -> sq = 0.035  \
# inp = 3d*3v -> sq = 0.037   - cl lo hi
# inp = 2d*3v -> sq = 0.043   |
# inp = 1d*3v -> sq = 0.059  /

# inp = 1d*4v -> sq = 0.059    - cl lo hi stochVol
# inp = 2d*4v -> sq = 0.044   |
# inp = 3d*4v -> sq = 0.038  /

# 500 tries 
# inp = 4d*3v - cl lo hi  
# 0, 0 -> sq = 0.067
# 1, 1 -> sq = 0.043
# 1, 0 -> sq = 0.044
# 0, 1 -> sq = 0.067
# conType = 0 -> sq = 0.043
# conType = 1 -> sq = 0.045
# conType = 2 -> sq = 0.042
# Nmb = in*2+1 (25) -> sq = 0.042
# Nmb = in+1 (13) -> sq = 0.051
# Nmb = in*3+1 (37) ->sq = 0.047
# Nmb = (19) -> sq = 0.049
# Nmb = (22) -> sq = 0.048
# Nmb = (31) -> sq = 0.042
# Nmb = (28) -> sq = 0.044

# mean price (C+L+H)/3
# 4d*3v -> sq = 0.011  - hi, lo, cl
# 5d*3v -> sq = 0.006  /
# 4d*4v -> sq = 0.012  - hi, lo, cl, volStoch

# close price
# 5d*3v -> sq = 0.0099 - hi, lo, cl

inputs = 15
memBlocks = inputs*2+1
stochPeriod = 5
maxtries = 500
specString = str(inputs)+", 1, 1, 1"
for memoryBlock in range(memBlocks):
    specString += "\n" + str(memoryBlock) + ", 1, 1, 1"
for memoryBlock in range(memBlocks):
    specString += "\n" + str(memoryBlock) + ", " + str(memoryBlock) + ", 2"
specString += "\n0, " + str(memBlocks)
print specString
net = LSTM_g.LSTM_g(specString)
#print net.toString(True)

with open('EURUSD1440.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    prevresult = 0.0
    prevclose = 0.0
    args = deque([])
    errsum = 0.0
    numtries = 0
    volseq = deque([])
    for row in reader:
        #print ' '.join(row)
        op = float(row[2]) * 0.5
        hi = float(row[3]) * 0.5
        lo = float(row[4]) * 0.5
        cl = float(row[5]) * 0.5
        vol = float(row[6])
        #volseq.append(vol)
        #while len(volseq) > stochPeriod:
        #    volseq.popleft()
        #volHi = vol
        #volLo = vol
        #for vv in volseq:
        #    if volHi < vv:
        #        volHi = vv
        #    if volLo > vv:
        #        volLo = vv
        #volStoch = 0.0
        #if volHi != volLo:
        #    volStoch = (vol - volLo) / (volHi - volLo)
        if prevresult != 0.0:
            target = cl
            error = net.getError([target])
            net.learn([target])
            numtries += 1
            errsum += (prevresult - target)**2
            print error, prevresult - target, math.sqrt(errsum/numtries)
        #args.append(op)
        args.append(hi)
        args.append(lo)
        args.append(cl)
        #args.append(vol)
        #args.append(hi-lo)
        #args.append(volStoch)
        while len(args) > inputs:
            args.popleft()
        if len(args) == inputs:
            result = net.step(args)
            prevresult = result[0]
        prevclose = cl
        if maxtries != 0 and numtries >= maxtries:
            break
