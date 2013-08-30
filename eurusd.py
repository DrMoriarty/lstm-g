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

# standard net 2d*3v maxtries = 500
# in=6 mb=13 -> sq = 0.0102 0142 0155 0154 0146 | 01398
# in=6 mb=14 -> sq = 0.0139 0139 0155 0089 0136 | 01316
# in=6 mb=15 -> sq = 0.0148 0088 0161 0156 0099 | 01304
# in=6 mb=16 -> sq = 0.0100 0199 0150 0105 0127 | 01362
# in=6 mb=17 -> sq = 0.0126 0117 0121 0103 0102 | 01138
# in=6 mb=18 -> sq = 0.0139 0156 0210 0155 0126 | 01572
# in=6 mb=19 -> sq = 0.0124 0121 0162 0124 0112 | 01286
# in=6 mb=20 -> sq = 0.0133 0141 0128 0108 0141 | 01302
# in=6 mb=21 -> sq = 0.0106 0108 0129 0105 0114 | 01124  /__  inp * 3,5
# in=6 mb=22 -> sq = 0.0117 0138 0111 0141 0063 | 01140  \
# in=6 mb=23 -> sq = 0.0131 0114 0094 0129 0140 | 01216
# in=6 mb=24 -> sq = 0.0122 0117 0132 0130 0144 | 01290
# in=6 mb=25 -> sq = 0.0105 0107 0144 0124 0130 | 01220

inputs = 30
memBlocks = inputs*2+1
stochPeriod = 5
maxtries =  0
specString = str(inputs)+", 1, 1, 1"
for memoryBlock in range(memBlocks):
    specString += "\n" + str(memoryBlock) + ", 1, 1, 1"
for memoryBlock in range(memBlocks):
    specString += "\n" + str(memoryBlock) + ", " + str(memoryBlock) + ", 2"
specString += "\n0, " + str(memBlocks)
print specString
# netT = hi-lo
netT = LSTM_g.LSTM_g(specString)
# netM = (hi+lo+cl)/3
netM = LSTM_g.LSTM_g(specString)
# netC = cl
netC = LSTM_g.LSTM_g(specString)
#print net.toString(True)

with open('EURUSD1440.csv', 'rb') as csvfile:
    with open('netresult_30d.csv', 'wb') as outfile:
        reader = csv.reader(csvfile, delimiter=',')
        writer = csv.writer(outfile, delimiter=';')
        writer.writerow(['Date', 'Open', 'High', 'Low', 'Close', 'Predicted Hi-Lo', 'Predicted Median Price', 'Predicted Close'])
        resultT = 0.0
        resultM = 0.0
        resultC = 0.0
        args = deque([])
        errsum = 0.0
        numtries = 0
        volseq = deque([])
        for row in reader:
            date = row[0]
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
            if resultT != 0.0 and resultM != 0.0 and resultC != 0.0:
                writer.writerow([date, op*2.0, hi*2.0, lo*2.0, cl*2.0, resultT*2.0, resultM*2.0, resultC*2.0])
                target = cl
                #error = netC.getError([target])
                netT.learn([hi-lo])
                netM.learn([(hi+lo+cl)/3.0])
                netC.learn([cl])
                numtries += 1
                #errsum += (prevresult - target)**2
                #print error, prevresult - target, math.sqrt(errsum/numtries)
                print numtries
            
            args.append(hi)
            args.append(lo)
            args.append(cl)
            
            while len(args) > inputs:
                args.popleft()
            if len(args) == inputs:
                resultT = netT.step(args)[0]
                resultM = netM.step(args)[0]
                resultC = netC.step(args)[0]

            if maxtries != 0 and numtries >= maxtries:
                break
