#!/usr/bin/python

import cProfile, random, LSTM_g

def genSpecString(inputs, outputs, mblocks):
    specList = []
    specList.append("%d, %d, 1, 1" % (inputs, outputs))
    for memoryBlock in range(mblocks):
        specList.append("\n%d, 1, 1, 1" % memoryBlock)
    for memoryBlock in range(mblocks):
        specList.append("\n%d, %d, 2" % (memoryBlock,  memoryBlock))
    specList.append("\n0, %d" % mblocks)
    return ''.join(specList)
    

def trainNet():

    maxtries =  1000
    success = 0
    specString = genSpecString(1, 1, 1)
    #print specString
    xorNet = LSTM_g.LSTM_g(specString)
    #print xorNet.toString(False)
    in1 = 0
    
    for i in xrange(0, maxtries):
        in2 = in1
        in1 = random.randrange(2)
        out1 = 0
        if in1 != in2:
            out1 = 1
        result = xorNet.step([in1])[0]
        error = xorNet.getError([out1])
        xorNet.learn([out1])
        print in1, in2, out1, result, error
        if abs(out1 - result) < 0.5:
            success = success + 1
    print '%d success tries' % success


#cProfile.run('trainNet()')
trainNet()
