import os, csv, glob
import argparse

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


from datetime import datetime
import pandas as pd
import logging
from pathlib import Path

DATAPATH = Path('./SC2A2_training/')
TESTLOGFILE = Path('SC2A2_testlog.csv')

current_date = datetime.now().strftime("%Y%m%d")
training_file = os.path.basename(DATAPATH).split('.')[0]
log_filename = f'{current_date}_{training_file}_ADFtrainning.log'

# Remove existing log file if it exists
if os.path.exists(log_filename):
    os.remove(log_filename)

logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def smooth(x,window_len=10,window='hanning'):

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(),s,mode='valid')
    return np.round(y, decimals = 3)

def consecutiveSum(arr, window_len):
    if arr.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    arrSize = arr.size

    if arrSize < window_len:
        length = arrSize
    length = window_len
    maxSum = np.float64(1.0)
    for i in range(length):
        maxSum += arr[i]
    windowSum = maxSum
    for i in range(length,arrSize):
        windowSum += arr[i] - arr[i - length]
        maxSum = np.maximum(maxSum, windowSum)
    return maxSum

def labelSteps(datas, startPt = 30, rateTh = 0.3, width_LB = 15, avgRate_LB = 0.8):
    
    #if len(datas) >= 10:
    #	datas = smooth(datas)
    dataDiffs = np.diff(datas)

    listOfSteps = []
    inStep = False
    stepL = 0
    stepR = 0
    
    for cnt, diff in enumerate(dataDiffs):
        if cnt < startPt:
            continue
        if not inStep and diff >= rateTh:
            stepL = cnt
            inStep = True
            continue
        if inStep and (diff < rateTh or (cnt == len(dataDiffs) - 1)):
            stepR = cnt
            inStep = False
            LAMPStepFL = False
            stepDiff = 0
            if (stepR - stepL) >= width_LB:
                index = stepL
                while index <= stepR:
                    stepDiff = stepDiff + dataDiffs[index]
                    index += 1
                avgRate = stepDiff / (stepR - stepL + 1)
                LAMPStepFL = avgRate >= avgRate_LB
            step = [stepL, stepR, LAMPStepFL]
            stepL = cnt + 1
            listOfSteps.append(step)
            continue
    stepDiff = 0
    cp = 0
    maxDiff = 0
    maxIndex = 0
    stepWidth = 0
    for step in listOfSteps:
        if step[-1]:
            index = step[0] - 1
            stepWidth += step[1] - step[0] + 1

            # Accumulate signal increase of all Ture step as Step Diff
            while index < step[1] + 1:
                stepDiff = stepDiff + dataDiffs[index]
                # Capture time for highest diff as Cp
                if dataDiffs[index] >= maxDiff:
                    maxDiff = dataDiffs[index]
                    maxIndex = index
                index += 1
            if len(datas) > 10: cp = (maxIndex - datas[maxIndex + 1] / dataDiffs[maxIndex]) * 10 / 60 - 5
    avgRate = 0
    if stepWidth != 0: avgRate = stepDiff/stepWidth
    
    return listOfSteps, np.round(stepDiff, 1), round(cp, 1), round(stepWidth, 1), round(avgRate, 1), np.round(maxDiff, 1)


def readRunCsv(filename):

    x = []
    signalList = []
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    y5 = []

    rlt = []
    idInfo = []
    ChResult = []
    OverallResult = ""

    with open(filename,'r') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        idx = 0
        headerDist = {}

        for row in rows:
            if idx == 0:
                for n, header in enumerate(row):
                    headerDist[header] = n
            if idx == 1:
                idInfo.append([row[0], row[headerDist["Barcode"]]])
                OverallResult = row[headerDist["OverallResult"]]
            if idx == 8:
                x = row[7:]
                x = [float(i)/1000/60 - 5 for i in x]
            if idx == 11:
                ChResult.append(row[5])
                y1 = row[7:]
                rlt.append(row[6])
                y1 = np.array([float(i) for i in y1])
                if len(y1) >= 9: signalList.append(smooth(y1))
            if idx == 12:
                ChResult.append(row[5])
                y2 = row[7:]
                rlt.append(row[6])
                y2 = np.array([float(i) for i in y2])
                if len(y2) >= 9: signalList.append(smooth(y2))
            if idx == 13:
                ChResult.append(row[5])
                y3 = row[7:]
                rlt.append(row[6])
                y3 = np.array([float(i) for i in y3])
                if len(y3) >= 9: signalList.append(smooth(y3))
            if idx == 14:
                ChResult.append(row[5])
                y4 = row[7:]
                rlt.append(row[6])
                y4 = np.array([float(i) for i in y4])
                if len(y4) >= 9: signalList.append(smooth(y4))
            if idx == 15:
                ChResult.append(row[5])
                y5 = row[7:]
                rlt.append(row[6])
                y5 = np.array([float(i) for i in y5])
                if len(y5) >= 9: signalList.append(smooth(y5))

            idx += 1

    return idInfo, OverallResult, signalList


def readTestlog(filename):
    
    testLog = {}
    with open(filename,'r') as csvfile:
        items = csv.reader(csvfile, delimiter=',')
        idx = 0
        for row in items:
            if idx == 0:
                headers = row
                idx += 1
                continue
            inputGroup = row[11]
            testLog[row[0]] = inputGroup
    return testLog
def idAudit(filename):
    df = pd.read_csv(filename)
    idMapping = {}
    
    for idx, row in df.iterrows():
        idMapping[row['Test ID#']] = row['Sample ID on Device']
    
    dataPath = './NSCPI_training/'
    filenames = sorted(glob.glob(os.path.join(dataPath, '*.csv')))
    
    errLt = []
    for filename in filenames:
        testId = os.path.basename(filename).split('.csv')[0]
        idInfo, overallRlt, signalList = readRunCsv(filename)
        sampleId = idInfo[0][0]
        
        if idMapping[testId] != sampleId:
            print(testId + ' should be ' + sampleId + ' not ' + idMapping[testId])
        
    
def testsGrouping(filename):
    df = pd.read_csv(Path(filename))
    posTests = {}
    negTests = set()
    outlierCurves = []
    
    cnt = 0
    for idx, row in df.iterrows():
        if 'PC' in row['Used for:']:
            levelGp = row['Input [C]']
            id = row['Test ID#']
            posTests[id] = levelGp
            cnt += 1
        elif 'NTC' in row['Used for:']:
            negTests.add(row['Test ID#'])
            cnt += 1
            
    posTestNum = len(posTests)
    negTestNum = len(negTests)
    logger.info(f'POS total #: {posTestNum}, NEG total #: {negTestNum}')
    return posTests, negTests, outlierCurves        

def NTCMetric(negTests, dataPath=Path('./SC2A2_training/')):
    filenames = sorted(dataPath.glob('*.csv'))
    invalidCnt = 0
    trueNegCnt = 0
    falsePosCnt = 0
    negCurves = []
    pcCurves = []
    
    for filename in filenames:
        testId = os.path.basename(filename).split('.csv')[0]
        if testId not in negTests:
            continue
        idInfo, overallRlt, signalList = readRunCsv(filename)
        
        pcCurves.append([testId, 'ch1', signalList[0]])
        negCurves.append([testId, 'ch2', signalList[1]])
        negCurves.append([testId, 'ch3', signalList[2]])
        negCurves.append([testId, 'ch4', signalList[3]])
        negCurves.append([testId, 'ch5', signalList[4]])

    return negCurves, pcCurves
                
def POSMetric(posTests, paras=[30, 0.3, 15, 0.8], thresholdLt=[40, 40, 40, 40, 40], dataPath=Path('./SC2A2_training/')):
    filenames = sorted(dataPath.glob('*.csv'))

    posCurvesL = []
    posCurvesM = []
    posCurvesH = []
    pcCurves = []
    
    
    for filename in filenames:
        testId = os.path.basename(filename).split('.csv')[0]
        if testId not in posTests:
            continue
        idInfo, overallRlt, signalList = readRunCsv(filename)
        
        pcCurves.append([testId, 'ch1', signalList[0]])
        if posTests[testId] == 1:
            posCurvesL.append([testId, 'ch2', signalList[1]])
            posCurvesL.append([testId, 'ch3', signalList[2]])
            posCurvesL.append([testId, 'ch4', signalList[3]])
            posCurvesL.append([testId, 'ch5', signalList[4]])
        elif posTests[testId] == 5:
            posCurvesM.append([testId, 'ch2', signalList[1]])
            posCurvesM.append([testId, 'ch3', signalList[2]])
            posCurvesM.append([testId, 'ch4', signalList[3]])
            posCurvesM.append([testId, 'ch5', signalList[4]])
        elif posTests[testId] == 10:
            posCurvesH.append([testId, 'ch2', signalList[1]])
            posCurvesH.append([testId, 'ch3', signalList[2]])
            posCurvesH.append([testId, 'ch4', signalList[3]])
            posCurvesH.append([testId, 'ch5', signalList[4]])
            
    return posCurvesL, posCurvesM, posCurvesH, pcCurves
    
def getInvalTestsCsv(invalidTestLt):
    dataPath = './NSCPI_training/'
    filenames = sorted(glob.glob(os.path.join(dataPath, '*.csv')))
    for test in invalidTestLt:
        baseName = test[0] + '.csv'
        filePath = os.path.join(dataPath, baseName)
        # shutil.copy(filePath, dst)
        
def curvesMetric(posCurves, negCurves, pcCurves, paras = [75, 0.3, 15, 0.8, 40]):
    
    startPt, rateTh, width_LB, avgRate_LB, threshold = paras #rateTh, width_LB, avgRate_LB, threshold
    ivCnt, fpCnt, fnLCnt, fnMCnt, fnHCnt = 0, 0, 0, 0, 0
    pcThreshold = 40
    
    posCurvesL, posCurvesM, posCurvesH = posCurves
    curvesDist = {'PC' : pcCurves, 'NEG' : negCurves, 'POSL' : posCurvesL, 'POSM' : posCurvesM, 'POSH' : posCurvesH}
    falseDetectionList = []
    
    for type, curves in curvesDist.items():
        for curve in curves:
            testId = curve[0]
            ch = curve[1]
            signal = curve[-1]
            _, diff, cp, stepWidth, avgRate, maxDiff= labelSteps(signal, startPt, rateTh, width_LB, avgRate_LB)
            rlt = (diff >= threshold) if type != 'PC' else (diff >= pcThreshold)
            
            if not rlt and type != 'NEG':
                if type == 'PC':
                    ivCnt += 1
                    falseDetectionList.append(['IV', testId, ch, signal])
                elif type == 'POSL':
                    fnLCnt += 1
                    falseDetectionList.append(['FNL', testId, ch, signal])
                elif type == 'POSM':
                    fnMCnt += 1
                    falseDetectionList.append(['FNM', testId, ch, signal])
                elif type == 'POSH':
                    fnHCnt += 1
                    falseDetectionList.append(['FNH', testId, ch, signal])
            elif rlt and type == 'NEG':
                fpCnt += 1
                falseDetectionList.append(['FP', testId, ch, signal])
            
    logger.debug(f'rateTh = {rateTh}, width_LB = {width_LB}, avgRate_LB = {avgRate_LB}, threshold = {threshold}')
    return fpCnt, fnHCnt, fnMCnt, fnLCnt, ivCnt, falseDetectionList


def getMetric():

    # dataPath = './data/'
    dataPath = './NCSPI_data/'
    filenames = sorted(glob.glob(os.path.join(dataPath, '*.csv')))
    # testLog = readTestlog('./S2R_testlog.csv')
    # print(testLog)
    posTests, negTests, outliers = testsGrouping(testlogFile)
    

    idInfo, overallRlt, signalList = "", "", []
    invalidCnt = 0
    truePosCnt = 0
    falsePosCnt = 0
    trueNegCnt = 0
    falseNegCnt = 0

    for filename in filenames:
        idInfo, overallRlt, signalList = readRunCsv(filename)
        sampleGroup = testLog[os.path.splitext(os.path.basename(filename))[0]]

        groundTruth = False
        if sampleGroup != 'Negative':
            groundTruth = True

        startPt, rateTh, width_LB, avgRate_LB = [30, 0.3, 15, 0.8] #rateTh, width_LB, avgRate_LB
        thresholdList = [40, 40, 40, 40, 40]
        rltList = [False, False, False, False, False]

        if len(signalList) != 0:

            for i in range(5):
                _, diff, cp, stepWidth, avgRate, maxDiff= labelSteps(signalList[i], startPt, rateTh, width_LB, avgRate_LB)
                rltList[i] = (diff >= thresholdList[i])
        if rltList[0] == False:
            invalidCnt += 1
        else:
            if rltList[1] or rltList[2] or rltList[3] or rltList[4] == True:
                if groundTruth: 
                    truePosCnt += 1
                else:
                    falsePosCnt += 1
            else:
                if groundTruth: 
                    falseNegCnt += 1
                else:
                    trueNegCnt += 1
    cf_matrix = [[truePosCnt, falsePosCnt], [falseNegCnt, trueNegCnt]]
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['True','False'])
    ax.yaxis.set_ticklabels(['True','False'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()


def paraSweep( paraName, range, step, testlogFile = Path('SC2A2_testlog.csv'), dataPath = Path('./SC2A2_training/')):
    # idAudit(testlogFile)
    posTests, negTests, outliers = testsGrouping(testlogFile)
    negCurves, pcNTC = NTCMetric(negTests, dataPath)
    
    posCurvesL, posCurvesM, posCurvesH, pcPOS = POSMetric(posTests, dataPath)
    posCurves = [posCurvesL, posCurvesM, posCurvesH]
    pcCurves = pcNTC + pcPOS
    
    
    paras = [75, 0.5, 15, 0.9, 40]
    index = 0
    if paraName == 'startPt':
        index = 0
    elif paraName == 'rateTh':
        index = 1
    elif paraName == 'width_LB':
        index = 2
    elif paraName == 'avgRate_LB':
        index = 3
    elif paraName == 'threshold': 
        index = 4
    paraSweeptLt = np.arange(range[0], range[1], step)
    print(f"Sweeping {paraName} from {range[0]} to {range[1]} with step {step}")
    
    for para in paraSweeptLt:
        paras[index] = np.round(para,2)
        
        fpCnt, fnHCnt, fnMCnt, fnLCnt, ivCnt, fdList = curvesMetric(posCurves, negCurves, pcCurves, paras)
        # construct result into dataframe
        d = {'FP': [fpCnt, len(negCurves)], 'FNH': [fnHCnt, len(posCurvesH)], 'FNM': [fnMCnt, len(posCurvesM)], 'FNL': [fnLCnt, len(posCurvesL)], 'IV': [ivCnt, len(pcCurves)]}
        df = pd.DataFrame(data = d, index = ['# of curves', 'Total # of curves'])
        
        print(df)

def getFalseDetectionList(paras = [75, 0.3, 15, 0.8, 40], plotType = 'FP'):
    testlogFile = Path('SC2A2_testlog.csv')
    # idAudit(testlogFile)
    posTests, negTests, outliers = testsGrouping(testlogFile)
    negCurves, pcNTC = NTCMetric(negTests)
    
    posCurves1, posCurves10, posCurves100, pcPOS = POSMetric(posTests)
    posCurves = [posCurves1, posCurves10, posCurves100]
    pcCurves = pcNTC + pcPOS
    
    fpCnt, fn100Cnt, fn10Cnt, fn1Cnt, ivCnt, fdList = curvesMetric(posCurves, negCurves, pcCurves, paras)
    with open('falseDetectionList.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(fdList)
    plotFalseDetectionCurves(fdList, plotType, paras)

def plotFalseDetectionCurves(fdList, plotType, paras):
    rate, width, avgRate, th = paras[1], paras[2], paras[3], paras[4]
    plt.style.use('seaborn-bright')

    plt.rc('axes', linewidth=2)
    font = {'weight' : 'bold',
    'size'   : 21}
    plt.rc('font', **font)
    plt.figure(num=None, figsize=(24, 12), dpi=40)

    plt.xlabel('Time (mins)', fontsize = 19, fontweight = 'bold')
    plt.ylabel('Signal (mvs)', fontsize = 19, fontweight = 'bold')
    plt.title(f'False Detection Curves for {plotType} with rateTh[{rate}], widthLb[{width}], avgRateLb[{avgRate}], Th[{th}]', fontsize = 19, fontweight = 'bold')
    

    for df in fdList:
        if plotType != df[0]:
            continue
        testId = df[1]
        ch = df[2]
        signal = df[3]
        xSeries = np.arange(0, len(signal), 1)
        xSeries = np.interp(xSeries, (xSeries.min(), xSeries.max()), (0, 30))
        plt.plot(xSeries, signal, label = testId + '_' + ch)
    
    plt.grid(True)
    plt.axis([0,30, 0, 500])
    plt.legend(ncol = 2, loc='upper right')
    fileName = f'falseDetection_{plotType}_rateTh_{rate}_widthLb_{width}_avgRateLb_{avgRate}_th_{th}.png'
    plt.savefig(fileName)
    
if __name__ == '__main__':

    # Testlog filename and data path
    TESTLOGFILE = Path('PD_testlog.csv')
    DATAPATH = Path('./PD_training/')

    msg = "Please specify the parameter (startPt, rateTh, width_LB, avgRate_LB, threshold) to sweep"

    # Initialize parser
    parser = argparse.ArgumentParser(description=msg)
    
    # Adding optional argument
    parser.add_argument("-p", help = "Parameter to sweep")
    parser.add_argument("-st", help = "start of parameter")
    parser.add_argument("-e", help = "end of parameter")
    parser.add_argument("-s", help = "Step of parameter")
    
    
    # Read arguments from command line
    args = parser.parse_args()

    availablePara = set(['startPt', 'rateTh', 'width_LB', 'avgRate_LB', 'threshold'])
    if args.p in availablePara:
        paraSweep(args.p, [int(args.st), int(args.e)], int(args.s), TESTLOGFILE, DATAPATH)
    else:
        print(msg)
    
    # paraSweep('threshold', [40, 110], 10)
    # getFalseDetectionList([75, 0.5, 15, 0.9, 80], 'IV')




