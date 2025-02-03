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

# Parse command line arguments
parser = argparse.ArgumentParser(description='ADF parameter optimization')
parser.add_argument('-d', '--data', type=str, default='./ADFtraining/',
                    help='Path to training data directory')
parser.add_argument('-t', '--testlog', type=str, default='testlog.csv',
                    help='Path to test log file')
parser.add_argument('-b', '--bounds', type=str, default='75,75|0.15,1|10,40|0.5,5|15,200',
                    help='Parameter bounds in format "startPt|rateTh|width_LB|avgRate_LB|threshold" where each is "min,max"')
parser.add_argument('-p', '--plot', action='store_true',
                    help='Flag to enable plotting false detection curves')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Enable debug level logging')


args = parser.parse_args()

PlotFalse = args.plot
argBounds = args.bounds

DATAPATH = Path(args.data)
TESTLOGFILE = Path(args.testlog)

current_date = datetime.now().strftime("%Y%m%d")
training_file = os.path.basename(DATAPATH).split('.')[0]
log_filename = f'{current_date}_{training_file}.log'



# Remove existing log file if it exists
if os.path.exists(log_filename):
    os.remove(log_filename)

logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
# Set logger level to debug
if args.verbose:
    logger.setLevel(logging.DEBUG)

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
    """Read and parse a run CSV file to extract test information and signal data.
    
    Args:
        filename (str): Path to the CSV file to read
        
    Returns:
        tuple: Contains:
            - idInfo (list): Test identification information [test_id, barcode]
            - OverallResult (str): Overall test result
            - signalList (list): List of smoothed signal data for each channel
    """
    # Initialize data structures
    x = []  # Time points
    signalList = []  # Processed signals
    channel_signals = [[] for _ in range(5)]  # Raw signals for each channel
    test_info = []  # Test identification info
    channel_results = []  # Results by channel
    overall_result = ""

    with open(filename, 'r') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        row_idx = 0
        header_positions = {}

        for row in rows:
            # Remove empty cells
            row = [cell for cell in row if cell.strip()]
            
            # Process header row
            if row_idx == 0:
                header_positions = {header: idx for idx, header in enumerate(row)}
            
            # Process test info row
            elif row_idx == 1:
                barcode = row[header_positions["Barcode"]] if "Barcode" in header_positions and header_positions["Barcode"] < len(row) else ""
                overall_result = row[header_positions["OverallResult"]] if "OverallResult" in header_positions and header_positions["OverallResult"] < len(row) else ""
                test_info.append([row[0], barcode])
            
            # Process time points row
            elif row_idx == 8:
                time_idx = next((i for i, cell in enumerate(row) if cell.strip() == 'Time'), None)
                if time_idx is not None:
                    x = row[time_idx + 4:]
                x = [float(i)/1000/60 - 5 for i in x]  # Convert to minutes
            
            # Process channel data rows (11-15)
            elif 11 <= row_idx <= 15:
                channel_idx = row_idx - 11
                target_idx = next((i for i, cell in enumerate(row) if cell.strip() == 'Target'), None)
                
                if target_idx is not None:
                    channel_results.append(row[target_idx + 1])
                    signal_data = row[target_idx + 4:]
                    channel_signals[channel_idx] = np.array([float(i) for i in signal_data])
                    
                    if len(channel_signals[channel_idx]) >= 9:
                        signalList.append(smooth(channel_signals[channel_idx]))
            
            row_idx += 1

    return test_info, overall_result, signalList
    
def testsGrouping(testlogFile):
    """Group tests based on sample type and layout information"""
    df = pd.read_csv(testlogFile)
    posTests = {}
    negTests = {}
    outliers = []
    
    # Default layout when not specified
    DEFAULT_LAYOUT = ['PC', 'Target', 'Target', 'Target', 'Target']
    
    for _, row in df.iterrows():
        test_id = row['Run UID']
        sample_type = row['Sample Type']
        
        # Handle missing Layout column or empty layout
        try:
            if pd.isna(row.get('Layout')) or not row['Layout'].strip():
                layout = DEFAULT_LAYOUT
            else:
                layout = [item.strip() for item in row['Layout'].split(',')]
                # If layout doesn't have exactly 5 items, use default
                if len(layout) != 5:
                    layout = DEFAULT_LAYOUT
        except (AttributeError, KeyError):
            # Layout column doesn't exist
            layout = DEFAULT_LAYOUT
            
        # Store test info with layout
        if sample_type == 'Positive':
            posTests[test_id] = {
                'conc': row['Sample Concentration'],
                'layout': layout
            }
        elif sample_type == 'Negative':
            negTests[test_id] = {
                'layout': layout
            }
        else:
            outliers.append(test_id)
            
    # Log the test counts
    logger.info(f'POS total #: {len(posTests)}, NEG total #: {len(negTests)}')
    if outliers:
        logger.warning(f'Found {len(outliers)} outlier tests: {outliers}')
            
    return posTests, negTests, outliers

def NTCMetric(negTests, dataPath):
    """Process negative control test data considering layout information"""
    filenames = sorted(dataPath.glob('*.csv'))
    negCurves = []
    pcCurves = []
    
    for filename in filenames:
        testId = os.path.basename(filename).split('.csv')[0]
        if testId not in negTests:
            continue
            
        _, _, signalList = readRunCsv(filename)
        if not signalList:
            continue
            
        layout = negTests[testId]['layout']
        
        # Process PC channel (ch1) if marked as PC
        if layout[0].strip().upper() == 'PC' and len(signalList) > 0:
            pcCurves.append([testId, 'ch1', signalList[0]])
            
        # Process target channels (ch2-ch5) if not marked as PC
        for i, layout_mark in enumerate(layout[1:], 1):
            if (layout_mark.strip().upper() != 'PC' and 
                i < len(signalList)):
                negCurves.append([testId, f'ch{i+1}', signalList[i]])
    
    return negCurves, pcCurves
                
def POSMetric(posTests, dataPath):
    """Process positive test data considering layout information"""
    filenames = sorted(dataPath.glob('*.csv'))
    posCurvesL = []  # Low concentration
    posCurvesM = []  # Medium concentration
    posCurvesH = []  # High concentration
    pcCurves = []
    
    # Map concentration ranges to curve lists
    conc_map = {
        1: posCurvesL,
        5: posCurvesM, 
        10: posCurvesH
    }
    
    for filename in filenames:
        testId = os.path.basename(filename).split('.csv')[0]
        if testId not in posTests:
            continue
            
        _, _, signalList = readRunCsv(filename)
        if not signalList:
            continue
            
        test_info = posTests[testId]
        layout = test_info['layout']
        conc = test_info['conc']
        
        # Process PC channel (ch1) if marked as PC
        if layout[0].strip().upper() == 'PC' and len(signalList) > 0:
            pcCurves.append([testId, 'ch1', signalList[0]])
            
        # Process target channels (ch2-ch5) if not marked as PC
        if conc in conc_map:
            curves = conc_map[conc]
            for i, layout_mark in enumerate(layout[1:], 1):
                if (layout_mark.strip().upper() != 'PC' and 
                    i < len(signalList)):
                    curves.append([testId, f'ch{i+1}', signalList[i]])
                
    return posCurvesL, posCurvesM, posCurvesH, pcCurves
    
def getInvalTestsCsv(invalidTestLt):
    dataPath = './NSCPI_training/'
    filenames = sorted(glob.glob(os.path.join(dataPath, '*.csv')))
    for test in invalidTestLt:
        baseName = test[0] + '.csv'
        filePath = os.path.join(dataPath, baseName)
        # shutil.copy(filePath, dst)
        
def curvesMetric(posCurves, negCurves, pcCurves, paras = [75, 0.3, 15, 0.8, 40]):
    
    startPt, rateTh, width_LB, avgRate_LB, threshold = paras
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
    testlogFile = 'SC2A2_testlog.csv'
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
    plt.style.use('seaborn')

    plt.rc('axes', linewidth=2)
    font = {'weight' : 'bold',
    'size'   : 21}
    plt.rc('font', **font)
    plt.figure(num=None, figsize=(24, 12), dpi=40)

    plt.xlabel('Time (mins)', fontsize = 19, fontweight = 'bold')
    plt.ylabel('Signal (mvs)', fontsize = 19, fontweight = 'bold')
    plt.title(f'False Detection Curves for {plotType} with rateTh[{rate}], widthLb[{width}], avgRateLb[{avgRate}], Th[{th}]', fontsize = 19, fontweight = 'bold')
    

    for df in fdList:
        if df[0] not in plotType:
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






