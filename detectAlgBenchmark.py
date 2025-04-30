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

# Constants for algorithm parameters
DEFAULT_START_PT = 75
DEFAULT_RATE_TH = 0.5
DEFAULT_WIDTH_LB = 15
DEFAULT_AVG_RATE_LB = 0.9
DEFAULT_THRESHOLD = 40

# Constants for concentration categories
LOW_CONC = 10
MEDIUM_CONC = 500
HIGH_CONC = 1000
DEFAULT_CONC = 10

# Constants for time conversion
TIME_CONVERSION_FACTOR = 10 / 60
TIME_OFFSET = 5

# Default channel layout
DEFAULT_LAYOUT = ['PC', 'Target', 'Target', 'Target', 'Target']

# Parse command line arguments
parser = argparse.ArgumentParser(description='ADF parameter optimization')
parser.add_argument('-d', '--data', type=str, default='./ADFtraining/',
                    help='Path to training data directory')
parser.add_argument('-t', '--testlog', type=str, default='testlog.csv',
                    help='Path to test log file')
parser.add_argument('-b', '--bounds', type=str, default='75,75|0.5,10|15,15|0.5,5|40,350',
                    help='Parameter bounds in format "startPt|rateTh|width_LB|avgRate_LB|threshold" where each is "min,max"')
parser.add_argument('-p', '--plot', action='store_true',
                    help='Flag to enable plotting false detection curves')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Enable debug level logging')
parser.add_argument('-o', '--output', type=str, default='falseDetectionList.csv',
                    help='Output file for false detection list')


args = parser.parse_args()

PlotFalse = args.plot
argBounds = args.bounds

DATAPATH = Path(args.data)
TESTLOGFILE = Path(args.testlog)
OUTPUT_FILE = args.output

current_date = datetime.now().strftime("%Y%m%d")
training_file = os.path.basename(DATAPATH).split('.')[0]
log_filename = f'{current_date}_{training_file}.log'



# Remove existing log file if it exists
if os.path.exists(log_filename):
    try:
        os.remove(log_filename)
    except PermissionError:
        # If file is in use, append to it instead
        print(f"Warning: Cannot remove log file {log_filename} - it may be in use. Will append to it.")
        # Continue execution - logging will append to existing file

logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
# Set logger level to debug
if args.verbose:
    logger.setLevel(logging.DEBUG)

def smooth(x, window_len=10, window='hanning'):
    """
    Smooth the data using a window with requested size and shape.
    
    Args:
        x (array): Input signal
        window_len (int): Length of the smoothing window
        window (str): Type of window function ('flat', 'hanning', 'hamming', 'bartlett', 'blackman')
        
    Returns:
        array: Smoothed signal
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    valid_windows = {
        'flat': np.ones,
        'hanning': np.hanning,
        'hamming': np.hamming,
        'bartlett': np.bartlett,
        'blackman': np.blackman
    }

    if window not in valid_windows:
        valid_window_names = ', '.join(f"'{name}'" for name in valid_windows.keys())
        raise ValueError(f"Window must be one of {valid_window_names}")

    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    
    if window == 'flat':  # moving average
        w = valid_windows[window](window_len)
    else:
        w = valid_windows[window](window_len)

    y = np.convolve(w/w.sum(), s, mode='valid')
    return np.round(y, decimals=3)

def labelSteps(datas, startPt=DEFAULT_START_PT, rateTh=DEFAULT_RATE_TH, 
               width_LB=DEFAULT_WIDTH_LB, avgRate_LB=DEFAULT_AVG_RATE_LB):
    """
    Identify steps in the data that meet specified criteria.
    
    Args:
        datas (array): Input signal data
        startPt (int): Starting point to look for steps
        rateTh (float): Rate threshold to identify step start/end
        width_LB (int): Minimum width of valid steps
        avgRate_LB (float): Minimum average rate for valid steps
        
    Returns:
        tuple: Contains step information, metrics about the steps
    """
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

            # Accumulate signal increase of all True steps as Step Diff
            while index < step[1] + 1:
                stepDiff = stepDiff + dataDiffs[index]
                # Capture time for highest diff as Cp
                if dataDiffs[index] >= maxDiff:
                    maxDiff = dataDiffs[index]
                    maxIndex = index
                index += 1
            
            # Calculate Cp: Adjusts maxIndex by the ratio of signal to rate at that point
            # Then converts to minutes using TIME_CONVERSION_FACTOR and adjusts by TIME_OFFSET
            if len(datas) > 10 and maxIndex < len(datas)-1 and maxIndex < len(dataDiffs):
                # Adjust index by the ratio of signal to rate
                adjusted_index = maxIndex
                if dataDiffs[maxIndex] > 0:  # Prevent division by zero
                    adjusted_index = maxIndex - datas[maxIndex + 1] / dataDiffs[maxIndex]
                # Convert to minutes
                cp = adjusted_index * TIME_CONVERSION_FACTOR - TIME_OFFSET
                
    avgRate = 0
    if stepWidth != 0: 
        avgRate = stepDiff/stepWidth
    
    return listOfSteps, np.round(stepDiff, 1), round(cp, 1), round(stepWidth, 1), round(avgRate, 1), np.round(maxDiff, 1)


def readRunCsv(filename):
    """
    Read and parse a run CSV file to extract test information and signal data.
    
    Args:
        filename (str): Path to the CSV file to read
        
    Returns:
        tuple: Contains:
            - idInfo (list): Test identification information [test_id, barcode]
            - OverallResult (str): Overall test result
            - signalList (list): List of smoothed signal data for each channel
    """
    # Initialize data structures
    signalList = []  # Processed signals
    test_info = []  # Test identification info
    overall_result = ""

    try:
        # Read the entire file to analyze its structure
        with open(filename, 'r') as csvfile:
            rows = list(csv.reader(csvfile, delimiter=','))
            
        if not rows or len(rows) < 5:
            logger.warning(f"Empty or truncated CSV file: {filename}")
            return [], "", []
        
        # First locate the test metadata in the header section
        for i in range(min(5, len(rows))):
            if 'Barcode' in rows[i] and 'OverallResult' in rows[i]:
                header_row = i
                if i+1 < len(rows):
                    info_row = i+1
                    
                    # Extract test info
                    try:
                        header = rows[header_row]
                        info = rows[info_row]
                        
                        # Find column indices
                        barcode_idx = header.index('Barcode') if 'Barcode' in header else -1
                        result_idx = header.index('OverallResult') if 'OverallResult' in header else -1
                        ruid_idx = header.index('Ruid') if 'Ruid' in header else 1  # Default to column 1
                        
                        if barcode_idx >= 0 and barcode_idx < len(info):
                            barcode = info[barcode_idx]
                        else:
                            barcode = ""
                            
                        if result_idx >= 0 and result_idx < len(info):
                            overall_result = info[result_idx]
                        else:
                            overall_result = ""
                            
                        if ruid_idx >= 0 and ruid_idx < len(info):
                            test_id = info[ruid_idx]
                        else:
                            test_id = os.path.basename(filename).split('_')[0]
                            
                        test_info = [test_id, barcode]
                        break
                    except Exception as e:
                        logger.warning(f"Error parsing header in {filename}: {str(e)}")
        
        # Now locate the signals section - look for row with "SampleId,Ruid,Well,Type,TargetName,Result,VoltageDifference,Readings"
        signals_header_row = -1
        for i in range(len(rows)):
            if 'Well' in rows[i] and 'Type' in rows[i] and 'Readings' in rows[i]:
                signals_header_row = i
                break
                
        if signals_header_row == -1:
            logger.warning(f"Could not find signals section in {filename}")
            return test_info, overall_result, []
            
        # Process Target rows with channel data
        target_rows = []
        for i in range(signals_header_row + 1, len(rows)):
            if len(rows[i]) > 4 and 'Target' in rows[i][3]:  # Type column is Target
                target_rows.append(i)
        
        # Extract channel data for each target row
        for row_idx in target_rows:
            if row_idx < len(rows):
                row = rows[row_idx]
                
                # Check if this is a channel with signal data
                if len(row) < 8:  # Need at least 8 columns for readings
                    continue
                    
                try:
                    # Remove empty strings and convert to float
                    readings_start_idx = 7  # Readings column starts at index 7
                    signal_data = []
                    
                    for val in row[readings_start_idx:]:
                        if val.strip():  # Skip empty cells
                            try:
                                signal_data.append(float(val))
                            except ValueError:
                                # Skip non-numeric values
                                pass
                    
                    if len(signal_data) >= 9:
                        # Apply smoothing to the signal data
                        signalList.append(smooth(np.array(signal_data)))
                    else:
                        # Add an empty placeholder for consistent channel indexing
                        signalList.append(np.array([]))
                        
                except Exception as e:
                    logger.warning(f"Error processing signal data in {filename}, row {row_idx}: {str(e)}")
                    # Add an empty placeholder
                    signalList.append(np.array([]))
        
        # If we found no valid signal data, log a warning
        if not signalList:
            logger.warning(f"No valid signal data found in {filename}")
    
    except Exception as e:
        logger.error(f"Failed to parse CSV file {filename}: {str(e)}")
        return test_info, overall_result, []
        
    return test_info, overall_result, signalList
    
def testsGrouping(testlogFile):
    """
    Group tests based on sample type and layout information
    
    Args:
        testlogFile (Path): Path to the test log CSV file
        
    Returns:
        tuple: Contains dictionaries of positive and negative tests and a list of outliers
    """
    df = pd.read_csv(testlogFile)
    posTests = {}
    negTests = {}
    outliers = []

    for _, row in df.iterrows():
        test_id = row['Run UID']
        sample_type = None # Initialize sample_type

        # Try getting from 'Sample Type' column first
        if 'Sample Type' in row.index and not pd.isna(row['Sample Type']) and row['Sample Type'].strip():
            sample_type = row['Sample Type']
        # Fallback to 'Expected Result' column
        elif 'Expected Result' in row.index and not pd.isna(row['Expected Result']) and row['Expected Result'].strip():
             sample_type = row['Expected Result']

        # Handle cases where neither column provides a valid sample type
        if sample_type is None:
            logger.warning(f"Could not determine sample type for Test ID {test_id} from 'Sample Type' or 'Expected Result'. Skipping.")
            outliers.append(test_id) # Treat as outlier or handle as needed
            continue # Skip processing this row

        # Handle missing Layout column safely
        layout_str = row.get('Layout') # Use .get() for safe access
        if pd.isna(layout_str) or not layout_str.strip():
            layout = DEFAULT_LAYOUT.copy()
        else:
            layout = [item.strip() for item in layout_str.split(',')]
            # If layout doesn't have exactly 5 items, use default
            if len(layout) != 5:
                logger.warning(f"Layout for Test ID {test_id} is invalid: '{layout_str}'. Using default.")
                layout = DEFAULT_LAYOUT.copy()

        # Handle missing Sample Concentration column safely
        conc = row.get('Sample Concentration') # Use .get() for safe access
        # Default to DEFAULT_CONC if column is missing or value is NaN
        if pd.isna(conc):
            conc = DEFAULT_CONC
        else:
            try:
                 # Ensure concentration is a number
                 conc = float(conc)
            except ValueError:
                 logger.warning(f"Invalid concentration value for Test ID {test_id}: '{row['Sample Concentration']}'. Using default {DEFAULT_CONC}.")
                 conc = DEFAULT_CONC


        # Store test info with layout and concentration
        if sample_type == 'Positive':
            posTests[test_id] = {
                'conc': conc,
                'layout': layout
            }
        elif sample_type == 'Negative':
            negTests[test_id] = {
                'layout': layout
            }
        else:
            # Handle other sample types if necessary, or treat as outliers
            logger.warning(f"Unknown sample type '{sample_type}' for Test ID {test_id}. Treating as outlier.")
            outliers.append(test_id)

    # Log the test counts
    logger.info(f'POS total #: {len(posTests)}, NEG total #: {len(negTests)}')
    if outliers:
        logger.warning(f'Found {len(outliers)} outlier tests: {outliers}')

    return posTests, negTests, outliers

def NTCMetric(negTests, dataPath):
    """
    Process negative control test data considering layout information
    
    Args:
        negTests (dict): Dictionary of negative tests with layout information
        dataPath (Path): Path to the data directory
        
    Returns:
        tuple: Contains lists of negative curves and PC curves
    """
    filenames = sorted(dataPath.glob('*.csv'))
    negCurves = []
    pcCurves = []
    missing_pc_info = []  # Store test_ids and filenames with missing PC curves
    
    # Create a dictionary to map test IDs to filenames
    test_files = {}
    for filename in filenames:
        file_basename = os.path.basename(filename)
        # Add file to dictionary for any test ID it contains
        for test_id in negTests:
            if test_id in file_basename:
                test_files[test_id] = filename
    
    for test_id, test_info in negTests.items():
        # Skip if not a PC layout in channel 1
        if test_info['layout'][0].strip().upper() != 'PC':
            continue
            
        if test_id not in test_files:
            logger.debug(f"No file found for negative test ID: {test_id}")
            missing_pc_info.append((test_id, "File not found"))
            continue
            
        filename = test_files[test_id]
        _, _, signalList = readRunCsv(filename)
        if not signalList:
            logger.debug(f"No signal data found for test ID: {test_id}")
            missing_pc_info.append((test_id, os.path.basename(str(filename))))
            continue
            
        layout = test_info['layout']
        
        # Process PC channel (ch1) if marked as PC
        if layout[0].strip().upper() == 'PC':
            if len(signalList) > 0 and len(signalList[0]) > 0:
                pcCurves.append([test_id, 'ch1', signalList[0]])
            else:
                logger.debug(f"Missing PC curve data for test ID: {test_id}")
                missing_pc_info.append((test_id, os.path.basename(str(filename))))
            
        # Process target channels (ch2-ch5) if not marked as PC
        for i, layout_mark in enumerate(layout[1:], 1):
            if (layout_mark.strip().upper() != 'PC' and 
                i < len(signalList)):
                # Check if signal data is valid
                if i < len(signalList) and signalList[i] is not None and len(signalList[i]) > 0:
                    negCurves.append([test_id, f'ch{i+1}', signalList[i]])
    
    logger.info(f"Number of negative curves: {len(negCurves)}")
    
    # Log details about missing PC curves
    if missing_pc_info:
        logger.warning(f"Missing PC curves from {len(missing_pc_info)} negative tests")
        for test_id, filename in missing_pc_info:
            logger.warning(f"  - Test ID {test_id}: {filename}")
    
    return negCurves, pcCurves, missing_pc_info
                
def POSMetric(posTests, dataPath):
    """
    Process positive test data considering layout information
    
    Args:
        posTests (dict): Dictionary of positive tests with layout and concentration
        dataPath (Path): Path to the data directory
        
    Returns:
        tuple: Contains lists of positive curves at different concentrations and PC curves
    """
    filenames = sorted(dataPath.glob('*.csv'))
    posCurvesL = []  # Low concentration
    posCurvesM = []  # Medium concentration
    posCurvesH = []  # High concentration
    pcCurves = []
    missing_pc_info = []  # Store test_ids and filenames with missing PC curves
    
    # Create a dictionary to map test IDs to filenames
    test_files = {}
    for filename in filenames:
        file_basename = os.path.basename(filename)
        # Add file to dictionary for any test ID it contains
        for test_id in posTests:
            if test_id in file_basename:
                test_files[test_id] = filename
    
    # Map concentration ranges to curve lists
    conc_map = {
        LOW_CONC: posCurvesL,
        MEDIUM_CONC: posCurvesM, 
        HIGH_CONC: posCurvesH
    }
    
    for test_id, test_info in posTests.items():
        # Skip if not a PC layout in channel 1
        if test_info['layout'][0].strip().upper() != 'PC':
            continue
            
        if test_id not in test_files:
            logger.debug(f"No file found for positive test ID: {test_id}")
            missing_pc_info.append((test_id, "File not found"))
            continue
            
        filename = test_files[test_id]
        _, _, signalList = readRunCsv(filename)
        if not signalList:
            logger.debug(f"No signal data found for test ID: {test_id}")
            missing_pc_info.append((test_id, os.path.basename(str(filename))))
            continue
            
        layout = test_info['layout']
        conc = test_info['conc']
        
        # Process PC channel (ch1) if marked as PC
        if layout[0].strip().upper() == 'PC':
            if len(signalList) > 0 and len(signalList[0]) > 0:
                pcCurves.append([test_id, 'ch1', signalList[0]])
            else:
                logger.debug(f"Missing PC curve data for test ID: {test_id}")
                missing_pc_info.append((test_id, os.path.basename(str(filename))))
            
        # Process target channels (ch2-ch5) if not marked as PC
        curves = None
        if conc in conc_map:
            curves = conc_map[conc]
        else:
            # If concentration doesn't match predefined levels, use closest one
            closest_conc = LOW_CONC  # Default to low
            if conc > (MEDIUM_CONC + LOW_CONC) / 2:
                if conc > (HIGH_CONC + MEDIUM_CONC) / 2:
                    closest_conc = HIGH_CONC
                else:
                    closest_conc = MEDIUM_CONC
            curves = conc_map[closest_conc]
            logger.debug(f"Test ID {test_id} has non-standard concentration {conc}. Using {closest_conc} category.")
        
        for i, layout_mark in enumerate(layout[1:], 1):
            if (layout_mark.strip().upper() != 'PC' and 
                i < len(signalList)):
                if signalList[i] is not None and len(signalList[i]) > 0:
                    curves.append([test_id, f'ch{i+1}', signalList[i]])
    
    logger.info(f"Number of positive curves: {len(posCurvesL) + len(posCurvesM) + len(posCurvesH)}")
    
    # Log details about missing PC curves
    if missing_pc_info:
        logger.warning(f"Missing PC curves from {len(missing_pc_info)} positive tests")
        for test_id, filename in missing_pc_info:
            logger.warning(f"  - Test ID {test_id}: {filename}")
    
    # Log expected PC count from positive tests only
    expected_pc_from_pos = sum(1 for test_info in posTests.values() 
                              if test_info['layout'][0].strip().upper() == 'PC')
    if len(pcCurves) != expected_pc_from_pos:
        logger.warning(f"Expected {expected_pc_from_pos} PC curves from positive tests, found {len(pcCurves)}.")
        logger.warning(f"This discrepancy may be due to missing files, invalid data, or tests without PC layout.")
    
    return posCurvesL, posCurvesM, posCurvesH, pcCurves, missing_pc_info
    
def curvesMetric(posCurves, negCurves, pcCurves, paras=[DEFAULT_START_PT, DEFAULT_RATE_TH, DEFAULT_WIDTH_LB, DEFAULT_AVG_RATE_LB, DEFAULT_THRESHOLD]):
    """
    Calculate metrics for curve classification based on given parameters
    
    Args:
        posCurves (list): List of positive curves at different concentrations
        negCurves (list): List of negative curves
        pcCurves (list): List of PC curves
        paras (list): Parameters for the detection algorithm
        
    Returns:
        tuple: Contains counts of false positives, false negatives, invalid PCs, and detection details
    """
    startPt, rateTh, width_LB, avgRate_LB, threshold = paras
    ivCnt, fpCnt, fnLCnt, fnMCnt, fnHCnt = 0, 0, 0, 0, 0
    
    posCurvesL, posCurvesM, posCurvesH = posCurves
    curvesDist = {'PC': pcCurves, 'NEG': negCurves, 'POSL': posCurvesL, 'POSM': posCurvesM, 'POSH': posCurvesH}
    falseDetectionList = []
    
    for type, curves in curvesDist.items():
        for curve in curves:
            testId = curve[0]
            ch = curve[1]
            signal = curve[-1]
            _, diff, cp, stepWidth, avgRate, maxDiff = labelSteps(signal, startPt, rateTh, width_LB, avgRate_LB)
            rlt = (diff >= threshold) 
            
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
            
    logger.debug(f'startPt = {startPt}, rateTh = {rateTh}, width_LB = {width_LB}, avgRate_LB = {avgRate_LB}, threshold = {threshold}')
    return fpCnt, fnHCnt, fnMCnt, fnLCnt, ivCnt, falseDetectionList

def plotFalseDetectionCurves(fdList, plotType, paras, save_path=None, show_annotations=True, max_curves_per_plot=50):
    """
    Plot false detection curves for analysis
    
    Args:
        fdList (list): List of false detection data
        plotType (str): Type of false detection to plot ('FP', 'FNL', etc.)
        paras (list): Parameters used for detection
        save_path (str, optional): Custom path to save the plot. If None, uses default naming.
        show_annotations (bool): Whether to display annotations showing detection metrics
        max_curves_per_plot (int): Maximum number of curves to show in a single plot
        
    Returns:
        list: List of figure objects for further customization if needed
    """
    startPt, rate, width, avgRate, th = paras[0], paras[1], paras[2], paras[3], paras[4]
    
    # Filter curves for the requested plot type
    relevant_curves = [df for df in fdList if df[0] in plotType]
    
    if not relevant_curves:
        logger.warning(f"No curves found for plot type: {plotType}")
        # Create a simple empty plot
        fig, ax = plt.subplots(figsize=(24, 12), dpi=80)
        ax.text(15, 250, f"No {plotType} curves found", 
                horizontalalignment='center', fontsize=24)
        plt.grid(True)

        if save_path:
            fileName = save_path
        else:
            fileName = f'falseDetection_{plotType}_rateTh_{rate}_widthLb_{width}_avgRateLb_{avgRate}_th_{th}.png'
        plt.tight_layout()
        plt.savefig(fileName, dpi=120)
        logger.info(f"Saved empty plot to {fileName}")
        return [fig]
    
    # Calculate how many plots we need
    num_plots = (len(relevant_curves) + max_curves_per_plot - 1) // max_curves_per_plot
    logger.info(f"Splitting {len(relevant_curves)} curves of type {plotType} into {num_plots} plots")
    
    all_figures = []
    curves_metrics = []  # Store metrics data for CSV export
    
    # Create multiple plots if needed
    for plot_idx in range(num_plots):
        # Use sns.set_style instead of plt.style.use
        sns.set_style("whitegrid")
        
        # Set color palette based on plot type
        if plotType == 'FP':
            color_palette = sns.color_palette("Reds_d", 8)
        elif plotType in ['FNL', 'FNM', 'FNH']:
            color_palette = sns.color_palette("Blues_d", 8)
        elif plotType == 'IV':
            color_palette = sns.color_palette("Purples_d", 8)
        else:
            color_palette = sns.color_palette("husl", 8)
        
        plt.rc('axes', linewidth=2)
        font = {'weight': 'bold', 'size': 21}
        plt.rc('font', **font)
        
        # Standard figure size since we no longer need space for the table
        fig, ax = plt.subplots(figsize=(24, 12), dpi=80)
        all_figures.append(fig)
        
        # Calculate start and end indices for this chunk
        start_idx = plot_idx * max_curves_per_plot
        end_idx = min(start_idx + max_curves_per_plot, len(relevant_curves))
        chunk_curves = relevant_curves[start_idx:end_idx]
        
        # Plot title based on plot type - Simplified to not include parameters
        title_map = {
            'FP': 'False Positive Detection Curves',
            'FNL': 'False Negative (Low Conc.) Detection Curves',
            'FNM': 'False Negative (Medium Conc.) Detection Curves',
            'FNH': 'False Negative (High Conc.) Detection Curves',
            'IV': 'Invalid PC Detection Curves'
        }
        
        title = title_map.get(plotType, f'False Detection Curves for {plotType}')
        if num_plots > 1:
            title += f' (Group {plot_idx+1} of {num_plots})'
            
        # Use a cleaner title without parameters
        plt.title(title, fontsize=22, fontweight='bold')
        
        plt.xlabel('Time (mins)', fontsize=19, fontweight='bold')
        plt.ylabel('Signal (mvs)', fontsize=19, fontweight='bold')
        
        # Add vertical line at startPt
        time_at_startPt = startPt * TIME_CONVERSION_FACTOR - TIME_OFFSET
        if time_at_startPt >= 0 and time_at_startPt <= 30:
            ax.axvline(x=time_at_startPt, color='green', linestyle=':', alpha=0.7,
                    label=f'Start Point ({startPt})')
        
        # Plot curves with colors from palette
        curves_info = []
        plotted_curves = 0
        max_signal = 1500  # Default max
        
        for i, df in enumerate(chunk_curves):
            testId = df[1]
            ch = df[2]
            signal = df[3]
            curve_label = f"{testId}_{ch}"
            
            # Calculate time series (x-axis)
            xSeries = np.arange(0, len(signal), 1)
            xSeries = np.interp(xSeries, (xSeries.min(), xSeries.max()), (0, 30))
            
            # Plot with color from palette (cycling through)
            color = color_palette[i % len(color_palette)]
            line, = ax.plot(xSeries, signal, label=curve_label, color=color, linewidth=2)
            
            # Calculate metrics for this curve for annotation
            steps, diff, cp, stepWidth, avgRate_val, maxDiff = labelSteps(signal, startPt, rate, 
                                                                width, avgRate)
            
            # Store metrics for CSV export
            curves_metrics.append({
                'Type': plotType,
                'TestID': testId,
                'Channel': ch,
                'Diff': diff,
                'Cp': cp,
                'StepWidth': stepWidth,
                'AvgRate': avgRate_val,
                'MaxDiff': maxDiff
            })
            
            # Store curve info for annotation
            curves_info.append({
                'line': line,
                'label': curve_label,
                'metrics': {
                    'diff': diff,
                    'cp': cp,
                    'stepWidth': stepWidth,
                    'avgRate': avgRate_val,
                    'maxDiff': maxDiff
                },
                'signal': signal,
                'xSeries': xSeries
            })
            
            plotted_curves += 1
        
        # Add parameters as a box at the right side of title
        param_text = f"Parameters: startPt={startPt}, rateTh={rate:.2f}, widthLb={width}, avgRateLb={avgRate:.2f}, Th={th:.2f}"
        param_box = dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7)
        ax.text(0.98, 0.98, param_text, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', horizontalalignment='right',
                bbox=param_box)
        
        # Add annotations if requested and curves exist
        if show_annotations and plotted_curves > 0 and curves_info:
            # No metrics table, just adjust layout for legend
            plt.subplots_adjust(bottom=0.15, top=0.92, left=0.07, right=0.93)
            
            # Add markers at critical points for each curve
            for info in curves_info:
                signal = info['signal']
                xSeries = info['xSeries']
                metrics = info['metrics']
                
                # Find index of the Cp value
                if metrics['cp'] > 0:
                    cp_index = int((metrics['cp'] + TIME_OFFSET) / TIME_CONVERSION_FACTOR)
                    if cp_index < len(signal):
                        ax.plot(metrics['cp'], signal[cp_index], 'o', color='black', markersize=8,
                            markeredgecolor=info['line'].get_color(), markeredgewidth=2)
                            
        # Adjust plot settings
        plt.grid(True)
        ax.set_xlim([0, 35])
        ax.set_ylim([0, max_signal])
        
        # Add legend at the bottom of the image (outside plotting area) for all plot types
        if plotted_curves > 0:
            # Calculate optimal number of columns based on number of curves
            if plotted_curves <= 4:
                ncols = plotted_curves
            else:
                ncols = min(6, plotted_curves // 2 + 1)  # Limit to at most 6 columns
                
            # Position legend below the plot with improved width matching
            legend = ax.legend(ncol=ncols, loc='upper center', 
                              fontsize='small', framealpha=0.8, 
                              bbox_to_anchor=(0.5, -0.05), borderaxespad=0.8)
            
            # Set legend title to plot type
            legend_title_map = {
                'FP': 'False Positives',
                'FNL': 'False Negatives (Low)',
                'FNM': 'False Negatives (Medium)',
                'FNH': 'False Negatives (High)',
                'IV': 'Invalid PC'
            }
            legend_title = legend_title_map.get(plotType, plotType)
            legend.set_title(legend_title, prop={'size': 'small', 'weight': 'bold'})
            
            # Adjust figure size to match legend width
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.25)  # Add more space at bottom for legend
        
        # Save plot with informative name
        if save_path:
            if num_plots > 1:
                # Insert group number before file extension
                base, ext = os.path.splitext(save_path)
                fileName = f"{base}_group{plot_idx+1}{ext}"
            else:
                fileName = save_path
        else:
            if num_plots > 1:
                fileName = f'falseDetection_{plotType}_{rate:.2f}_{width}_{avgRate:.2f}_{th:.2f}_group{plot_idx+1}.png'
            else:
                fileName = f'falseDetection_{plotType}_{rate:.2f}_{width}_{avgRate:.2f}_{th:.2f}.png'
        
        # Use bbox_inches='tight' to ensure all elements are included without cropping
        plt.savefig(fileName, dpi=120, bbox_inches='tight')
        logger.info(f"Saved plot to {fileName}")
    
    # Save metrics to CSV file
    if curves_metrics:
        # Generate CSV filename based on plot type
        if save_path:
            # Use save_path to derive CSV name
            base, _ = os.path.splitext(save_path)
            csv_filename = f"{base}_metrics.csv"
        else:
            csv_filename = f'falseDetection_{plotType}_{rate:.2f}_{width}_{avgRate:.2f}_{th:.2f}_metrics.csv'
        
        # Convert metrics to DataFrame and save to CSV
        metrics_df = pd.DataFrame(curves_metrics)
        metrics_df.to_csv(csv_filename, index=False)
        logger.info(f"Saved metrics to {csv_filename}")
    
    return all_figures





