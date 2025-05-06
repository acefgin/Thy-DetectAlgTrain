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
TIME_OFFSET = 0

# Default channel layout
DEFAULT_LAYOUT = ['PC', 'Target', 'Target', 'Target', 'Target']

# Parse command line arguments
parser = argparse.ArgumentParser(description='ADF parameter optimization')
parser.add_argument('-d', '--data', type=str, default='./ADFtraining/',
                    help='Path to training data directory')
parser.add_argument('-t', '--testlog', type=str, default='testlog.csv',
                    help='Path to test log file')
parser.add_argument('-b', '--bounds', type=str, default='75,75|0.3,10|15,30|0.5,5|40,350',
                    help='Parameter bounds in format "startPt|rateTh|width_LB|avgRate_LB|threshold" where each is "min,max"')
parser.add_argument('-p', '--plot', action='store_true',
                    help='Flag to enable plotting false detection curves')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Enable debug level logging')
parser.add_argument('-o', '--output', type=str, default='falseDetectionList.csv',
                    help='Output file for false detection list')
parser.add_argument('-m', '--smooth-method', type=str, default='adaptive',
                    choices=['standard', 'savgol', 'median', 'adaptive'],
                    help='Signal smoothing method to use')
parser.add_argument('-a', '--adaptive', action='store_true', default=True,
                    help='Enable adaptive window sizing for smoothing')
parser.add_argument('-c', '--cutoff', type=float, default=None,
                    help='Cutoff time in minutes to ignore data after (default: use all data)')


args = parser.parse_args()

PlotFalse = args.plot
argBounds = args.bounds
SMOOTH_METHOD = args.smooth_method
ADAPTIVE_SMOOTH = args.adaptive
CUTOFF_TIME = args.cutoff

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

# Log smoothing method configuration
logger.info(f"Using '{SMOOTH_METHOD}' smoothing method with adaptive sizing: {ADAPTIVE_SMOOTH}")

# Log cutoff time if specified
if CUTOFF_TIME is not None:
    logger.info(f"Using cutoff time: {CUTOFF_TIME} minutes (ignoring data after this time point)")
    print(f"Using cutoff time: {CUTOFF_TIME} minutes")
else:
    logger.info("No cutoff time specified - using all data points")

def smooth(x, window_len=10, window='hanning', method='standard', poly_order=3, adaptive=False):
    """
    Smooth the data using various filtering methods while preserving curve trends.
    
    Args:
        x (array): Input signal
        window_len (int): Length of the smoothing window
        window (str): Type of window function ('flat', 'hanning', 'hamming', 'bartlett', 'blackman')
        method (str): Smoothing method ('standard', 'savgol', 'median', 'adaptive')
        poly_order (int): Polynomial order for Savitzky-Golay filter
        adaptive (bool): Whether to use adaptive window sizing based on signal characteristics
        
    Returns:
        array: Smoothed signal
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        return x  # Return original signal if it's smaller than window size

    if window_len < 3:
        return x  # No smoothing for very small windows

    # Adaptive window sizing if requested
    if adaptive:
        # Calculate signal variability
        signal_std = np.std(x)
        
        # Adjust window length based on variability
        # Small window for high variability, larger for low variability
        if signal_std > np.mean(np.abs(x)) * 0.15:  # High variability 
            window_len = max(3, int(window_len * 0.7))
        elif signal_std < np.mean(np.abs(x)) * 0.05:  # Low variability
            window_len = min(int(window_len * 1.3), len(x) // 4)
            
        # Ensure window length is odd for symmetry
        if window_len % 2 == 0:
            window_len += 1

    # Apply different smoothing methods
    if method == 'savgol':
        try:
            from scipy import signal
            # Ensure window_len is odd for Savitzky-Golay
            if window_len % 2 == 0:
                window_len += 1
            
            # Ensure poly_order is less than window_len
            if poly_order >= window_len:
                poly_order = window_len - 1
            
            y = signal.savgol_filter(x, window_len, poly_order)
            return np.round(y, decimals=3)
        except ImportError:
            logger.warning("SciPy not available, falling back to standard smoothing")
            method = 'standard'
            
    if method == 'median':
        try:
            from scipy import ndimage
            y = ndimage.median_filter(x, size=window_len)
            return np.round(y, decimals=3)
        except ImportError:
            logger.warning("SciPy not available, falling back to standard smoothing")
            method = 'standard'
            
    if method == 'adaptive':
        # Adaptive method combines standard and edge-preserving techniques
        # Use median filter for the initial pass to remove spikes
        try:
            from scipy import ndimage
            # Median filter to remove spikes
            x_med = ndimage.median_filter(x, size=min(5, window_len))
            
            # Then apply Savitzky-Golay for trend preservation
            from scipy import signal
            if window_len % 2 == 0:
                window_len += 1
            if poly_order >= window_len:
                poly_order = window_len - 1
                
            y = signal.savgol_filter(x_med, window_len, poly_order)
            return np.round(y, decimals=3)
        except ImportError:
            logger.warning("SciPy not available, falling back to standard smoothing")
            method = 'standard'

    # Standard smoothing with improved edge handling
    if method == 'standard':
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

        # Improved edge handling by mirroring the signal
        s = np.r_[2*x[0] - x[window_len:0:-1], x, 2*x[-1] - x[-2:-window_len-2:-1]]
        
        w = valid_windows[window](window_len)
        y = np.convolve(w/w.sum(), s, mode='valid')
        
        # Further adjust endpoints to avoid distortion
        # Blend the first few and last few points to avoid edge effects
        blend_len = min(3, window_len // 3)
        if len(y) > 2*blend_len:
            y[:blend_len] = np.linspace(x[0], y[blend_len], blend_len)
            y[-blend_len:] = np.linspace(y[-blend_len-1], x[-1], blend_len)

        return np.round(y, decimals=3)
        
    return np.round(x, decimals=3)  # Fallback to original signal

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
            - idInfo (list): Test identification information [sample_id, test_id, barcode]
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
                        sample_id_idx = header.index('SampleId') if 'SampleId' in header else -1
                        
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
                        
                        # Extract Sample ID if available    
                        if sample_id_idx >= 0 and sample_id_idx < len(info) and info[sample_id_idx].strip():
                            sample_id = info[sample_id_idx]
                        else:
                            # Use test_id as fallback if sample_id is not available
                            sample_id = test_id
                            
                        test_info = [sample_id, test_id, barcode]
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
                        # Apply smoothing using method from command-line arguments
                        signalList.append(smooth(np.array(signal_data), method=SMOOTH_METHOD, adaptive=ADAPTIVE_SMOOTH))
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
        # Check if Sample ID is available (added for human-friendly ID reference)
        if 'Sample ID' in row.index and not pd.isna(row['Sample ID']) and row['Sample ID'].strip():
            sample_id = row['Sample ID']
        else:
            # Use Run UID as fallback
            sample_id = test_id
            
        sample_type = None # Initialize sample_type

        # Try getting from 'Sample Type' column first
        if 'Sample Type' in row.index and not pd.isna(row['Sample Type']) and row['Sample Type'].strip():
            sample_type = row['Sample Type']
        # Fallback to 'Expected Result' column
        elif 'Expected Result' in row.index and not pd.isna(row['Expected Result']) and row['Expected Result'].strip():
             sample_type = row['Expected Result']

        # Handle cases where neither column provides a valid sample type
        if sample_type is None:
            logger.warning(f"Could not determine sample type for Test ID {sample_id} (UID: {test_id}) from 'Sample Type' or 'Expected Result'. Skipping.")
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
                logger.warning(f"Layout for Test ID {sample_id} (UID: {test_id}) is invalid: '{layout_str}'. Using default.")
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
                 logger.warning(f"Invalid concentration value for Test ID {sample_id} (UID: {test_id}): '{row['Sample Concentration']}'. Using default {DEFAULT_CONC}.")
                 conc = DEFAULT_CONC


        # Store test info with layout, concentration, and sample_id
        if sample_type == 'Positive':
            posTests[test_id] = {
                'conc': conc,
                'layout': layout,
                'sample_id': sample_id
            }
        elif sample_type == 'Negative':
            negTests[test_id] = {
                'layout': layout,
                'sample_id': sample_id
            }
        else:
            # Handle other sample types if necessary, or treat as outliers
            logger.warning(f"Unknown sample type '{sample_type}' for Test ID {sample_id} (UID: {test_id}). Treating as outlier.")
            outliers.append(test_id)

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
    
    # Count tests with matching raw data
    matched_data_count = 0
    processed_test_count = 0
    
    for test_id, test_info in negTests.items():
        # Skip if not a PC layout in channel 1
        if test_info['layout'][0].strip().upper() != 'PC':
            continue
            
        if test_id not in test_files:
            logger.debug(f"No file found for negative test ID: {test_info['sample_id']} (UID: {test_id})")
            missing_pc_info.append((test_info['sample_id'], "File not found"))
            continue
        
        # Count tests with matched data files
        matched_data_count += 1
            
        filename = test_files[test_id]
        test_info_from_file, _, signalList = readRunCsv(filename)
        # Use the file's sample_id if available, otherwise use the one from the test log
        sample_id = test_info_from_file[0] if test_info_from_file and test_info_from_file[0] else test_info['sample_id']
        
        if not signalList:
            logger.debug(f"No signal data found for test ID: {sample_id} (UID: {test_id})")
            missing_pc_info.append((sample_id, os.path.basename(str(filename))))
            continue
        
        # Count tests that were successfully processed with valid signal data
        processed_test_count += 1
            
        layout = test_info['layout']
        
        # Process PC channel (ch1) if marked as PC
        if layout[0].strip().upper() == 'PC':
            if len(signalList) > 0 and len(signalList[0]) > 0:
                pcCurves.append([sample_id, 'ch1', signalList[0]])
            else:
                logger.debug(f"Missing PC curve data for test ID: {sample_id} (UID: {test_id})")
                missing_pc_info.append((sample_id, os.path.basename(str(filename))))
            
        # Process target channels (ch2-ch5) if not marked as PC
        for i, layout_mark in enumerate(layout[1:], 1):
            if (layout_mark.strip().upper() != 'PC' and 
                i < len(signalList)):
                # Check if signal data is valid
                if i < len(signalList) and signalList[i] is not None and len(signalList[i]) > 0:
                    negCurves.append([sample_id, f'ch{i+1}', signalList[i]])
    
    logger.info(f"NEG test count: {processed_test_count}")
    logger.info(f"NEG curve count: {len(negCurves)}")

    
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
    
    # Count tests with matching raw data
    matched_data_count = 0
    processed_test_count = 0
    
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
            logger.debug(f"No file found for positive test ID: {test_info['sample_id']} (UID: {test_id})")
            missing_pc_info.append((test_info['sample_id'], "File not found"))
            continue
        
        # Count tests with matched data files
        matched_data_count += 1
            
        filename = test_files[test_id]
        test_info_from_file, _, signalList = readRunCsv(filename)
        # Use the file's sample_id if available, otherwise use the one from the test log
        sample_id = test_info_from_file[0] if test_info_from_file and test_info_from_file[0] else test_info['sample_id']
        
        if not signalList:
            logger.debug(f"No signal data found for test ID: {sample_id} (UID: {test_id})")
            missing_pc_info.append((sample_id, os.path.basename(str(filename))))
            continue
        
        # Count tests that were successfully processed with valid signal data
        processed_test_count += 1
            
        layout = test_info['layout']
        conc = test_info['conc']
        
        # Process PC channel (ch1) if marked as PC
        if layout[0].strip().upper() == 'PC':
            if len(signalList) > 0 and len(signalList[0]) > 0:
                pcCurves.append([sample_id, 'ch1', signalList[0]])
            else:
                logger.debug(f"Missing PC curve data for test ID: {sample_id} (UID: {test_id})")
                missing_pc_info.append((sample_id, os.path.basename(str(filename))))
            
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
            logger.debug(f"Test ID {sample_id} (UID: {test_id}) has non-standard concentration {conc}. Using {closest_conc} category.")
        
        for i, layout_mark in enumerate(layout[1:], 1):
            if (layout_mark.strip().upper() != 'PC' and 
                i < len(signalList)):
                if signalList[i] is not None and len(signalList[i]) > 0:
                    curves.append([sample_id, f'ch{i+1}', signalList[i]])
    
    logger.info(f"POS test count: {processed_test_count}")
    logger.info(f"POS curve count: {len(posCurvesL) + len(posCurvesM) + len(posCurvesH)}")
    
    return posCurvesL, posCurvesM, posCurvesH, pcCurves, missing_pc_info
    
def curvesMetric(posCurves, negCurves, pcCurves, core_params, threshold_PC=DEFAULT_THRESHOLD, threshold_T=DEFAULT_THRESHOLD):
    """
    Calculate metrics for curve classification based on given parameters
    
    Args:
        posCurves (list): List of positive curves at different concentrations
        negCurves (list): List of negative curves
        pcCurves (list): List of PC curves
        core_params (list): Core detection parameters [startPt, rateTh, width_LB, avgRate_LB]
        threshold_PC (float): Threshold for PC curves (channel 1)
        threshold_T (float): Threshold for target curves (channels 2-5)
        
    Returns:
        tuple: Contains counts of false positives, false negatives, invalid PCs, and detection details
    """
    startPt, rateTh, width_LB, avgRate_LB = core_params
    ivCnt, fpCnt, fnLCnt, fnMCnt, fnHCnt = 0, 0, 0, 0, 0
    
    posCurvesL, posCurvesM, posCurvesH = posCurves
    curvesDist = {'PC': pcCurves, 'NEG': negCurves, 'POSL': posCurvesL, 'POSM': posCurvesM, 'POSH': posCurvesH}
    falseDetectionList = []
    
    for type, curves in curvesDist.items():
        for curve in curves:
            # First element is now sample_id (human-readable)
            sample_id = curve[0]
            ch = curve[1]
            signal = curve[-1]
            steps, diff, cp, stepWidth, avgRate, maxDiff = labelSteps(signal, startPt, rateTh, width_LB, avgRate_LB)
            
            # Use appropriate threshold based on curve type
            threshold = threshold_PC if type == 'PC' else threshold_T
            rlt = (diff >= threshold) 
            
            # Store metrics for all curves
            curve_metrics = {
                'sample_id': sample_id,
                'channel': ch,
                'type': type,
                'diff': diff,
                'cp': cp,
                'stepWidth': stepWidth,
                'avgRate': avgRate,
                'maxDiff': maxDiff,
                'amplified?': rlt,
                'threshold': threshold  # Store which threshold was used
            }
            
            # Append to falseDetectionList only if it's a false detection
            if not rlt and type != 'NEG':
                if type == 'PC':
                    ivCnt += 1
                    falseDetectionList.append(['IV', sample_id, ch, signal, curve_metrics])
                elif type == 'POSL':
                    fnLCnt += 1
                    falseDetectionList.append(['FNL', sample_id, ch, signal, curve_metrics])
                elif type == 'POSM':
                    fnMCnt += 1
                    falseDetectionList.append(['FNM', sample_id, ch, signal, curve_metrics])
                elif type == 'POSH':
                    fnHCnt += 1
                    falseDetectionList.append(['FNH', sample_id, ch, signal, curve_metrics])
            elif rlt and type == 'NEG':
                fpCnt += 1
                falseDetectionList.append(['FP', sample_id, ch, signal, curve_metrics])
            else:
                # More specific categorization of correctly identified curves
                # - Use 'TN' for True Negatives (correctly identified negative samples) 
                # - Use 'TP' for True Positives (correctly identified positive samples in ch2-ch5)
                # - Use 'VALID' for valid PC (correctly identified positive samples in ch1)
                if type == 'NEG':
                    category = 'TN'  # True Negative
                else:
                    # Check if it's from channel 1 (PC) or other channels
                    if ch == 'ch1':
                        category = 'VALID'  # Valid PC
                    else:
                        category = 'TP'  # True Positive
                falseDetectionList.append([category, sample_id, ch, signal, curve_metrics])
            
    logger.debug(f'startPt = {startPt}, rateTh = {rateTh}, width_LB = {width_LB}, avgRate_LB = {avgRate_LB}, threshold_PC = {threshold_PC}, threshold_T = {threshold_T}')
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
            # df[0] is the error type, df[1] is the sample_id (human-readable)
            sample_id = df[1]
            ch = df[2]
            signal = df[3]
            curve_label = f"{sample_id}_{ch}"
            
            # Calculate time series (x-axis)
            xSeries = np.arange(0, len(signal), 1)
            xSeries = np.interp(xSeries, (xSeries.min(), xSeries.max()), (0, 35))
            
            # Plot with color from palette (cycling through)
            color = color_palette[i % len(color_palette)]
            line, = ax.plot(xSeries, signal, label=curve_label, color=color, linewidth=2)
            
            # Calculate metrics for this curve for annotation
            steps, diff, cp, stepWidth, avgRate_val, maxDiff = labelSteps(signal, startPt, rate, 
                                                                width, avgRate)
            
            # Store metrics for CSV export
            curves_metrics.append({
                'Type': plotType,
                'SampleID': sample_id,
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

def save_false_detection_list(fdList, output_file=OUTPUT_FILE, params=None):
    """
    Save the false detection list and all curves metrics to CSV files
    
    Args:
        fdList (list): List of false detection data and all curve metrics
        output_file (str): Path to save the CSV file
        params (list, optional): Parameters used for detection
    """
    if not fdList:
        logger.warning(f"No detection data to save to {output_file}")
        return
    
    # Create DataFrame from the false detection list
    fd_data = []
    all_curves_data = []  # For storing all curves metrics
    
    for fd in fdList:
        # Extract details but exclude the signal data (fd[3]) which is too large for CSV
        fd_type = fd[0]  # FP, FNL, FNM, FNH, IV, PASS, TN
        sample_id = fd[1]  # Sample ID (human-readable)
        channel = fd[2]  # Channel (ch1, ch2, etc.)
        
        # If the curve_metrics are stored in the list item
        if len(fd) > 4 and isinstance(fd[4], dict):
            metrics = fd[4]
            curve_data = {
                'Type': fd_type,
                'SampleID': sample_id,
                'Channel': channel,
                'Diff': metrics['diff'],
                'Cp': metrics['cp'],
                'StepWidth': metrics['stepWidth'],
                'AvgRate': metrics['avgRate'],
                'MaxDiff': metrics['maxDiff'],
                'Amplified?': metrics['amplified?'],
                'Result': get_result_description(fd_type)
            }
            
            # All curves go to the complete metrics file
            all_curves_data.append(curve_data)
            
            # Only false detections go to the false detection list
            if fd_type in ['FP', 'FNL', 'FNM', 'FNH', 'IV']:
                fd_data.append(curve_data)
                
        # For backward compatibility with old format
        elif params:
            signal = fd[3]
            startPt, rateTh, width_LB, avgRate_LB, threshold = params
            _, diff, cp, stepWidth, avgRate, maxDiff = labelSteps(signal, startPt, rateTh, width_LB, avgRate_LB)
            
            curve_data = {
                'Type': fd_type,
                'SampleID': sample_id,
                'Channel': channel,
                'Diff': diff,
                'Cp': cp,
                'StepWidth': stepWidth,
                'AvgRate': avgRate,
                'MaxDiff': maxDiff,
                'Threshold': threshold,
                'Result': get_result_description(fd_type)
            }
            
            all_curves_data.append(curve_data)
            
            # Only false detections go to the false detection list
            if fd_type in ['FP', 'FNL', 'FNM', 'FNH', 'IV']:
                fd_data.append(curve_data)
        else:
            # Simplified output without metrics
            curve_data = {
                'Type': fd_type,
                'SampleID': sample_id,
                'Channel': channel,
                'Result': get_result_description(fd_type)
            }
            
            all_curves_data.append(curve_data)
            
            # Only false detections go to the false detection list
            if fd_type in ['FP', 'FNL', 'FNM', 'FNH', 'IV']:
                fd_data.append(curve_data)
    
    # Create DataFrames and save to CSV
    fd_df = pd.DataFrame(fd_data)
    fd_df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(fd_data)} false detections to {output_file}")
    
    # Save all curves metrics to a separate file
    all_metrics_output = output_file.replace('.csv', '_all_curves.csv')
    all_curves_df = pd.DataFrame(all_curves_data)
    all_curves_df.to_csv(all_metrics_output, index=False)
    logger.info(f"Saved metrics for {len(all_curves_data)} curves to {all_metrics_output}")
    
    return fd_df, all_curves_df

def get_result_description(fd_type):
    """
    Return a descriptive result based on the detection type.
    
    Args:
        fd_type (str): The detection type category
        
    Returns:
        str: A descriptive result string
    """
    result_descriptions = {
        'FP': 'False Positive',
        'FNL': 'False Negative (Low Conc.)',
        'FNM': 'False Negative (Medium Conc.)',
        'FNH': 'False Negative (High Conc.)',
        'IV': 'Invalid PC',
        'TP': 'True Positive',
        'VALID': 'Valid PC',
        'TN': 'True Negative'
    }
    
    return result_descriptions.get(fd_type, f'Unknown ({fd_type})')

# A manual function to calculate the curves metrics
# accepting a set of parameters for positive control (PC): startPt, rateTh, width_LB, avgRate_LB, threshold_PC, and a set of parameters for target (T): startPt, rateTh, width_LB, avgRate_LB, threshold_T
# based on the overall result rules: 
#    1) If positive control (PC) is invalid, count as invalid test
#    2) If PC is valid, any false positive curve per sampleID makes test a false positive
#    3) If PC is valid, all four channels (ch2, ch3, ch4, and ch5) must be false negatives to count as a false negative test
# Calculate the confusion matrix for the tests (ground truth for testlog based on testsGrouping)
def curvesMetric_manul(posCurves, negCurves, pcCurves, core_params, threshold_PC, threshold_T, cutoff_time=None):
    """
    Calculate test-level metrics using separate parameter sets for PC and target channels
    
    Args:
        posCurves (list): List containing [posCurvesL, posCurvesM, posCurvesH]
        negCurves (list): List of negative curves
        pcCurves (list): List of PC curves
        core_params (list): Core detection parameters [startPt, rateTh, width_LB, avgRate_LB]
        threshold_PC (float): Threshold for PC validation (channel 1)
        threshold_T (float): Threshold for target detection (channels 2-5)
        cutoff_time (float, optional): Time point (in minutes) after which data points are ignored.
            If None, all data points are used.
        
    Returns:
        tuple: Contains confusion matrix counts and detailed results
    """
    # Extract core parameters
    startPt, rateTh, width_LB, avgRate_LB = core_params
    
    # Initialize counters for confusion matrix
    tp_count = 0  # True positive tests
    tn_count = 0  # True negative tests
    fp_count = 0  # False positive tests
    fn_count = 0  # False negative tests
    iv_count = 0  # Invalid tests (invalid PC)
    
    # Unpack positive curves
    posCurvesL, posCurvesM, posCurvesH = posCurves
    
    # Create dictionaries to store PC evaluation results and target channel results by sample_id
    pc_results = {}  # {sample_id: is_valid}
    pos_target_results = {}  # {sample_id: {channel: is_positive}}
    neg_target_results = {}  # {sample_id: {channel: is_positive}}
    
    # Detailed results for all curves
    all_results = []
    
    # Function to trim signal based on cutoff time
    def trim_signal(signal, cutoff=None):
        if cutoff is None:
            return signal
        
        # Convert cutoff time to array index
        # Time formula: time = index * TIME_CONVERSION_FACTOR - TIME_OFFSET
        # So: index = (time + TIME_OFFSET) / TIME_CONVERSION_FACTOR
        cutoff_index = int((cutoff + TIME_OFFSET) / TIME_CONVERSION_FACTOR)
        
        # Ensure the cutoff index is within the valid range
        cutoff_index = max(1, min(cutoff_index, len(signal)))
        
        # Return the trimmed signal
        return signal[:cutoff_index]
    
    # First, evaluate all PC curves
    for pc in pcCurves:
        sample_id = pc[0]
        channel = pc[1]
        signal = pc[2]
        
        # Trim signal if cutoff_time is specified
        trimmed_signal = trim_signal(signal, cutoff_time)
        
        # Apply detection algorithm with core parameters
        steps, diff, cp, stepWidth, avgRate, maxDiff = labelSteps(
            trimmed_signal, startPt, rateTh, width_LB, avgRate_LB)
        
        # PC is valid if diff >= threshold_PC
        is_pc_valid = (diff >= threshold_PC)
        
        # Store PC result for this sample
        pc_results[sample_id] = is_pc_valid
        
        # Save detailed metrics
        curve_metrics = {
            'sample_id': sample_id,
            'channel': channel,
            'type': 'PC',
            'diff': diff,
            'cp': cp,
            'stepWidth': stepWidth,
            'avgRate': avgRate,
            'maxDiff': maxDiff,
            'amplified?': is_pc_valid,
            'test_result': 'Valid PC' if is_pc_valid else 'Invalid PC',
            'cutoff_time': cutoff_time,
            'threshold': threshold_PC
        }
        
        # Add to results list with category
        all_results.append(['VALID' if is_pc_valid else 'IV', sample_id, channel, trimmed_signal, curve_metrics])
    
    # Next, evaluate all positive target curves
    for curve_group in [posCurvesL, posCurvesM, posCurvesH]:
        for curve in curve_group:
            sample_id = curve[0]
            channel = curve[1]
            signal = curve[2]
            
            # Trim signal if cutoff_time is specified
            trimmed_signal = trim_signal(signal, cutoff_time)
            
            # Apply detection algorithm with target parameters
            steps, diff, cp, stepWidth, avgRate, maxDiff = labelSteps(
                trimmed_signal, startPt, rateTh, width_LB, avgRate_LB)
            
            # Target is positive if diff >= threshold_T
            is_positive = (diff >= threshold_T)
            
            # Initialize dictionary for this sample if not exists
            if sample_id not in pos_target_results:
                pos_target_results[sample_id] = {}
            
            # Store result for this channel
            pos_target_results[sample_id][channel] = is_positive
            
            # Save detailed metrics
            curve_metrics = {
                'sample_id': sample_id,
                'channel': channel,
                'type': 'TARGET_POS',
                'diff': diff,
                'cp': cp,
                'stepWidth': stepWidth,
                'avgRate': avgRate,
                'maxDiff': maxDiff,
                'amplified?': is_positive,
                'test_result': 'True Positive' if is_positive else 'False Negative',
                'cutoff_time': cutoff_time,
                'threshold': threshold_T
            }
            
            # Add to results list with appropriate category
            category = 'TP' if is_positive else 'FN'
            all_results.append([category, sample_id, channel, trimmed_signal, curve_metrics])
    
    # Evaluate all negative target curves
    for curve in negCurves:
        sample_id = curve[0]
        channel = curve[1]
        signal = curve[2]
        
        # Trim signal if cutoff_time is specified
        trimmed_signal = trim_signal(signal, cutoff_time)
        
        # Apply detection algorithm with target parameters
        steps, diff, cp, stepWidth, avgRate, maxDiff = labelSteps(
            trimmed_signal, startPt, rateTh, width_LB, avgRate_LB)
        
        # Target should be negative (is_positive should be False)
        is_positive = (diff >= threshold_T)
        
        # Initialize dictionary for this sample if not exists
        if sample_id not in neg_target_results:
            neg_target_results[sample_id] = {}
        
        # Store result for this channel
        neg_target_results[sample_id][channel] = is_positive
        
        # Save detailed metrics
        curve_metrics = {
            'sample_id': sample_id,
            'channel': channel,
            'type': 'TARGET_NEG',
            'diff': diff,
            'cp': cp,
            'stepWidth': stepWidth,
            'avgRate': avgRate,
            'maxDiff': maxDiff,
            'amplified?': not is_positive,  # For negatives, qualified means NOT positive
            'test_result': 'True Negative' if not is_positive else 'False Positive',
            'cutoff_time': cutoff_time,
            'threshold': threshold_T
        }
        
        # Add to results list with appropriate category
        category = 'TN' if not is_positive else 'FP'
        all_results.append([category, sample_id, channel, trimmed_signal, curve_metrics])
    
    # Now evaluate overall test results based on the rules
    
    # Track samples that have been counted
    counted_samples = set()
    
    # Process positive samples
    for sample_id, channel_results in pos_target_results.items():
        # Skip if already counted or no PC result
        if sample_id in counted_samples or sample_id not in pc_results:
            continue
        
        # Rule 1: If PC is invalid, count as invalid test
        if not pc_results[sample_id]:
            iv_count += 1
            # Add detailed logging to track tests with data but invalid PC
            logger.info(f"Sample {sample_id} has data but invalid PC - counted as invalid test")
            counted_samples.add(sample_id)
            continue
        
        # Rule 3: If all four channels are false negatives, count as false negative test
        # First check if we have results for channels 2-5
        has_all_channels = all(f'ch{i}' in channel_results for i in range(2, 6))
        
        if has_all_channels:
            # Check if all channels are negative
            all_negative = all(not is_positive for channel, is_positive in channel_results.items())
            
            if all_negative:
                fn_count += 1
            else:
                tp_count += 1
        else:
            # If we don't have all channels, check if any is positive
            any_positive = any(is_positive for channel, is_positive in channel_results.items())
            
            if any_positive:
                tp_count += 1
            else:
                # If we only have negative results but not all channels, count as false negative
                fn_count += 1
        
        counted_samples.add(sample_id)
    
    # Process negative samples
    for sample_id, channel_results in neg_target_results.items():
        # Skip if already counted or no PC result
        if sample_id in counted_samples or sample_id not in pc_results:
            continue
        
        # Rule 1: If PC is invalid, count as invalid test
        if not pc_results[sample_id]:
            iv_count += 1
            # Add detailed logging to track tests with data but invalid PC
            logger.info(f"Sample {sample_id} has data but invalid PC - counted as invalid test")
            counted_samples.add(sample_id)
            continue
        
        # Rule 2: If any channel is false positive, count as false positive test
        any_positive = any(is_positive for channel, is_positive in channel_results.items())
        
        if any_positive:
            fp_count += 1
        else:
            tn_count += 1
        
        counted_samples.add(sample_id)
    
    # Handle samples that only have PC results (no target channels)
    for sample_id, is_valid in pc_results.items():
        if sample_id in counted_samples:
            continue
        
        if not is_valid:
            iv_count += 1
            # Add detailed logging to track tests with data but invalid PC
            logger.info(f"Sample {sample_id} has data but invalid PC - counted as invalid test")
            counted_samples.add(sample_id)
        else:
            # Count valid PC with no target channels as indeterminate (without logging)
            iv_count += 1  # Consider as invalid/indeterminate
            counted_samples.add(sample_id)
    
    # After processing PC results, check for tests with target data but no PC results
    # These are still valid tests that need to be counted
    
    # First, find all sample IDs with target channel data
    target_sample_ids = set(sample_id for sample_id in pos_target_results.keys()) | \
                       set(sample_id for sample_id in neg_target_results.keys())
    
    # Find samples with target data but no PC results
    samples_with_target_no_pc = target_sample_ids - set(pc_results.keys())
    
    for sample_id in samples_with_target_no_pc:
        if sample_id in counted_samples:
            continue
            
        # Count these as invalid tests due to missing PC
        iv_count += 1
        logger.info(f"Sample {sample_id} has target data but no PC data - counted as invalid test")
        counted_samples.add(sample_id)
    
    # Track samples with no data at all, but DO NOT count them as invalid
    all_sample_ids = set(posTests.keys()) | set(negTests.keys())
    missing_samples = all_sample_ids - counted_samples
    for sample_id in missing_samples:
        # Note: not incrementing iv_count anymore
        counted_samples.add(sample_id)
    
    # Print the parameters
    logger.info(f'Core parameters: startPt={startPt}, rateTh={rateTh}, width_LB={width_LB}, avgRate_LB={avgRate_LB}')
    logger.info(f'PC threshold: {threshold_PC}, Target threshold: {threshold_T}')
    if cutoff_time is not None:
        logger.info(f'Using cutoff time: {cutoff_time} minutes')
    
    # Print results for debugging
    logger.info(f'Test-level results: TP={tp_count}, TN={tn_count}, FP={fp_count}, FN={fn_count}, IV={iv_count}')
    # Print the confusion matrix in precision, recall, F1 score, and accuracy
    precision = round(tp_count / (tp_count + fp_count), 2) if (tp_count + fp_count) > 0 else 0
    recall = round(tp_count / (tp_count + fn_count), 2) if (tp_count + fn_count) > 0 else 0
    f1_score = round(2 * precision * recall / (precision + recall), 2) if (precision + recall) > 0 else 0
    accuracy = round((tp_count + tn_count) / (tp_count + tn_count + fp_count + fn_count), 2) if (tp_count + tn_count + fp_count + fn_count) > 0 else 0
    logger.info(f'Precision: {precision}, Recall: {recall}, F1 score: {f1_score}, Accuracy: {accuracy}')
    
    # Print confusion matrix as a table
    logger.info("Confusion Matrix:")
    logger.info(f"{'=' * 54}")
    logger.info(f"| {'':<16} | {'Actual Positive':<12} | {'Actual Negative':<12} |")
    logger.info(f"|{'-' * 18}|{'-' * 14}|{'-' * 14}|")
    logger.info(f"| {'Predicted Pos':<16} | {tp_count:<12} | {fp_count:<12} |")
    logger.info(f"| {'Predicted Neg':<16} | {fn_count:<12} | {tn_count:<12} |")
    logger.info(f"{'=' * 54}")
    
    # Print metrics table
    logger.info("Performance Metrics:") 
    logger.info(f"{'=' * 32}")
    logger.info(f"| {'Metric':<12} | {'Value':<10} |")
    logger.info(f"|{'-' * 14}|{'-' * 12}|")
    logger.info(f"| {'Precision':<12} | {precision:<10.2f} |")
    logger.info(f"| {'Recall':<12} | {recall:<10.2f} |")
    logger.info(f"| {'F1 Score':<12} | {f1_score:<10.2f} |")
    logger.info(f"| {'Accuracy':<12} | {accuracy:<10.2f} |")
    logger.info(f"| {'Invalid':<12} | {iv_count:<10} |")
    logger.info(f"{'=' * 32}")
    
    return tp_count, tn_count, fp_count, fn_count, iv_count, all_results

if __name__ == "__main__":
    # Import and group tests
    posTests, negTests, outliers = testsGrouping(TESTLOGFILE)

    # Get curves and track missing PC info
    negCurves, pcNTC, neg_missing_pc = NTCMetric(negTests, DATAPATH)
    posCurvesL, posCurvesM, posCurvesH, pcPOS, pos_missing_pc = POSMetric(posTests, DATAPATH)
    posCurves = [posCurvesL, posCurvesM, posCurvesH]
    pcCurves = pcNTC + pcPOS

    # Define parameters
    op_rateTh = 0.67
    op_width_LB = 15
    op_avgRate_LB = 1.62
    op_threshold_PC = 90
    op_threshold_T = 100
    
    # Core parameters common to both PC and target
    core_params = [DEFAULT_START_PT, op_rateTh, op_width_LB, op_avgRate_LB]
    
    # Calculate the curves metrics
    fpCnt, fnHCnt, fnMCnt, fnLCnt, ivCnt, _ = curvesMetric_manul(
        posCurves, negCurves, pcCurves, 
        core_params, op_threshold_PC, op_threshold_T, 
        CUTOFF_TIME
    )

    