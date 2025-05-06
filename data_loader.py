import os
import csv
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from config import DEFAULT_LAYOUT, DEFAULT_CONC, LOW_CONC, MEDIUM_CONC, HIGH_CONC
from utils import smooth

# Get logger
logger = logging.getLogger()

def readRunCsv(filename, smooth_method='adaptive', adaptive_smooth=True):
    """
    Read and parse a run CSV file to extract test information and signal data.
    
    Args:
        filename (str): Path to the CSV file to read
        smooth_method (str): Signal smoothing method to use
        adaptive_smooth (bool): Whether to use adaptive smoothing
        
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
                        signalList.append(smooth(np.array(signal_data), method=smooth_method, adaptive=adaptive_smooth))
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

def NTCMetric(negTests, dataPath, smooth_method='adaptive', adaptive_smooth=True):
    """
    Process negative control test data considering layout information
    
    Args:
        negTests (dict): Dictionary of negative tests with layout information
        dataPath (Path): Path to the data directory
        smooth_method (str): Signal smoothing method to use
        adaptive_smooth (bool): Whether to use adaptive smoothing
        
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
        test_info_from_file, _, signalList = readRunCsv(filename, smooth_method, adaptive_smooth)
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
                
def POSMetric(posTests, dataPath, smooth_method='adaptive', adaptive_smooth=True):
    """
    Process positive test data considering layout information
    
    Args:
        posTests (dict): Dictionary of positive tests with layout and concentration
        dataPath (Path): Path to the data directory
        smooth_method (str): Signal smoothing method to use
        adaptive_smooth (bool): Whether to use adaptive smoothing
        
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
        test_info_from_file, _, signalList = readRunCsv(filename, smooth_method, adaptive_smooth)
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