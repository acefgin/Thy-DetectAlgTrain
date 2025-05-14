import pandas as pd
import logging
import os

from utils import get_result_description

# Get logger
logger = logging.getLogger()

def save_false_detection_list(fdList, output_file="falseDetectionList.csv", params=None):
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
            from detection import labelSteps
            
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

def save_dual_threshold_results(pc_fdList, target_fdList, core_params, threshold_PC, threshold_T, output_file="falseDetectionList.csv"):
    """
    Combine results from PC and target evaluations and save to file with dual threshold information
    
    Args:
        pc_fdList (list): False detection list from PC evaluation
        target_fdList (list): False detection list from target evaluation
        core_params (list): Core parameters [startPt, rateTh, width_LB, avgRate_LB]
        threshold_PC (float): Threshold for PC validation
        threshold_T (float): Threshold for target detection
        output_file (str): Path to save the CSV file
    """
    # Create a unique key for each result based on Type, SampleID, and Channel
    # to help identify and eliminate duplicates
    unique_entries = {}
    all_fdList = []
    
    # Process PC list first
    for fd in pc_fdList or []:
        if fd and len(fd) >= 3:
            key = (fd[0], fd[1], fd[2])  # (Type, SampleID, Channel)
            unique_entries[key] = fd
    
    # Process target list, only adding non-duplicates
    for fd in target_fdList or []:
        if fd and len(fd) >= 3:
            key = (fd[0], fd[1], fd[2])  # (Type, SampleID, Channel)
            unique_entries[key] = fd
    
    # Convert back to list
    all_fdList = list(unique_entries.values())
    
    # Create a params list that includes the thresholds
    save_params = core_params.copy()
    save_params.append(f"{threshold_PC:.2f}(PC)/{threshold_T:.2f}(T)")  # Store both thresholds
    
    # Save the combined results
    fd_df, all_curves_df = save_false_detection_list(all_fdList, output_file, save_params)
    
    # Log the results
    ivCnt_PC = len([fd for fd in all_fdList if fd[0] == 'IV'])
    fpCnt = len([fd for fd in all_fdList if fd[0] == 'FP'])
    fnLCnt = len([fd for fd in all_fdList if fd[0] == 'FNL'])
    fnMCnt = len([fd for fd in all_fdList if fd[0] == 'FNM'])
    fnHCnt = len([fd for fd in all_fdList if fd[0] == 'FNH'])
    
    logger.info(f"Saved combined false detection list to {output_file}")
    logger.info(f"PC Invalid Count: {ivCnt_PC}, Target FP+FN Count: {fpCnt+fnHCnt+fnMCnt+fnLCnt}")
    
    return all_fdList 