import numpy as np
import logging
from config import TIME_CONVERSION_FACTOR, TIME_OFFSET, DEFAULT_START_PT, DEFAULT_RATE_TH, DEFAULT_WIDTH_LB, DEFAULT_AVG_RATE_LB, DEFAULT_THRESHOLD

# Get logger
logger = logging.getLogger()

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

    stepDiff_nonLAMP = 0
    cp_nonLAMP = 0
    maxIndex_nonLAMP = 0
    maxDiff_nonLAMP = 0
    stepWidth_nonLAMP = 0
    avgRate_nonLAMP = 0  # Initialize here to avoid reference before assignment
    
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
        else:
            index = step[0] - 1
            stepWidth_nonLAMP += step[1] - step[0] + 1
            # Calculate metrics for non-LAMP steps
            while index < step[1] + 1:
                stepDiff_nonLAMP = stepDiff_nonLAMP + dataDiffs[index]
                 # Capture time for highest diff as Cp
                if dataDiffs[index] >= maxDiff_nonLAMP:
                    maxDiff_nonLAMP = dataDiffs[index]
                    maxIndex_nonLAMP = index
                index += 1

            # Calculate Cp: Adjusts maxIndex by the ratio of signal to rate at that point
            # Then converts to minutes using TIME_CONVERSION_FACTOR and adjusts by TIME_OFFSET
            if len(datas) > 10 and maxIndex_nonLAMP < len(datas)-1 and maxIndex_nonLAMP < len(dataDiffs):
                # Adjust index by the ratio of signal to rate
                adjusted_index = maxIndex_nonLAMP
                if dataDiffs[maxIndex_nonLAMP] > 0:  # Prevent division by zero
                    adjusted_index = maxIndex_nonLAMP - datas[maxIndex_nonLAMP + 1] / dataDiffs[maxIndex_nonLAMP]
                # Convert to minutes
                cp_nonLAMP = adjusted_index * TIME_CONVERSION_FACTOR - TIME_OFFSET
                
    avgRate = 0
    if stepWidth != 0: 
        avgRate = stepDiff/stepWidth
    if stepWidth_nonLAMP != 0:
        avgRate_nonLAMP = stepDiff_nonLAMP/stepWidth_nonLAMP
    
    LAMP_step = False
    if stepWidth != 0:
        LAMP_step = True
        ans = [listOfSteps, np.round(stepDiff, 1), round(cp, 1), round(stepWidth, 1), round(avgRate, 1), np.round(maxDiff, 1), LAMP_step]
    else:
        LAMP_step = False
        ans = [listOfSteps, np.round(stepDiff_nonLAMP, 1), round(cp_nonLAMP, 1), round(stepWidth_nonLAMP, 1), round(avgRate_nonLAMP, 1), np.round(maxDiff_nonLAMP, 1), LAMP_step]

    return ans

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
            steps, diff, cp, stepWidth, avgRate, maxDiff, LAMP_step = labelSteps(signal, startPt, rateTh, width_LB, avgRate_LB)
            
            # Use appropriate threshold based on curve type
            threshold = threshold_PC if type == 'PC' else threshold_T
            rlt = (diff >= threshold) and LAMP_step
            
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
    from utils import trim_signal
    
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
    
    # First, evaluate all PC curves
    for pc in pcCurves:
        sample_id = pc[0]
        channel = pc[1]
        signal = pc[2]
        
        # Trim signal if cutoff_time is specified
        trimmed_signal = trim_signal(signal, cutoff_time)
        
        # Apply detection algorithm with core parameters
        steps, diff, cp, stepWidth, avgRate, maxDiff, LAMP_step = labelSteps(
            trimmed_signal, startPt, rateTh, width_LB, avgRate_LB)
        
        # PC is valid if diff >= threshold_PC
        is_pc_valid = (diff >= threshold_PC) and LAMP_step
        
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
            steps, diff, cp, stepWidth, avgRate, maxDiff, LAMP_step = labelSteps(
                trimmed_signal, startPt, rateTh, width_LB, avgRate_LB)
            
            # Target is positive if diff >= threshold_T
            is_positive = (diff >= threshold_T) and LAMP_step
            
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
        steps, diff, cp, stepWidth, avgRate, maxDiff, LAMP_step = labelSteps(
            trimmed_signal, startPt, rateTh, width_LB, avgRate_LB)
        
        # Target should be negative (is_positive should be False)
        is_positive = (diff >= threshold_T) and LAMP_step
        
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