import os
# Set NumExpr to use 16 cores
os.environ["NUMEXPR_MAX_THREADS"] = "16"

import logging

# Import initialization function first
from config import init_config

# Initialize configuration before importing other modules that use these variables
args = init_config()

# Now import modules that depend on initialized configuration
from config import DATAPATH, TESTLOGFILE, OUTPUT_FILE, PlotFalse, SMOOTH_METHOD, ADAPTIVE_SMOOTH, CUTOFF_TIME
from data_loader import testsGrouping, NTCMetric, POSMetric
from detection import curvesMetric, curvesMetric_manul
from visualization import plotFalseDetectionCurves
from export import save_false_detection_list, save_dual_threshold_results

def main():
    """Main entry point for the ADF detection algorithm."""
    # Get logger
    logger = logging.getLogger()
    
    # Import and group tests
    logger.info("Loading and grouping tests from test log...")
    posTests, negTests, outliers = testsGrouping(TESTLOGFILE)
    logger.info(f"Found {len(posTests)} positive tests, {len(negTests)} negative tests, and {len(outliers)} outliers")
    
    # Get curves and track missing PC info
    logger.info("Processing negative test data...")
    negCurves, pcNTC, neg_missing_pc = NTCMetric(negTests, DATAPATH, SMOOTH_METHOD, ADAPTIVE_SMOOTH)
    
    logger.info("Processing positive test data...")
    posCurvesL, posCurvesM, posCurvesH, pcPOS, pos_missing_pc = POSMetric(posTests, DATAPATH, SMOOTH_METHOD, ADAPTIVE_SMOOTH)
    
    posCurves = [posCurvesL, posCurvesM, posCurvesH]
    pcCurves = pcNTC + pcPOS
    
    logger.info(f"PC curve count: {len(pcCurves)}")
    
    # Define parameters
    op_rateTh = 0.67
    op_width_LB = 15
    op_avgRate_LB = 1.62
    op_threshold_PC = 90
    op_threshold_T = 100
    
    # Core parameters common to both PC and target
    core_params = [75, op_rateTh, op_width_LB, op_avgRate_LB]
    
    # Calculate metrics using curvesMetric
    logger.info("Calculating detection metrics with default parameters...")
    fpCnt, fnHCnt, fnMCnt, fnLCnt, ivCnt, fdList = curvesMetric(
        posCurves, negCurves, pcCurves, 
        core_params, op_threshold_PC, op_threshold_T
    )
    
    logger.info(f"False Positive count: {fpCnt}")
    logger.info(f"False Negative High count: {fnHCnt}")
    logger.info(f"False Negative Medium count: {fnMCnt}")
    logger.info(f"False Negative Low count: {fnLCnt}")
    logger.info(f"Invalid PC count: {ivCnt}")
    
    # Save the false detection list
    logger.info(f"Saving detection results to {OUTPUT_FILE}...")
    save_false_detection_list(fdList, OUTPUT_FILE, core_params + [op_threshold_PC])
    
    # Calculate and display test-level metrics
    logger.info("Calculating test-level metrics...")
    tp_count, tn_count, fp_count, fn_count, iv_count, all_results = curvesMetric_manul(
        posCurves, negCurves, pcCurves, 
        core_params, op_threshold_PC, op_threshold_T, 
        CUTOFF_TIME
    )
    
    # Calculate and print metrics
    precision = round(tp_count / (tp_count + fp_count), 2) if (tp_count + fp_count) > 0 else 0
    recall = round(tp_count / (tp_count + fn_count), 2) if (tp_count + fn_count) > 0 else 0
    f1_score = round(2 * precision * recall / (precision + recall), 2) if (precision + recall) > 0 else 0
    accuracy = round((tp_count + tn_count) / (tp_count + tn_count + fp_count + fn_count), 2) if (tp_count + tn_count + fp_count + fn_count) > 0 else 0
    
    # Display metrics to console
    print("\nTest-level Performance Results:")
    print(f"Parameters: startPt={core_params[0]}, rateTh={core_params[1]:.2f}, width_LB={core_params[2]}, avgRate_LB={core_params[3]:.2f}")
    print(f"PC threshold: {op_threshold_PC}, Target threshold: {op_threshold_T}")
    
    # Print confusion matrix as a table
    print("\nConfusion Matrix:")
    print("=" * 54)
    print(f"| {'':<16} | {'Actual Positive':<12} | {'Actual Negative':<12} |")
    print(f"|{'-' * 18}|{'-' * 14}|{'-' * 14}|")
    print(f"| {'Predicted Pos':<16} | {tp_count:<12} | {fp_count:<12} |")
    print(f"| {'Predicted Neg':<16} | {fn_count:<12} | {tn_count:<12} |")
    print("=" * 54)
    
    # Print metrics table
    print("\nPerformance Metrics:")
    print("=" * 32)
    print(f"| {'Metric':<12} | {'Value':<10} |")
    print(f"|{'-' * 14}|{'-' * 12}|")
    print(f"| {'Precision':<12} | {precision:<10.2f} |")
    print(f"| {'Recall':<12} | {recall:<10.2f} |")
    print(f"| {'F1 Score':<12} | {f1_score:<10.2f} |")
    print(f"| {'Accuracy':<12} | {accuracy:<10.2f} |")
    print(f"| {'Invalid':<12} | {iv_count:<10} |")
    print("=" * 32)
    
    # Generate plots if requested
    if PlotFalse:
        logger.info("Generating plots for false detections...")
        plot_params = core_params + [op_threshold_T]
        
        for plot_type in ['FP', 'FNL', 'FNM', 'FNH', 'IV']:
            # Count how many curves of this type we have
            type_curves = [fd for fd in fdList if fd[0] == plot_type]
            if type_curves:
                logger.info(f"Plotting {len(type_curves)} curves of type {plot_type}")
                plotFalseDetectionCurves(fdList, plot_type, plot_params)
    
    logger.info("Processing completed successfully.")

if __name__ == "__main__":
    main() 