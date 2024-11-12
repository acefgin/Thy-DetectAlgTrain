from scipy.optimize import minimize
from detectAlgBenchmark import curvesMetric, testsGrouping, NTCMetric, POSMetric, plotFalseDetectionCurves
from detectAlgBenchmark import DATAPATH, TESTLOGFILE, argBounds, PlotFalse
import logging

# Configure logging
logger = logging.getLogger()
logger.info("======== Start ADF parameters optimization ========")

# Import and group tests
posTests, negTests, outliers = testsGrouping(TESTLOGFILE)

# Log parameters bounds
logger.info(f"### startPt|rateTh|width_LB|avgRate_LB|threshold: {argBounds} ###")

def objective_function_ivCnt(params):

    # Constrain width_LB to be an integer
    params[2] = int(round(params[2]))
    
    # Get necessary data from detectAlgBenchmark.py
    negCurves, pcNTC = NTCMetric(negTests, DATAPATH)
    posCurvesL, posCurvesM, posCurvesH, pcPOS = POSMetric(posTests, DATAPATH)
    posCurves = [posCurvesL, posCurvesM, posCurvesH]
    pcCurves = pcNTC + pcPOS
    
    # Calculate metrics using curvesMetric function
    _, _, _, _, ivCnt, _ = curvesMetric(posCurves, negCurves, pcCurves, params)
    
    # Return ivCnt as optimization objective
    return ivCnt

def objective_function_fp_fn(params):

    # Constrain width_LB to be an integer
    params[2] = int(round(params[2]))

    # Get necessary data from detectAlgBenchmark.py
    negCurves, pcNTC = NTCMetric(negTests, DATAPATH)
    posCurvesL, posCurvesM, posCurvesH, pcPOS = POSMetric(posTests, DATAPATH)
    posCurves = [posCurvesL, posCurvesM, posCurvesH]
    pcCurves = pcNTC + pcPOS
    
    # Calculate metrics using curvesMetric function
    fpCnt, fnHCnt, fnMCnt, fnLCnt, _, _ = curvesMetric(posCurves, negCurves, pcCurves, params)
    
    # Calculate total false curves count as optimization objective
    false_curves_cnt = (fpCnt + fnHCnt + fnMCnt + fnLCnt)
    
    return false_curves_cnt

# Define parameter bounds
bounds_str = argBounds.split('|')
bounds = []
for bound_str in bounds_str:
    min_val, max_val = map(float, bound_str.split(','))
    bounds.append((min_val, max_val))

# Initial guess
initial_guess = [75, 0.5, 15, 0.5, 15]

# Optimize using scipy.optimize.minimize
# Try different optimization methods
methods = ['Nelder-Mead', 'Powell', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']

# First optimize objective_function_fp_fn
best_result_fp_fn = None
best_error_fp_fn = float('inf')
best_method_fp_fn = None

print("Training ADF parameters... Please wait...")
for method in methods:
    try:
        logger.info(f"Trying method for fp_fn: {method}")
        result = minimize(objective_function_fp_fn, initial_guess, method=method, bounds=bounds)
        if result.success and result.fun < best_error_fp_fn:
            best_result_fp_fn = result
            best_error_fp_fn = result.fun
            best_method_fp_fn = method
    except:
        continue

result_fp_fn = best_result_fp_fn if best_result_fp_fn else minimize(objective_function_fp_fn, initial_guess, method='L-BFGS-B', bounds=bounds)

logger.info(f"Best method for fp_fn: {best_method_fp_fn}")

# Output optimization results for fp_fn
logger.info("======== FP_FN Optimization Results ========")
logger.info("Optimized parameters for fp_fn:")
logger.info(f"startPt: {result_fp_fn.x[0]:.2f}")
logger.info(f"rateTh: {result_fp_fn.x[1]:.2f}")
logger.info(f"width_LB: {int(result_fp_fn.x[2])}")
logger.info(f"avgRate_LB: {result_fp_fn.x[3]:.2f}")
logger.info(f"threshold: {result_fp_fn.x[4]:.2f}")
logger.info(f"Minimum false curves count: {result_fp_fn.fun:.4f}")
logger.info("======== End of FP_FN Optimization ========")

# Optimize width_LB, avgRate_LB and threshold for objective_function_ivCnt
def objective_function_ivCnt_params(params):
    full_params = [result_fp_fn.x[0], params[0], int(params[1]), params[2], params[3]]
    return objective_function_ivCnt(full_params)

# Optimize objective_function_ivCnt
best_result_ivCnt = None 
best_error_ivCnt = float('inf')
best_method_ivCnt = None

initial_params = [result_fp_fn.x[1], result_fp_fn.x[2], 0.5, 15]
param_bounds = [bounds[1], bounds[2], bounds[3], bounds[4]]

for method in methods:
    try:
        logger.info(f"Trying method for ivCnt: {method}")
        result = minimize(objective_function_ivCnt_params,
                          initial_params,  # Initial guess for 3 parameters
                          method=method,
                          bounds=param_bounds)  # Bounds for 3 parameters
        if result.success and result.fun < best_error_ivCnt:
            best_result_ivCnt = result
            best_error_ivCnt = result.fun
            best_method_ivCnt = method
    except:
        continue

result_ivCnt = best_result_ivCnt if best_result_ivCnt else minimize(objective_function_ivCnt_params,
                                                                    initial_params,
                                                                    method='L-BFGS-B',
                                                                    bounds=param_bounds)

logger.info(f"Best method for ivCnt: {best_method_ivCnt}")

# Output optimization results for ivCnt
logger.info("======== PC Optimization Results ========")
logger.info(f"startPt: {result_fp_fn.x[0]:.2f}")
logger.info(f"rateTh: {result_ivCnt.x[0]:.2f}")
logger.info(f"width_LB: {int(result_ivCnt.x[1])}")
logger.info(f"avgRate_LB: {result_ivCnt.x[2]:.2f}")
logger.info(f"threshold: {result_ivCnt.x[3]:.2f}")
logger.info(f"Minimum invalid PC count: {result_ivCnt.fun:.4f}")
logger.info("======== End of PC Optimization Results ========")

def log_false_detections(params, method_name, error_type="FP_FN"):
    negCurves, pcNTC = NTCMetric(negTests, DATAPATH)
    posCurvesL, posCurvesM, posCurvesH, pcPOS = POSMetric(posTests, DATAPATH)
    posCurves = [posCurvesL, posCurvesM, posCurvesH]
    pcCurves = pcNTC + pcPOS
    
    _, _, _, _, _, fdList = curvesMetric(posCurves, negCurves, pcCurves, params)
    
    logger.info(f"====== False Detection Details for {error_type} - Method: {method_name} ======")
    logger.info(f"Parameters: startPt={params[0]:.2f}, rateTh={params[1]:.2f}, "
               f"width_LB={int(params[2])}, avgRate_LB={params[3]:.2f}, threshold={params[4]:.2f}")
    cnt = 0
    for fd in fdList:
        if error_type == "FP_FN" and fd[0] == 'IV':
            continue
        if error_type == "PC" and fd[0] != 'IV':
            continue
        logger.info(f"Type: {fd[0]}, TestID: {fd[1]}, Channel: {fd[2]}")
        cnt += 1
    logger.info(f"Total false detections: {cnt}")
    logger.info("=" * 60)

    if PlotFalse:
        if error_type == "FP_FN":
            plotFalseDetectionCurves(fdList, {'FP', 'FNL', 'FNM', 'FNH'}, params)
        elif error_type == "PC":
            plotFalseDetectionCurves(fdList, {'IV'}, params)

# After finding best_result_fp_fn
if best_result_fp_fn:
    log_false_detections(best_result_fp_fn.x, best_method_fp_fn, "FP_FN")

# After finding best_result_ivCnt
if best_result_ivCnt:
    full_params = [result_fp_fn.x[0], result_ivCnt.x[0], int(result_ivCnt.x[1]), 
                   result_ivCnt.x[2], result_ivCnt.x[3]]
    log_false_detections(full_params, best_method_ivCnt, "PC")

print("Task completed! Please check the log file for details.")
input("Press Enter to exit...")

