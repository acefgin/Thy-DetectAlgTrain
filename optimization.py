from scipy.optimize import minimize
from detectAlgBenchmark import curvesMetric, testsGrouping, NTCMetric, POSMetric, plotFalseDetectionCurves
from detectAlgBenchmark import DATAPATH, TESTLOGFILE, argBounds, PlotFalse, OUTPUT_FILE, save_false_detection_list
import logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import os
import numba

# Configure logging
logger = logging.getLogger()

# Import and group tests
posTests, negTests, outliers = testsGrouping(TESTLOGFILE)

# Get curves and track missing PC info
negCurves, pcNTC, neg_missing_pc = NTCMetric(negTests, DATAPATH)
posCurvesL, posCurvesM, posCurvesH, pcPOS, pos_missing_pc = POSMetric(posTests, DATAPATH)
posCurves = [posCurvesL, posCurvesM, posCurvesH]
pcCurves = pcNTC + pcPOS

logger.info(f"Number of PC curves: {len(pcCurves)}")

# Log parameters bounds
logger.info(f"### startPt|rateTh|width_LB|avgRate_LB|threshold: {argBounds} ###")

# ----------------- CACHING IMPLEMENTATION -----------------
@lru_cache(maxsize=1024)
def cached_curves_metric(params_tuple):
    """Cached version of curvesMetric function to avoid redundant calculations"""
    # Convert tuple to list for the actual function
    params = list(params_tuple)
    # Constrain width_LB to be an integer
    params[2] = int(round(params[2]))
    return curvesMetric(posCurves, negCurves, pcCurves, params)

# Modify your objective functions to use caching
def objective_function_fp_fn(params):
    # Make params hashable for caching
    params_tuple = tuple(map(float, params))
    
    # Get metrics from cached function
    fpCnt, fnHCnt, fnMCnt, fnLCnt, ivCnt, _ = cached_curves_metric(params_tuple)
    
    # Keep your original weighting logic
    weights = {
        'fp': 1.0,
        'fnH': 1.0,
        'fnM': 1.0,
        'fnL': 1.0,
        'iv': 1.0
    }
    
    false_curves_cnt = (
        weights['fp'] * fpCnt +
        weights['fnH'] * fnHCnt +
        weights['fnM'] * fnMCnt + 
        weights['fnL'] * fnLCnt
    )
    
    return false_curves_cnt

# Define parameter bounds
bounds_str = argBounds.split('|')
bounds = []
for bound_str in bounds_str:
    min_val, max_val = map(float, bound_str.split(','))
    bounds.append((min_val, max_val))

# Create scaled bounds (all 0 to 1) for the optimization
scaled_bounds = [(0, 1) for _ in range(len(bounds))]

# Create a scaled objective function
def scaled_objective(scaled_params):
    # Convert scaled parameters back to original scale
    actual_params = [
        scaled_params[0] * (bounds[0][1] - bounds[0][0]) + bounds[0][0],
        scaled_params[1] * (bounds[1][1] - bounds[1][0]) + bounds[1][0],
        scaled_params[2] * (bounds[2][1] - bounds[2][0]) + bounds[2][0],
        scaled_params[3] * (bounds[3][1] - bounds[3][0]) + bounds[3][0],
        scaled_params[4] * (bounds[4][1] - bounds[4][0]) + bounds[4][0]
    ]
    return objective_function_fp_fn(actual_params)

# Function to convert from original scale to scaled (0-1) parameters
def scale_params(params):
    scaled = []
    for i, param in enumerate(params):
        # Check if bounds are identical to avoid division by zero
        if bounds[i][0] == bounds[i][1]:
            scaled.append(0.5)  # Use 0.5 as default for fixed parameters
        else:
            scaled.append((param - bounds[i][0]) / (bounds[i][1] - bounds[i][0]))
    return scaled

# Function to convert from scaled (0-1) parameters to original scale
def unscale_params(scaled_params):
    unscaled = []
    for i, param in enumerate(scaled_params):
        # Check if bounds are identical to avoid unnecessary calculation
        if bounds[i][0] == bounds[i][1]:
            unscaled.append(bounds[i][0])  # Use the fixed value
        else:
            value = param * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
            # Round width_LB to integer (parameter at index 2)
            if i == 2 or i == 0:
                value = int(round(value))
            unscaled.append(value)
    return unscaled

# Use multiple initial guesses to avoid local minima (already in scaled form)
initial_guesses_orig = [
    [75, 0.5, 15, 0.5, 15],    # Original
    [75, 5.0, 25, 2.5, 100],   # Middle values
    [75, 9.0, 35, 4.5, 200]    # Near upper bounds
]

# Scale the initial guesses
initial_guesses = [scale_params(guess) for guess in initial_guesses_orig]

# Define optimization methods to try
methods = ['Nelder-Mead', 'Powell', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']

# Add global optimization methods
from scipy.optimize import differential_evolution, dual_annealing, basinhopping

# Track best result across all methods and initial guesses
best_result = None
best_error = float('inf')
best_method = None
best_initial = None

# For tracking optimization progress
all_results = []

logger.info("Training ADF parameters with enhanced optimization and parameter scaling...")

# Try global optimization methods first
try:
    # Try differential evolution (a global optimizer)
    result = differential_evolution(
        scaled_objective, 
        scaled_bounds,
        maxiter=100,
        popsize=15,
        tol=0.01
    )
    # Store result with unscaled parameters for later use
    result.x_unscaled = unscale_params(result.x)
    all_results.append((result, "DE", "global"))
    
    if result.success and result.fun < best_error:
        best_result = result
        best_error = result.fun
        best_method = "Differential Evolution"
        best_initial = "global"
except Exception as e:
    logger.debug(f"Differential Evolution failed: {str(e)}")

try:
    # Try dual annealing (another global optimizer)
    result = dual_annealing(
        scaled_objective, 
        scaled_bounds,
        maxiter=1000
    )
    # Store result with unscaled parameters for later use
    result.x_unscaled = unscale_params(result.x)
    all_results.append((result, "DA", "global"))
    
    if result.success and result.fun < best_error:
        best_result = result
        best_error = result.fun
        best_method = "Dual Annealing"
        best_initial = "global"
except Exception as e:
    logger.debug(f"Dual Annealing failed: {str(e)}")

# Define which methods support which options to avoid OptimizeWarning
def get_method_options(method):
    """Return appropriate options for each optimization method"""
    options = {}
    if method in ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr']:
        options['maxiter'] = 1000
        
    if method in ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg']:
        options['disp'] = False
        
    return options

# Then try local optimizers with multiple starting points
for idx, init_guess in enumerate(initial_guesses):
    orig_guess = initial_guesses_orig[idx]  # For reporting only
    for method in methods:
        try:
            # Get appropriate options for this method
            options = get_method_options(method)
            
            # Only pass options if they exist
            if options:
                result = minimize(
                    scaled_objective, 
                    init_guess,
                    method=method, 
                    bounds=scaled_bounds,
                    options=options
                )
            else:
                result = minimize(
                    scaled_objective, 
                    init_guess,
                    method=method, 
                    bounds=scaled_bounds
                )
            
            # Store result with unscaled parameters for later use
            result.x_unscaled = unscale_params(result.x)
            all_results.append((result, method, orig_guess))
            
            if result.success and result.fun < best_error:
                best_result = result
                best_error = result.fun
                best_method = method
                best_initial = orig_guess
        except Exception as e:
            logger.debug(f"Method '{method}' with initial {orig_guess} failed: {str(e)}")
            continue

# Sort all results by performance
all_results.sort(key=lambda x: x[0].fun)

# Log top 3 results for comparison
logger.info("Top 3 optimization results:")
for i, (result, method, init_guess) in enumerate(all_results[:3]):
    logger.info(f"Rank {i+1}: Method {method}, Initial: {init_guess}")
    logger.info(f"  Parameters (unscaled): {[round(x, 2) for x in result.x_unscaled]}")
    logger.info(f"  Score: {result.fun:.2f}")

# Analyze parameter sensitivity (using unscaled parameters)
logger.info("Parameter sensitivity analysis:")
params_values = {}
for param_idx, param_name in enumerate(["startPt", "rateTh", "width_LB", "avgRate_LB", "threshold"]):
    values = [result[0].x_unscaled[param_idx] for result in all_results[:10]]  # Top 10 results
    range_val = max(values) - min(values)
    logger.info(f"{param_name}: range = {range_val:.2f}, min = {min(values):.2f}, max = {max(values):.2f}")

# Test if bounds are limiting optimization (using unscaled parameters)
edge_params = best_result.x_unscaled
for i in range(len(edge_params)):
    if abs(edge_params[i] - bounds[i][0]) < 0.01 or abs(edge_params[i] - bounds[i][1]) < 0.01:
        logger.info(f"Parameter {i} ({['startPt', 'rateTh', 'width_LB', 'avgRate_LB', 'threshold'][i]}) is at bound {bounds[i]}")

def calculate_baseline_stats(negCurves, posCurves, pcCurves):
    """Calculate and log baseline statistics for curves"""
    def _avg_stats(curves):
        """Calculate average statistics for a set of curves"""
        if not curves:
            return 0, 0, 0
            
        stats = {'max_rate': 0, 'max_delta': 0, 'avg_rate': 0}
        count = 0
        
        for curve in curves:
            if isinstance(curve, (list, tuple)) and len(curve) >= 3:
                try:
                    # Get numpy array data
                    curve_data = curve[2]
                    
                    if hasattr(curve_data, '__len__'):  # Check if it's an array
                        # Calculate curve differences
                        data_diffs = curve_data[1:] - curve_data[:-1]
                        
                        # Calculate statistics based on labelSteps strategy
                        max_diff = max(data_diffs)  # Maximum rate of change as rateTh
                        
                        # Find continuous regions above threshold
                        rateTh = best_result.x_unscaled[1]  # Use unscaled parameter
                        width = 0
                        max_width = 0
                        max_width_start = 0  # Start position of max width region
                        curr_start = 0  # Start position of current region
                        
                        for i, diff in enumerate(data_diffs):
                            if diff > rateTh:
                                if width == 0:
                                    curr_start = i
                                width += 1

                                if width > max_width:
                                    max_width = width
                                    max_width_start = curr_start
                            else:
                                width = 0
                                
                        # Calculate average rate of change in max width region
                        if max_width > 0:
                            max_width_diffs = data_diffs[max_width_start:max_width_start+max_width]
                            avg_rate = sum(max_width_diffs)/len(max_width_diffs)
                            max_delta = sum(max_width_diffs)  # Maximum delta
                        else:
                            avg_rate = 0
                            max_delta = 0
                            
                        stats['max_rate'] += max_diff
                        stats['max_delta'] += max_delta
                        stats['avg_rate'] += avg_rate
                        count += 1
                                       
                except (ValueError, TypeError, IndexError) as e:
                    logger.debug(f"Error processing curve: {e}")
                    continue
                    
        if count == 0:
            return 0, 0, 0
            
        return (stats['max_rate']/count, stats['max_delta']/count, stats['avg_rate']/count)

    # Calculate and log statistics
    logger.info("======== Baseline Statistics ========")
    
    for curves, name in [(negCurves, "Negative"), (pcCurves, "PC")]:
        max_rate, max_delta, avg_rate = _avg_stats(curves)
        logger.info(f"{name} Curves Baseline:")
        logger.info(f"Average max_rate: {max_rate:.2f}")
        logger.info(f"Average max_delta: {max_delta:.2f}")
        logger.info(f"Average avgRate_LB: {avg_rate:.2f}")

    # Handle low, mid and high curves separately
    posCurvesName = ["Low", "Mid", "High"]
    for name, curves in zip(posCurvesName, posCurves):
        if not curves:
            continue
        max_rate, max_delta, avg_rate = _avg_stats(curves)
        logger.info(f"{name} Positive Curves Baseline:")
        logger.info(f"Average max_rate: {max_rate:.2f}")
        logger.info(f"Average max_delta: {max_delta:.2f}")
        logger.info(f"Average avgRate_LB: {avg_rate:.2f}")

    logger.info("======== End of Baseline Statistics ========")

calculate_baseline_stats(negCurves, posCurves, pcCurves)

def objective_function_ivCnt(params):
    # Constrain width_LB to be an integer
    params[2] = int(round(params[2]))
    
    # Calculate metrics using curvesMetric function
    _, _, _, _, ivCnt, _ = curvesMetric(posCurves, negCurves, pcCurves, params)
    
    # Return ivCnt as optimization objective
    return ivCnt

def objective_function_ivCnt_params(base_params, params):
    # Use base parameters plus the threshold parameter to evaluate
    full_params = [*base_params[:4], params[0]]
    ivCnt = objective_function_ivCnt(full_params)

    # Remove or reduce threshold penalty to allow lower thresholds to be explored
    threshold_weight = 0.01  # Reduced from 0.05
    penalty = threshold_weight * params[0]

    return ivCnt - penalty if ivCnt != 0 else ivCnt

# Module-level function for threshold evaluation to avoid pickling nested functions
def evaluate_threshold(args):
    """Evaluate a threshold value - callable from multiprocessing"""
    threshold, base_params = args
    full_params = [*base_params[:4], threshold]
    return threshold, objective_function_ivCnt(full_params)

# ----------------- OPTIMIZED THRESHOLD SEARCH -----------------
def optimized_threshold_search(base_params, current_objective):
    """More efficient threshold search with adaptive refinement"""
    # Generate an initial coarse grid of threshold values
    coarse_grid_size = 50  # Increased from 30 for more thorough search
    threshold_grid = np.linspace(bounds[4][0], bounds[4][1], coarse_grid_size)
    
    # Evaluate initial thresholds
    results = []
    for threshold in threshold_grid:
        result = evaluate_threshold((threshold, base_params))
        results.append(result)
    
    # Sort by ivCnt (first priority) then by threshold (prefer higher if ivCnt is equal)
    results.sort(key=lambda x: (x[1], -x[0]))
    
    # Find best threshold from coarse grid
    best_threshold, best_ivCnt = results[0]
    
    # Perform refinement phase around the best threshold
    refinement_results = []
    
    # Calculate region bounds with a buffer
    region_size = (bounds[4][1] - bounds[4][0]) / coarse_grid_size
    lower_bound = max(bounds[4][0], best_threshold - 2*region_size)
    upper_bound = min(bounds[4][1], best_threshold + 2*region_size)
    
    # Create a finer grid in this promising region
    fine_grid_size = 50
    fine_grid = np.linspace(lower_bound, upper_bound, fine_grid_size)
    
    # Evaluate the fine grid
    for fine_threshold in fine_grid:
        result = evaluate_threshold((fine_threshold, base_params))
        refinement_results.append(result)
    
    # Combine all results
    all_results = results + refinement_results
    
    # Sort by ivCnt (first priority) then by threshold (prefer higher if ivCnt is equal)
    all_results.sort(key=lambda x: (x[1], -x[0]))
    
    # Return the best result
    best_refined_threshold, best_refined_ivCnt = all_results[0]
    
    return best_refined_threshold, best_refined_ivCnt

# Try different initial thresholds within a reasonable range
threshold_bounds = [(bounds[4][0], bounds[4][1])]

# Track overall best result across all parameter sets
global_best_result_ivCnt = None
global_best_error_ivCnt = float('inf')
global_best_method_ivCnt = None
global_best_actual_ivCnt = float('inf')
global_best_threshold = bounds[4][1]
global_best_base_params = None

logger.info("Training ADF parameters for PC validity using ONLY the top 3 FP_FN parameter sets...")

# Take top 3 parameter sets from FP_FN optimization
top_param_sets = all_results[:3]
logger.info("=== Top 3 parameter sets from FP_FN optimization ===")
for param_set_idx, (result, method, init_guess) in enumerate(top_param_sets):
    base_params = result.x_unscaled
    logger.info(f"Rank #{param_set_idx+1}: {[round(x, 2) for x in base_params]} (Score: {result.fun:.2f})")

# Now optimize threshold for each of the top 3 parameter sets
for param_set_idx, (result, method, init_guess) in enumerate(top_param_sets):
    base_params = result.x_unscaled
    logger.info(f"Optimizing threshold for FP_FN parameter set #{param_set_idx+1}: {[round(x, 2) for x in base_params]}")
    
    # Use enhanced threshold search
    best_threshold, best_actual_ivCnt = optimized_threshold_search(base_params, objective_function_ivCnt)
    
    logger.info(f"Parameter set #{param_set_idx+1} - Optimized threshold - ivCnt: {best_actual_ivCnt}, threshold: {best_threshold:.2f}")
    
    # Update global best if this is better - properly compare ivCnt first, then threshold
    if (best_actual_ivCnt < global_best_actual_ivCnt or
        (best_actual_ivCnt == global_best_actual_ivCnt and best_threshold > global_best_threshold)):
        global_best_actual_ivCnt = best_actual_ivCnt
        global_best_threshold = best_threshold
        global_best_base_params = base_params.copy()  # Make a copy to avoid reference issues
        logger.info(f"New global best result - Parameter set #{param_set_idx+1} - ivCnt: {best_actual_ivCnt}, threshold: {best_threshold:.2f}")

# Output final results for all three parameter sets with their optimized thresholds
logger.info("======== Final Results for All 3 Parameter Sets ========")
for param_set_idx, (result, method, init_guess) in enumerate(top_param_sets):
    base_params = result.x_unscaled.copy()
    
    # Optimize threshold one more time to ensure consistency
    best_threshold, best_actual_ivCnt = optimized_threshold_search(base_params, objective_function_ivCnt)
    
    # Set the optimized threshold
    final_params = [*base_params[:4], best_threshold]
    
    logger.info(f"Parameter Set #{param_set_idx+1}:")
    logger.info(f"  startPt: {final_params[0]:.2f}")
    logger.info(f"  rateTh: {final_params[1]:.2f}")
    logger.info(f"  width_LB: {int(final_params[2])}")
    logger.info(f"  avgRate_LB: {final_params[3]:.2f}")
    logger.info(f"  threshold: {final_params[4]:.2f}")
    logger.info(f"  Invalid PC count: {best_actual_ivCnt}")
    logger.info(f"  FP_FN Score: {result.fun:.2f}")

# Use the global best result for final output
best_method_ivCnt = global_best_method_ivCnt
result_ivCnt = global_best_result_ivCnt

logger.info("======== Best PC Optimization Result ========")
if global_best_base_params is not None:
    final_params = [*global_best_base_params[:4], global_best_threshold]
    actual_ivCnt = objective_function_ivCnt(final_params)
    
    logger.info(f"startPt: {final_params[0]:.2f}")
    logger.info(f"rateTh: {final_params[1]:.2f}")
    logger.info(f"width_LB: {int(final_params[2])}")
    logger.info(f"avgRate_LB: {final_params[3]:.2f}")
    logger.info(f"threshold: {final_params[4]:.2f}")
    logger.info(f"Invalid PC count: {actual_ivCnt}")
logger.info("======== End of PC Optimization Results ========")

def log_false_detections(params, method_name, error_type="FP_FN"):
    """Log details of false detections for the given parameters"""
    # Get all false detections
    _, _, _, _, _, fdList = curvesMetric(posCurves, negCurves, pcCurves, params)
    
    # Filter based on error type
    if error_type == "FP_FN":
        filtered_list = [fd for fd in fdList if fd[0] != 'IV']
        error_name = "FP_FN"
    elif error_type == "PC":
        filtered_list = [fd for fd in fdList if fd[0] == 'IV']
        error_name = "PC"
    else:
        filtered_list = fdList
        error_name = "All"
    
    # Debug log to verify what's happening
    iv_curves = [fd for fd in fdList if fd[0] == 'IV']
    logger.info(f"DEBUG: PlotFalse={PlotFalse}, Error type={error_type}, IV curves count={len(iv_curves)}")
    
    logger.info(f"====== False Detection Details for {error_name} - Method: {method_name} ======")
    logger.info(f"Parameters: startPt={params[0]:.2f}, rateTh={params[1]:.2f}, "
               f"width_LB={int(params[2])}, avgRate_LB={params[3]:.2f}, threshold={params[4]:.2f}")
    
    for fd in filtered_list:
        # fd[0] is detection type, fd[1] is sample_id (human-readable)
        logger.info(f"Type: {fd[0]}, Sample ID: {fd[1]}, Channel: {fd[2]}")
    
    logger.info(f"Total false detections: {len(filtered_list)}")
    logger.info("=" * 60)

    # Generate plots if enabled
    if PlotFalse:
        # Use parameter values from the optimization results
        if error_type == "FP_FN":
            plot_types = ['FP', 'FNL', 'FNM', 'FNH']
            for plot_type in plot_types:
                # Count how many curves of this type we have
                type_curves = [fd for fd in fdList if fd[0] == plot_type]
                logger.info(f"Plotting {len(type_curves)} curves of type {plot_type}")
                
                # Use the current methodology parameters for processing but optimized params for naming
                save_path = f'falseDetection_{plot_type}.png'
                plotFalseDetectionCurves(fdList, plot_type, params, save_path=save_path, max_curves_per_plot=50)
        elif error_type == "PC":
            # Count how many IV curves we have
            iv_curves = [fd for fd in fdList if fd[0] == 'IV']
            logger.info(f"Plotting {len(iv_curves)} invalid PC curves")
            
            # Use the current methodology parameters for processing but optimized params for naming
            save_path = f'falseDetection_IV.png'
            plotFalseDetectionCurves(fdList, 'IV', params, save_path=save_path, max_curves_per_plot=50)
            
    return fdList

# Log false detections for the best FP_FN parameters
if best_result:
    fp_fn_fdList = log_false_detections(best_result.x_unscaled, best_method, "FP_FN")

# Log false detections for the best PC parameters
pc_fdList = None
if global_best_base_params is not None:  # Changed condition to check base params instead
    # Use FP_FN optimal parameters, only update threshold
    full_params = [
        global_best_base_params[0],  # startPt
        global_best_base_params[1],  # rateTh
        int(global_best_base_params[2]),  # width_LB
        global_best_base_params[3],  # avgRate_LB
        global_best_threshold
    ]
    pc_fdList = log_false_detections(full_params, "Threshold Optimization", "PC")  # Updated method name
else:
    logger.warning("No global best parameters found for PC optimization")

# Combine all false detections from both optimizations for final output
if best_result and global_best_base_params is not None:
    # Get the most optimized parameters
    best_params = best_result.x_unscaled.copy()
    # Update threshold with best PC threshold
    best_params[4] = global_best_threshold
    
    # Run once more to get all false detections with these optimized parameters
    _, _, _, _, _, all_fdList = curvesMetric(posCurves, negCurves, pcCurves, best_params)
    
    # Save false detection list to the specified output file
    save_false_detection_list(all_fdList, OUTPUT_FILE, best_params)
    
    logger.info(f"Saved combined false detection list to {OUTPUT_FILE}")

logger.info("Optimization completed successfully.")

# ----------------- PARALLELIZED OPTIMIZATION -----------------
def run_optimization_task(args):
    """Function to run a single optimization task for parallel execution"""
    method, init_guess, scaled_bounds = args
    try:
        # Get appropriate options for this method
        options = get_method_options(method)
        
        # Only pass options if they exist
        if options:
            result = minimize(
                scaled_objective, 
                init_guess,
                method=method, 
                bounds=scaled_bounds,
                options=options
            )
        else:
            result = minimize(
                scaled_objective, 
                init_guess,
                method=method, 
                bounds=scaled_bounds
            )
        
        # Store unscaled parameters
        result.x_unscaled = unscale_params(result.x)
        return (result, method, init_guess)
    except Exception as e:
        logger.debug(f"Method '{method}' with initial {init_guess} failed: {str(e)}")
        return None

logger.info("Training ADF parameters with sequential optimization...")

# Run optimization sequentially to avoid multiprocessing issues
optimization_tasks = []
for idx, init_guess in enumerate(initial_guesses):
    for method in methods:
        optimization_tasks.append((method, init_guess, scaled_bounds))

all_results = []
# Run sequentially instead of with multiprocessing
for task in optimization_tasks:
    result = run_optimization_task(task)
    if result:
        all_results.append(result)
        
        # Update best result as before
        if result[0].success and result[0].fun < best_error:
            best_result = result[0]
            best_error = result[0].fun
            best_method = result[1]
            best_initial = result[2]
            logger.info(f"New best result: method={best_method}, error={best_error:.2f}")

# ----------------- NUMBA JIT COMPILATION -----------------
@numba.jit(nopython=True)
def find_continuous_regions(data_diffs, rateTh):
    """JIT-compiled function to find continuous regions above threshold"""
    width = 0
    max_width = 0
    max_width_start = 0
    curr_start = 0
    
    for i, diff in enumerate(data_diffs):
        if diff > rateTh:
            if width == 0:
                curr_start = i
            width += 1
            
            if width > max_width:
                max_width = width
                max_width_start = curr_start
        else:
            width = 0
            
    return max_width, max_width_start

# Add this optimized function to use inside curvesMetric
@numba.jit(nopython=True)
def process_curve_jit(curve_data, rateTh, width_LB, avgRate_LB):
    """Process a single curve using JIT compilation for speed"""
    # Calculate differences
    data_diffs = np.zeros(len(curve_data) - 1, dtype=np.float64)
    for i in range(len(curve_data) - 1):
        data_diffs[i] = curve_data[i+1] - curve_data[i]
    
    # Find maximum rate change
    max_diff = np.max(data_diffs) if len(data_diffs) > 0 else 0.0
    
    # Find continuous regions
    max_width, max_width_start = find_continuous_regions(data_diffs, rateTh)
    
    # Calculate average rate
    avg_rate = 0.0
    if max_width > 0:
        sum_diffs = 0.0
        for i in range(max_width_start, max_width_start + max_width):
            sum_diffs += data_diffs[i]
        avg_rate = sum_diffs / max_width
    
    # Return results
    return max_diff, max_width, max_width_start, avg_rate
