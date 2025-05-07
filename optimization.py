import os
# Set NumExpr to use 16 cores
os.environ["NUMEXPR_MAX_THREADS"] = "16"

from scipy.optimize import minimize
# First import and run the initialization function
from config import init_config

# Initialize configuration immediately
args = init_config()

# Now import modules and variables that depend on initialized configuration
from config import DATAPATH, TESTLOGFILE, argBounds, PlotFalse, OUTPUT_FILE
# Update other imports to use our new modular structure
from detection import curvesMetric
from data_loader import testsGrouping, NTCMetric, POSMetric 
from visualization import plotFalseDetectionCurves
from export import save_dual_threshold_results

import logging
import numpy as np
from functools import lru_cache
import os
import numba
from scipy.optimize import differential_evolution, dual_annealing
import matplotlib.pyplot as plt

# Configure logging
logger = logging.getLogger()

# Import and group tests
posTests, negTests, outliers = testsGrouping(TESTLOGFILE)

# Get curves and track missing PC info
negCurves, pcNTC, neg_missing_pc = NTCMetric(negTests, DATAPATH)
posCurvesL, posCurvesM, posCurvesH, pcPOS, pos_missing_pc = POSMetric(posTests, DATAPATH)
posCurves = [posCurvesL, posCurvesM, posCurvesH]
pcCurves = pcNTC + pcPOS

logger.info(f"PC curve count: {len(pcCurves)}")

# Optimization monitoring class
class OptimizationMonitor:
    def __init__(self, method_name, initial_guess=None):
        self.method_name = method_name
        self.initial_guess = initial_guess
        self.iterations = []
        self.func_vals = []
        self.best_func_vals = []
        self.fp_counts = []
        self.fn_counts = []
        self.iteration_count = 0
        
    def callback(self, xk, convergence=None):
        """Callback function for optimization methods"""
        # Different optimization methods have different callback signatures
        self.iteration_count += 1
        
        # Calculate function value and FP/FN counts
        func_val = scaled_objective(xk)
        
        # Track raw function values
        self.iterations.append(self.iteration_count)
        self.func_vals.append(func_val)
        
        # Get detailed FP/FN counts for this parameter set
        unscaled_params = unscale_params(xk)
        # Use only the core parameters and threshold
        core_params = unscaled_params[:4]
        threshold = unscaled_params[4]
        
        # Get metrics using curvesMetric
        fpCnt, fnHCnt, fnMCnt, fnLCnt, _, _ = curvesMetric(posCurves, negCurves, [], core_params, threshold, threshold)
        total_fn = fnHCnt + fnMCnt + fnLCnt
        
        # Store FP and FN counts
        self.fp_counts.append(fpCnt)
        self.fn_counts.append(total_fn)
        
        # Track best function value seen so far
        if not self.best_func_vals:
            self.best_func_vals.append(func_val)
        else:
            self.best_func_vals.append(min(func_val, self.best_func_vals[-1]))
        
        return False  # Continue optimization

def plot_optimization_convergence(monitors, filename='optimization_convergence.png'):
    """Plot convergence data from optimization monitors"""
    plt.figure(figsize=(15, 10))
    
    # Create a color cycle to use consistently across subplots
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    # Log available monitors for debugging
    logger.info(f"Available optimization monitors: {[m.method_name for m in monitors]}")
    logger.info(f"Monitor data points: {[(m.method_name, len(m.iterations)) for m in monitors]}")
    
    # Filter monitors to focus on DE, DA, and find the best local optimizer
    global_monitors = [m for m in monitors if m.method_name in ['DE', 'DA']]
    local_monitors = [m for m in monitors if m.method_name not in ['DE', 'DA']]
    
    # Log global monitors for debugging
    logger.info(f"Global monitors: {[m.method_name for m in global_monitors]}")
    for gm in global_monitors:
        logger.info(f"{gm.method_name} data: iterations={len(gm.iterations)}, func_vals={len(gm.func_vals)}")
        if gm.method_name == 'DA' and gm.iterations:
            logger.info(f"DA sample data: {list(zip(gm.iterations[:5], gm.func_vals[:5]))}...")
    
    # Find the best local optimization monitor based on the final function value
    best_local_monitor = None
    best_local_value = float('inf')
    for monitor in local_monitors:
        if monitor.func_vals and monitor.func_vals[-1] < best_local_value:
            best_local_value = monitor.func_vals[-1]
            best_local_monitor = monitor
    
    # Selected monitors to display
    selected_monitors = global_monitors.copy()
    if best_local_monitor:
        selected_monitors.append(best_local_monitor)
    
    logger.info(f"Selected monitors for display: {[m.method_name for m in selected_monitors]}")
    
    # Plot raw function values
    plt.subplot(2, 1, 1)
    for i, monitor in enumerate(selected_monitors):
        if monitor.iterations and monitor.func_vals:
            color_idx = i % len(colors)
            color = colors[color_idx]
            
            # Create concise label
            if monitor.method_name in ['DE', 'DA']:
                label = f"{monitor.method_name}"
            else:
                # For local methods, show method name and shortened initial guess
                label = f"Best Local: {monitor.method_name}"
                
            plt.plot(monitor.iterations, monitor.func_vals, label=label, marker='o', markersize=3, color=color)
    
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('Optimization Progress - Objective Function Value')
    plt.legend(loc='upper right', fontsize='medium')
    plt.grid(True)
    
    # Plot FP and FN counts for each method
    plt.subplot(2, 1, 2)
    for i, monitor in enumerate(selected_monitors):
        if monitor.iterations and monitor.fp_counts and monitor.fn_counts:
            color_idx = i % len(colors)
            method_color = colors[color_idx]
            
            # Create concise label
            if monitor.method_name in ['DE', 'DA']:
                fp_label = f"{monitor.method_name} FP"
                fn_label = f"{monitor.method_name} FN"
            else:
                fp_label = f"Best Local FP"
                fn_label = f"Best Local FN"
                
            plt.plot(monitor.iterations, monitor.fp_counts, label=fp_label, color=method_color, linestyle='-', marker='o', markersize=3)
            plt.plot(monitor.iterations, monitor.fn_counts, label=fn_label, color=method_color, linestyle='--', marker='x', markersize=3)
    
    plt.xlabel('Iteration')
    plt.ylabel('Count')
    plt.title('False Positive (solid) and False Negative (dashed) Counts')
    plt.legend(loc='upper right', fontsize='medium')
    plt.grid(True)
    
    # Add a title for the entire figure
    plt.suptitle('Optimization Convergence', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for the suptitle
    plt.savefig(filename)
    plt.close()
    
    # Log which methods were displayed
    logger.info(f"Saved focused optimization convergence plot to {filename}")
    logger.info(f"Displayed global optimizers: {[m.method_name for m in global_monitors]}")
    if best_local_monitor:
        logger.info(f"Best local optimizer: {best_local_monitor.method_name} with score {best_local_value:.2f}")
    else:
        logger.info("No local optimizer results available")

# ------------------CURVES STATISTICS--------------------

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
                        rateTh = 1.0  
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
    # Extract core_params and threshold
    core_params = params[:4]
    threshold = params[4]
    # Use the same threshold for both PC and target
    return curvesMetric(posCurves, negCurves, pcCurves, core_params, threshold, threshold)

# Modify your objective functions to use caching
def objective_function_fp_fn(params):
    # Make params hashable for caching
    params_tuple = tuple(map(float, params))
    
    # Get metrics from cached function
    fpCnt, fnHCnt, fnMCnt, fnLCnt, ivCnt, _ = cached_curves_metric(params_tuple)
    
    # Keep your original weighting logic
    weights = {
        'fp': 2.0,
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

# Make sure initial guesses are within bounds
def ensure_within_bounds(guess, bounds):
    """Ensure that initial guess is within specified bounds"""
    bounded_guess = []
    for i, value in enumerate(guess):
        lower, upper = bounds[i]
        bounded_guess.append(max(lower, min(upper, value)))
    return bounded_guess

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

# Scale the initial guesses and ensure they're within bounds
initial_guesses = [scale_params(ensure_within_bounds(guess, bounds)) for guess in initial_guesses_orig]

# Define optimization methods to try
methods = ['Nelder-Mead', 'Powell', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']

# Track best result across all methods and initial guesses
best_result = None
best_error = float('inf')
best_method = None
best_initial = None

# Store all optimization monitors
optimization_monitors = []

# Define which methods support which options to avoid OptimizeWarning
def get_method_options(method):
    """Return appropriate options for each optimization method"""
    options = {}
    if method in ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr']:
        options['maxiter'] = 1000
        
    if method in ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg']:
        options['disp'] = False
    
    # Remove unsupported options for COBYLA
    if method == 'COBYLA':
        # COBYLA doesn't support maxiter, but it does support maxfun
        if 'maxiter' in options:
            options.pop('maxiter')
            options['maxfun'] = 1000
        
    return options

# For tracking optimization progress
all_results = []

logger.info("Training ADF parameters with enhanced optimization and parameter scaling...")

# Try global optimization methods first
try:
    # Try differential evolution (a global optimizer)
    de_monitor = OptimizationMonitor("DE", "global")
    optimization_monitors.append(de_monitor)
    
    result = differential_evolution(
        scaled_objective, 
        scaled_bounds,
        maxiter=100,
        popsize=15,
        tol=0.01,
        callback=de_monitor.callback
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
    da_monitor = OptimizationMonitor("DA", "global")
    optimization_monitors.append(da_monitor)
    
    # Define a custom callback function to ensure it works correctly with DA
    def da_callback(x, f, context):
        da_monitor.callback(x)
        return False  # Continue optimization
    
    result = dual_annealing(
        scaled_objective, 
        scaled_bounds,
        maxiter=1000,
        callback=da_callback
    )
    # Store result with unscaled parameters for later use
    result.x_unscaled = unscale_params(result.x)
    all_results.append((result, "DA", "global"))
    
    if result.success and result.fun < best_error:
        best_result = result
        best_error = result.fun
        best_method = "Dual Annealing"
        best_initial = "global"
        
    # Check if DA monitor collected data
    logger.info(f"DA monitor data: iterations={len(da_monitor.iterations)}, values={len(da_monitor.func_vals)}")
except Exception as e:
    logger.debug(f"Dual Annealing failed: {str(e)}")

# Then try local optimizers with multiple starting points
for idx, init_guess in enumerate(initial_guesses):
    orig_guess = initial_guesses_orig[idx]  # For reporting only
    for method in methods:
        try:
            # Get appropriate options for this method
            options = get_method_options(method)
            
            # Create a monitor for this optimization run
            method_monitor = OptimizationMonitor(method, orig_guess)
            optimization_monitors.append(method_monitor)
            
            # Only pass options if they exist
            if options:
                result = minimize(
                    scaled_objective, 
                    init_guess,
                    method=method, 
                    bounds=scaled_bounds,
                    callback=method_monitor.callback,
                    options=options
                )
            else:
                result = minimize(
                    scaled_objective, 
                    init_guess,
                    method=method, 
                    bounds=scaled_bounds,
                    callback=method_monitor.callback
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

# Generate convergence plot after all optimization methods have completed
plot_optimization_convergence(optimization_monitors, 'optimization_convergence.png')

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
if all_results and len(all_results) > 0:
    edge_params = all_results[0][0].x_unscaled
    for i in range(len(edge_params)):
        if abs(edge_params[i] - bounds[i][0]) < 0.01 or abs(edge_params[i] - bounds[i][1]) < 0.01:
            logger.info(f"Parameter {i} ({['startPt', 'rateTh', 'width_LB', 'avgRate_LB', 'threshold'][i]}) is at bound {bounds[i]}")



def log_false_detections(params, method_name, error_type="FP_FN"):
    """Log details of false detections for the given parameters"""
    # Extract core_params and threshold
    core_params = params[:4]
    threshold = params[4]
    
    # Get all false detections - use the same threshold for both PC and target
    _, _, _, _, _, fdList = curvesMetric(posCurves, negCurves, pcCurves, core_params, threshold, threshold)
    
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

def objective_function_ivCnt(params):
    # Constrain width_LB to be an integer
    params[2] = int(round(params[2]))
    
    # Extract core_params and threshold
    core_params = params[:4]
    threshold = params[4]
    
    # Calculate metrics using curvesMetric function
    # Use threshold for both PC and target threshold, but only PC curves are passed
    _, _, _, _, ivCnt, _ = curvesMetric(posCurves, negCurves, pcCurves, core_params, threshold, threshold)
    
    # Return ivCnt as optimization objective
    return ivCnt

# Consolidated evaluation function for both PC and FP/FN cases
def evaluate_threshold_generic(args, optimization_type="PC"):
    """
    Generic threshold evaluation function for both PC and FP/FN optimization
    
    Args:
        args: Tuple containing (threshold, base_params)
        optimization_type: Type of optimization - "PC" for PC validation or "FP_FN" for false detection
        
    Returns:
        For PC: (threshold, ivCnt)
        For FP_FN: (threshold, score, fpCnt, fnCnt, fp_weight, fn_weight)
    """
    threshold, base_params = args
    core_params = base_params[:4]
    full_params = [*core_params, threshold]
    
    if optimization_type == "PC":
        # For PC validation, only count invalid PC curves
        _, _, _, _, ivCnt, _ = curvesMetric([[], [], []], [], pcCurves, core_params, threshold, threshold)
        return threshold, ivCnt
    else:  # FP_FN
        # For FP/FN, count false positives and false negatives with weighting
        fpCnt, fnHCnt, fnMCnt, fnLCnt, _, _ = curvesMetric(posCurves, negCurves, [], core_params, threshold, threshold)
        
        # Higher weight for FP to prioritize its reduction
        fp_weight = 2.0
        fn_weight = 1.0
        
        score = (fp_weight * fpCnt + 
                fn_weight * (fnHCnt + fnMCnt + fnLCnt))
        
        return threshold, score, fpCnt, (fnHCnt + fnMCnt + fnLCnt), fp_weight, fn_weight

# Generalized threshold optimization function
def optimize_threshold(base_params, optimization_type="FP_FN", threshold_bounds=None):
    """
    Generalized threshold optimization with adaptive refinement
    
    Args:
        base_params: Core parameters to use (startPt, rateTh, width_LB, avgRate_LB)
        optimization_type: Type of optimization ("FP_FN" or "PC")
        threshold_bounds: Optional bounds for threshold search (default: global bounds[4])
        
    Returns:
        Tuple containing the best threshold and additional metrics
    """
    logger.info(f"Optimizing threshold for {optimization_type} using provided core parameters...")
    
    # Use provided bounds or global bounds
    if threshold_bounds is None:
        threshold_bounds = bounds[4]
    
    # Generate an initial coarse grid of threshold values
    coarse_grid_size = 50
    threshold_grid = np.linspace(threshold_bounds[0], threshold_bounds[1], coarse_grid_size)
    
    # Evaluate initial thresholds using the consolidated evaluation function
    results = []
    for threshold in threshold_grid:
        result = evaluate_threshold_generic((threshold, base_params), optimization_type)
        results.append(result)
    
    # Sort by appropriate metric (different for FP_FN vs PC)
    # For both cases: first sort by metric (lower is better), then by threshold (higher is better)
    results.sort(key=lambda x: (x[1], -x[0]))
    
    # Find best threshold from coarse grid
    best_result = results[0]
    best_threshold = best_result[0]
    
    # Log best from coarse grid
    if optimization_type == "FP_FN":
        _, best_score, best_fp, best_fn, _, _ = best_result
        logger.info(f"Coarse search best: threshold={best_threshold:.2f}, FP={best_fp}, FN={best_fn}, score={best_score:.2f}")
    else:
        _, best_ivCnt = best_result
        logger.info(f"Coarse search best: threshold={best_threshold:.2f}, ivCnt={best_ivCnt}")
    
    # Perform refinement phase around the best threshold with wider search range
    refinement_results = []
    
    # Calculate region bounds with an expanded buffer
    region_size = (threshold_bounds[1] - threshold_bounds[0]) / coarse_grid_size
    lower_bound = max(threshold_bounds[0], best_threshold - 5*region_size)
    
    # For FP/FN, focus more on exploring higher thresholds
    if optimization_type == "FP_FN":
        upper_bound = min(threshold_bounds[1], best_threshold + 15*region_size)
    else:
        upper_bound = min(threshold_bounds[1], best_threshold + 5*region_size)
    
    logger.info(f"{optimization_type} fine search range: {lower_bound:.2f} to {upper_bound:.2f}")
    
    # Create a finer grid in this promising region with more points for better resolution
    fine_grid_size = 80
    fine_grid = np.linspace(lower_bound, upper_bound, fine_grid_size)
    
    # Evaluate the fine grid using the consolidated evaluation function
    for fine_threshold in fine_grid:
        result = evaluate_threshold_generic((fine_threshold, base_params), optimization_type)
        refinement_results.append(result)
    
    # Combine all results
    all_results = results + refinement_results
    
    # Sort by appropriate metric again
    all_results.sort(key=lambda x: (x[1], -x[0]))
    
    # For FP/FN, log the top 5 results
    if optimization_type == "FP_FN":
        top_results = all_results[:5]
        logger.info(f"Top 5 threshold values for {optimization_type}:")
        for i, (threshold, score, fp, fn, _, _) in enumerate(top_results):
            logger.info(f"Rank {i+1}: Threshold={threshold:.2f}, FP={fp}, FN={fn}, Total={fp+fn}, Score={score:.2f}")
    
    # Return the best result
    return all_results[0]

# Try different initial thresholds within a reasonable range
threshold_bounds = [(bounds[4][0], bounds[4][1])]

# Track overall best result across all parameter sets
global_best_result_ivCnt = None
global_best_error_ivCnt = float('inf')
global_best_method_ivCnt = None
global_best_actual_ivCnt = float('inf')
global_best_threshold = bounds[4][1]
global_best_base_params = None

logger.info("Training ADF parameters for FP_FN optimization using ONLY the top ranked parameter set...")

# Take the #1 parameter set from FP_FN optimization
top_param_set = all_results[0]
best_fp_fn_result, best_fp_fn_method, best_fp_fn_init_guess = top_param_set
base_params = best_fp_fn_result.x_unscaled
logger.info(f"Rank #1 parameters: {[round(x, 2) for x in base_params]} (Score: {best_fp_fn_result.fun:.2f})")

# Run the FP/FN threshold optimization
logger.info(f"Fine-tuning threshold for FP_FN parameter set: {[round(x, 2) for x in base_params]}")
best_result = optimize_threshold(base_params, "FP_FN")
best_threshold, best_score, best_fp, best_fn, fp_weight, fn_weight = best_result

# Create final parameter set with optimized threshold
optimized_fp_fn_params = [*base_params[:4], best_threshold]

# Run one more evaluation to show detailed breakdown
fpCnt, fnHCnt, fnMCnt, fnLCnt, ivCnt, _ = curvesMetric(posCurves, negCurves, pcCurves, optimized_fp_fn_params[:4], optimized_fp_fn_params[4], optimized_fp_fn_params[4])
logger.info("===== Final FP/FN Optimized Results =====")
logger.info(f"Parameters: startPt={optimized_fp_fn_params[0]:.2f}, rateTh={optimized_fp_fn_params[1]:.2f}, "
          f"width_LB={int(optimized_fp_fn_params[2])}, avgRate_LB={optimized_fp_fn_params[3]:.2f}, threshold={optimized_fp_fn_params[4]:.2f}")
logger.info(f"FP count: {fpCnt}")
logger.info(f"FN High count: {fnHCnt}")
logger.info(f"FN Medium count: {fnMCnt}")
logger.info(f"FN Low count: {fnLCnt}")
logger.info(f"Total FP+FN: {fpCnt+fnHCnt+fnMCnt+fnLCnt}")
logger.info(f"Invalid PC count: {ivCnt}")
logger.info("========================================")

# Update the best result (at index 0) with the optimized threshold if it's better
logger.info("Comparing original and fine-tuned threshold results...")
original_fp_fn_score = best_fp_fn_result.fun
optimized_fp_fn_score = fp_weight * fpCnt + fn_weight * (fnHCnt + fnMCnt + fnLCnt)  # Calculate using same weights
if optimized_fp_fn_score < original_fp_fn_score:
    logger.info(f"Fine-tuned threshold improved score from {original_fp_fn_score:.2f} to {optimized_fp_fn_score:.2f}")
    # Create a copy of the original result and update with new threshold 
    updated_result = best_fp_fn_result
    updated_result.x_unscaled = optimized_fp_fn_params
    updated_result.fun = optimized_fp_fn_score
    # Replace in all_results
    all_results[0] = (updated_result, best_fp_fn_method, best_fp_fn_init_guess)
    logger.info(f"Updated rank #1 parameters to: {[round(x, 2) for x in optimized_fp_fn_params]}")
else:
    logger.info(f"Original threshold is better. Original score: {original_fp_fn_score:.2f}, Fine-tuned score: {optimized_fp_fn_score:.2f}")

# Continue with PC validation threshold optimization
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
    
    # Use enhanced threshold search with our consolidated optimization function
    best_result = optimize_threshold(base_params, "PC")
    best_threshold, best_actual_ivCnt = best_result
    
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
    best_result = optimize_threshold(base_params, "PC")
    best_threshold, best_actual_ivCnt = best_result
    
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

# Log false detections for the best FP_FN parameters
if all_results and len(all_results) > 0:
    best_fp_fn_params = all_results[0][0].x_unscaled
    best_method_name = all_results[0][1]
    fp_fn_fdList = log_false_detections(best_fp_fn_params, best_method_name, "FP_FN")

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
if all_results and len(all_results) > 0 and global_best_base_params is not None:
    # Get best FP/FN parameters from all_results
    best_fp_fn_params = all_results[0][0].x_unscaled
    
    # Use core parameters from FP_FN optimization instead of PC optimization
    core_params = best_fp_fn_params.copy()[:4]  # Changed from global_best_base_params
    
    # Define separate thresholds for PC and target detection
    threshold_PC = global_best_threshold  # For PC validation (channel 1)
    threshold_T = best_fp_fn_params[4]  # For target detection (channels 2-5)

    # Log the dual-threshold approach
    logger.info("Using dual threshold approach:")
    logger.info(f"  Core parameters: startPt={core_params[0]:.2f}, rateTh={core_params[1]:.2f}, " 
               f"width_LB={int(core_params[2])}, avgRate_LB={core_params[3]:.2f}")
    logger.info(f"  PC threshold (ch1): {threshold_PC:.2f}")
    logger.info(f"  Target threshold (ch2-5): {threshold_T:.2f}")
    
    # Run metrics for both parameter sets with the new interface
    # For PC validation - use PC threshold for both (PC only processing)
    _, _, _, _, ivCnt_PC, pc_fdList = curvesMetric([[], [], []], [], pcCurves, core_params, threshold_PC, threshold_PC)
    # For target detection - use target threshold for both (target only processing)
    fpCnt, fnHCnt, fnMCnt, fnLCnt, _, target_fdList = curvesMetric(posCurves, negCurves, [], core_params, threshold_T, threshold_T)
    
    # Save results with the new helper function
    all_fdList = save_dual_threshold_results(pc_fdList, target_fdList, core_params, threshold_PC, threshold_T, OUTPUT_FILE)

logger.info("Optimization completed successfully.")
