import os
# Set NumExpr to use 16 cores
os.environ["NUMEXPR_MAX_THREADS"] = "16"

"""
Main entry point for detectAlg benchmarking that imports from modular files.
This maintains backward compatibility with existing code that imports from detectAlgBenchmark.py.
"""

# First, import the init_config function and initialize configuration
from config import init_config
args = init_config()

# Now import all configuration constants to re-export
from config import (
    DEFAULT_START_PT,
    DEFAULT_RATE_TH,
    DEFAULT_WIDTH_LB,
    DEFAULT_AVG_RATE_LB,
    DEFAULT_THRESHOLD,
    LOW_CONC,
    MEDIUM_CONC,
    HIGH_CONC,
    DEFAULT_CONC,
    TIME_CONVERSION_FACTOR,
    TIME_OFFSET,
    DEFAULT_LAYOUT,
    # Global variables initialized by init_config()
    DATAPATH,
    TESTLOGFILE,
    OUTPUT_FILE,
    PlotFalse,
    argBounds,
    SMOOTH_METHOD,
    ADAPTIVE_SMOOTH,
    CUTOFF_TIME
)

# Import utility functions
from utils import (
    smooth,
    get_result_description,
    trim_signal
)

# Import detection functions
from detection import (
    labelSteps,
    curvesMetric,
    curvesMetric_manul
)

# Import data loading functions
from data_loader import (
    readRunCsv,
    testsGrouping,
    NTCMetric,
    POSMetric
)

# Import visualization functions
from visualization import (
    plotFalseDetectionCurves
)

# Import export functions
from export import (
    save_false_detection_list,
    save_dual_threshold_results
)

# Run the main function if executed directly
if __name__ == "__main__":
    from main import main
    main() 