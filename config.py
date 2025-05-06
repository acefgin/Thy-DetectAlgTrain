import os
import argparse
from datetime import datetime
from pathlib import Path
import logging

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

# Function to initialize configuration
def init_config():
    args = parser.parse_args()
    
    global DATAPATH, TESTLOGFILE, OUTPUT_FILE, PlotFalse, argBounds, SMOOTH_METHOD, ADAPTIVE_SMOOTH, CUTOFF_TIME
    
    PlotFalse = args.plot
    argBounds = args.bounds
    SMOOTH_METHOD = args.smooth_method
    ADAPTIVE_SMOOTH = args.adaptive
    CUTOFF_TIME = args.cutoff

    DATAPATH = Path(args.data)
    TESTLOGFILE = Path(args.testlog)
    OUTPUT_FILE = args.output

    # Setup logging
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
    
    return args

# Global variables (will be initialized by init_config)
DATAPATH = None
TESTLOGFILE = None
OUTPUT_FILE = None
PlotFalse = None
argBounds = None
SMOOTH_METHOD = None
ADAPTIVE_SMOOTH = None
CUTOFF_TIME = None 