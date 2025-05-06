import numpy as np
import logging
import os
import pandas as pd
from config import TIME_CONVERSION_FACTOR, TIME_OFFSET

# Get logger
logger = logging.getLogger()

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

def trim_signal(signal, cutoff=None):
    """
    Trim a signal based on a cutoff time.
    
    Args:
        signal (array): Input signal to be trimmed
        cutoff (float, optional): Cutoff time in minutes. If None, return the original signal.
        
    Returns:
        array: Trimmed signal
    """
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