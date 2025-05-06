import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import logging
import os

from config import TIME_CONVERSION_FACTOR, TIME_OFFSET
from detection import labelSteps

# Get logger
logger = logging.getLogger()

def plotFalseDetectionCurves(fdList, plotType, paras, save_path=None, show_annotations=True, max_curves_per_plot=50):
    """
    Plot false detection curves for analysis
    
    Args:
        fdList (list): List of false detection data
        plotType (str): Type of false detection to plot ('FP', 'FNL', etc.)
        paras (list): Parameters used for detection
        save_path (str, optional): Custom path to save the plot. If None, uses default naming.
        show_annotations (bool): Whether to display annotations showing detection metrics
        max_curves_per_plot (int): Maximum number of curves to show in a single plot
        
    Returns:
        list: List of figure objects for further customization if needed
    """
    startPt, rate, width, avgRate, th = paras[0], paras[1], paras[2], paras[3], paras[4]
    
    # Filter curves for the requested plot type
    relevant_curves = [df for df in fdList if df[0] in plotType]
    
    if not relevant_curves:
        logger.warning(f"No curves found for plot type: {plotType}")
        # Create a simple empty plot
        fig, ax = plt.subplots(figsize=(24, 12), dpi=80)
        ax.text(15, 250, f"No {plotType} curves found", 
                horizontalalignment='center', fontsize=24)
        plt.grid(True)

        if save_path:
            fileName = save_path
        else:
            fileName = f'falseDetection_{plotType}_rateTh_{rate}_widthLb_{width}_avgRateLb_{avgRate}_th_{th}.png'
        plt.tight_layout()
        plt.savefig(fileName, dpi=120)
        logger.info(f"Saved empty plot to {fileName}")
        return [fig]
    
    # Calculate how many plots we need
    num_plots = (len(relevant_curves) + max_curves_per_plot - 1) // max_curves_per_plot
    logger.info(f"Splitting {len(relevant_curves)} curves of type {plotType} into {num_plots} plots")
    
    all_figures = []
    curves_metrics = []  # Store metrics data for CSV export
    
    # Create multiple plots if needed
    for plot_idx in range(num_plots):
        # Use sns.set_style instead of plt.style.use
        sns.set_style("whitegrid")
        
        # Set color palette based on plot type
        if plotType == 'FP':
            color_palette = sns.color_palette("Reds_d", 8)
        elif plotType in ['FNL', 'FNM', 'FNH']:
            color_palette = sns.color_palette("Blues_d", 8)
        elif plotType == 'IV':
            color_palette = sns.color_palette("Purples_d", 8)
        else:
            color_palette = sns.color_palette("husl", 8)
        
        plt.rc('axes', linewidth=2)
        font = {'weight': 'bold', 'size': 21}
        plt.rc('font', **font)
        
        # Standard figure size since we no longer need space for the table
        fig, ax = plt.subplots(figsize=(24, 12), dpi=80)
        all_figures.append(fig)
        
        # Calculate start and end indices for this chunk
        start_idx = plot_idx * max_curves_per_plot
        end_idx = min(start_idx + max_curves_per_plot, len(relevant_curves))
        chunk_curves = relevant_curves[start_idx:end_idx]
        
        # Plot title based on plot type - Simplified to not include parameters
        title_map = {
            'FP': 'False Positive Detection Curves',
            'FNL': 'False Negative (Low Conc.) Detection Curves',
            'FNM': 'False Negative (Medium Conc.) Detection Curves',
            'FNH': 'False Negative (High Conc.) Detection Curves',
            'IV': 'Invalid PC Detection Curves'
        }
        
        title = title_map.get(plotType, f'False Detection Curves for {plotType}')
        if num_plots > 1:
            title += f' (Group {plot_idx+1} of {num_plots})'
            
        # Use a cleaner title without parameters
        plt.title(title, fontsize=22, fontweight='bold')
        
        plt.xlabel('Time (mins)', fontsize=19, fontweight='bold')
        plt.ylabel('Signal (mvs)', fontsize=19, fontweight='bold')
        
        # Add vertical line at startPt
        time_at_startPt = startPt * TIME_CONVERSION_FACTOR - TIME_OFFSET
        if time_at_startPt >= 0 and time_at_startPt <= 30:
            ax.axvline(x=time_at_startPt, color='green', linestyle=':', alpha=0.7,
                    label=f'Start Point ({startPt})')
        
        # Plot curves with colors from palette
        curves_info = []
        plotted_curves = 0
        max_signal = 1500  # Default max
        
        for i, df in enumerate(chunk_curves):
            # df[0] is the error type, df[1] is the sample_id (human-readable)
            sample_id = df[1]
            ch = df[2]
            signal = df[3]
            curve_label = f"{sample_id}_{ch}"
            
            # Calculate time series (x-axis)
            xSeries = np.arange(0, len(signal), 1)
            xSeries = np.interp(xSeries, (xSeries.min(), xSeries.max()), (0, 35))
            
            # Plot with color from palette (cycling through)
            color = color_palette[i % len(color_palette)]
            line, = ax.plot(xSeries, signal, label=curve_label, color=color, linewidth=2)
            
            # Calculate metrics for this curve for annotation
            steps, diff, cp, stepWidth, avgRate_val, maxDiff = labelSteps(signal, startPt, rate, 
                                                                width, avgRate)
            
            # Store metrics for CSV export
            curves_metrics.append({
                'Type': plotType,
                'SampleID': sample_id,
                'Channel': ch,
                'Diff': diff,
                'Cp': cp,
                'StepWidth': stepWidth,
                'AvgRate': avgRate_val,
                'MaxDiff': maxDiff
            })
            
            # Store curve info for annotation
            curves_info.append({
                'line': line,
                'label': curve_label,
                'metrics': {
                    'diff': diff,
                    'cp': cp,
                    'stepWidth': stepWidth,
                    'avgRate': avgRate_val,
                    'maxDiff': maxDiff
                },
                'signal': signal,
                'xSeries': xSeries
            })
            
            plotted_curves += 1
        
        # Add parameters as a box at the right side of title
        param_text = f"Parameters: startPt={startPt}, rateTh={rate:.2f}, widthLb={width}, avgRateLb={avgRate:.2f}, Th={th:.2f}"
        param_box = dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7)
        ax.text(0.98, 0.98, param_text, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', horizontalalignment='right',
                bbox=param_box)
        
        # Add annotations if requested and curves exist
        if show_annotations and plotted_curves > 0 and curves_info:
            # No metrics table, just adjust layout for legend
            plt.subplots_adjust(bottom=0.15, top=0.92, left=0.07, right=0.93)
            
            # Add markers at critical points for each curve
            for info in curves_info:
                signal = info['signal']
                xSeries = info['xSeries']
                metrics = info['metrics']
                            
        # Adjust plot settings
        plt.grid(True)
        ax.set_xlim([0, 35])
        ax.set_ylim([0, max_signal])
        
        # Add legend at the bottom of the image (outside plotting area) for all plot types
        if plotted_curves > 0:
            # Calculate optimal number of columns based on number of curves
            if plotted_curves <= 4:
                ncols = plotted_curves
            else:
                ncols = min(6, plotted_curves // 2 + 1)  # Limit to at most 6 columns
                
            # Position legend below the plot with improved width matching
            legend = ax.legend(ncol=ncols, loc='upper center', 
                              fontsize='small', framealpha=0.8, 
                              bbox_to_anchor=(0.5, -0.05), borderaxespad=0.8)
            
            # Set legend title to plot type
            legend_title_map = {
                'FP': 'False Positives',
                'FNL': 'False Negatives (Low)',
                'FNM': 'False Negatives (Medium)',
                'FNH': 'False Negatives (High)',
                'IV': 'Invalid PC'
            }
            legend_title = legend_title_map.get(plotType, plotType)
            legend.set_title(legend_title, prop={'size': 'small', 'weight': 'bold'})
            
            # Adjust figure size to match legend width
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.25)  # Add more space at bottom for legend
        
        # Save plot with informative name
        if save_path:
            if num_plots > 1:
                # Insert group number before file extension
                base, ext = os.path.splitext(save_path)
                fileName = f"{base}_group{plot_idx+1}{ext}"
            else:
                fileName = save_path
        else:
            if num_plots > 1:
                fileName = f'falseDetection_{plotType}_{rate:.2f}_{width}_{avgRate:.2f}_{th:.2f}_group{plot_idx+1}.png'
            else:
                fileName = f'falseDetection_{plotType}_{rate:.2f}_{width}_{avgRate:.2f}_{th:.2f}.png'
        
        # Use bbox_inches='tight' to ensure all elements are included without cropping
        plt.savefig(fileName, dpi=120, bbox_inches='tight')
        logger.info(f"Saved plot to {fileName}")
    
    # Save metrics to CSV file
    if curves_metrics:
        # Generate CSV filename based on plot type
        if save_path:
            # Use save_path to derive CSV name
            base, _ = os.path.splitext(save_path)
            csv_filename = f"{base}_metrics.csv"
        else:
            csv_filename = f'falseDetection_{plotType}_{rate:.2f}_{width}_{avgRate:.2f}_{th:.2f}_metrics.csv'
        
        # Convert metrics to DataFrame and save to CSV
        metrics_df = pd.DataFrame(curves_metrics)
        metrics_df.to_csv(csv_filename, index=False)
        logger.info(f"Saved metrics to {csv_filename}")
    
    return all_figures 