# ADF Parameter Optimization Tool

## Project Overview

The ADF (Amplification Detection Framework) Parameter Optimization Tool is designed to automatically tune detection parameters for LAMP (Loop-mediated Isothermal Amplification) assays. The framework optimizes detection accuracy by minimizing false positive and false negative rates across a range of test samples.

This tool features:
- **Automated parameter optimization** using multiple algorithms
- **Dual threshold approach** for PC validation and target detection
- **Robust statistical reporting** with precision, recall, F1 score, and accuracy
- **Comprehensive output** with performance metrics and curve variation statistics

## Key Components

### Core Detection Parameters

The framework optimizes five critical parameters:

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `startPt` | Starting point for analysis in cycles | 75-120 |
| `rateTh` | Rate threshold for curve inflection | 0.5-9.0 |
| `width_LB` | Minimum width for positive detection | 15-35 |
| `avgRate_LB` | Average rate lower bound | 0.5-4.5 |
| `threshold` | Fluorescence difference threshold | 15-200 |

### Dual Threshold Approach

The tool employs a dual threshold strategy to separately optimize:
- **PC validation threshold** (channel 1)
- **Target detection threshold** (channels 2-5)

## Parameter Optimization Process

The optimization process follows a systematic workflow:

1. **Data Loading**: Test data is grouped into positive and negative tests
2. **Baseline Analysis**: Initial statistics are calculated to establish baselines
3. **Parameter Scaling**: Parameters are normalized for optimization stability
4. **Global Optimization**: Differential Evolution and Dual Annealing algorithms perform broad parameter space exploration
5. **Local Optimization**: Multiple local optimization methods refine parameters from promising starting points
6. **Threshold Fine-tuning**: Separate threshold optimization for PC validation and target detection
7. **Performance Validation**: Final parameters are evaluated on test data with comprehensive metrics

## Scoring Criteria

The optimization uses a weighted scoring system to balance false positives and negatives:

```python
weights = {
    'fp': 2.0,  # False positives weighted higher
    'fnH': 1.0, # False negative (High)
    'fnM': 1.0, # False negative (Medium)
    'fnL': 1.0, # False negative (Low)
    'iv': 1.0   # Invalid PC curves
}

score = weights['fp'] * fpCnt + weights['fnH'] * fnHCnt + 
        weights['fnM'] * fnMCnt + weights['fnL'] * fnLCnt
```

This weighting prioritizes reducing false positives while maintaining sensitivity.

## Optimization Methods

The tool employs multiple optimization strategies:

### Global Optimization
- **Differential Evolution**: Population-based evolutionary algorithm for global exploration
- **Dual Annealing**: Combines simulated annealing with local search

### Local Optimization
- **Nelder-Mead**: Derivative-free simplex method
- **Powell**: Direction set method for local searches
- **L-BFGS-B**: Limited-memory BFGS with bound constraints
- **TNC**: Truncated Newton method
- **SLSQP**: Sequential Least Squares Programming

Performance is enhanced through:
- Parameter scaling for stability
- Multiple starting points to avoid local minima
- Caching of evaluation results for efficiency
- Adaptive refinement for promising parameter regions

## Usage

To run the optimization tool:

```bash
python optimization.py
```

To run detection with optimized parameters:

```bash
python main.py
```

## Output and Analysis

The tool generates comprehensive outputs:

### Optimization Results
- Parameter convergence plots
- Ranked parameter sets with performance metrics
- Parameter sensitivity analysis

### Performance Metrics
- Confusion matrix (TP, TN, FP, FN)
- Precision, recall, F1 score, and accuracy
- Invalid PC curve counts

## For Developers

### Code Structure
- **`main.py`**: Primary detection functionality
- **`optimization.py`**: Parameter optimization framework
- **`config.py`**: Configuration and initialization
- **`data_loader.py`**: Test data parsing
- **`detection.py`**: Core detection algorithms
- **`visualization.py`**: Plotting and visualization
- **`export.py`**: Results export functionality

### Extending the Framework
To add new optimization algorithms:
1. Implement the algorithm interface in `optimization.py`
2. Add appropriate monitoring for convergence tracking
3. Ensure results are compatible with the existing evaluation framework

To modify scoring criteria:
1. Update the weight parameters in the objective function
2. Consider the impact on parameter sensitivity
3. Validate changes against known test datasets

### Performance Considerations
- Use caching for repetitive evaluations
- Consider parallel processing for large datasets
- Monitor memory usage for large curve collections

## References

The ADF framework implements detection strategies based on established LAMP amplification curve analysis methods, with enhancements for robustness and accuracy across diverse testing conditions.