import pandas as pd
import csv
from collections import defaultdict
import argparse
import os

def analyze_detection_results(csv_file, output_csv=None, detailed=False):
    """
    Analyze falseDetectionList.csv according to these rules:
    1) If positive control (PC) is invalid, count as invalid test
    2) If PC is valid, any false positive curve per sampleID makes test a false positive
    3) If PC is valid, all four channels (ch2, ch3, ch4, and ch5) must be false negatives
       to count as a false negative test
    
    Args:
        csv_file: Path to the input CSV file
        output_csv: Optional path to save results as CSV
        detailed: Whether to include detailed channel statistics
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Initialize counters
    invalid_tests = []
    false_positive_tests = set()
    
    # Store details per sample
    sample_details = defaultdict(lambda: {'channels': defaultdict(dict), 'type': None})
    
    # Track which channels are false negative for each sample
    false_negative_channels = defaultdict(set)
    
    # Track all channels tested per sample for false negative determination
    all_channels_per_sample = defaultdict(set)
    
    # All required channels for a false negative classification
    required_channels = {'ch2', 'ch3', 'ch4', 'ch5'}
    
    # First pass: find invalid PC and false positives
    for _, row in df.iterrows():
        sample_id = row['SampleID']
        test_type = row['Type']
        channel = row['Channel']
        
        # Store details for this sample-channel combination
        sample_details[sample_id]['channels'][channel] = {
            'type': test_type,
            'diff': row['Diff'],
            'cp': row['Cp'],
            'amplified?': row['Amplified?'],
            'result': row['Result']
        }
        
        # Rule 1: Invalid PC detection
        if test_type == 'IV' and 'Invalid PC' in row['Result']:
            invalid_tests.append(sample_id)
            sample_details[sample_id]['type'] = 'Invalid PC'
        
        # Rule 2: False positive detection
        elif test_type == 'FP' and row['Amplified?'] == True:
            false_positive_tests.add(sample_id)
            sample_details[sample_id]['type'] = 'False Positive'
        
        # Track false negative data for Rule 3
        elif test_type == 'FNL':
            # Only consider channels 2-5
            if channel in required_channels:
                # Record this channel for this sample
                all_channels_per_sample[sample_id].add(channel)
                
                # If it's a false negative, record it
                if 'False Negative' in row['Result']:
                    false_negative_channels[sample_id].add(channel)
    
    # Process false negatives
    # A sample is a true false negative only if ALL four channels (ch2-ch5) are false negatives
    false_negative_tests = []
    for sample_id, channels in all_channels_per_sample.items():
        # Skip samples that are already counted as invalid or false positive
        if sample_id in invalid_tests or sample_id in false_positive_tests:
            continue
            
        # Check if all four required channels are present and are false negatives
        if false_negative_channels[sample_id] == required_channels:
            false_negative_tests.append(sample_id)
            sample_details[sample_id]['type'] = 'False Negative'
    
    # Generate the report
    print(f"Analysis Results for {csv_file}")
    print("-" * 40)
    
    print(f"\nInvalid PC Tests: {len(invalid_tests)}")
    for sample in invalid_tests:
        print(f"  - {sample}")
        
        if detailed:
            for channel, details in sample_details[sample]['channels'].items():
                print(f"    * {channel}: {details['result']}")
    
    print(f"\nFalse Positive Tests: {len(false_positive_tests)}")
    for sample in sorted(false_positive_tests):
        print(f"  - {sample}")
        
        if detailed:
            fp_channels = [c for c, details in sample_details[sample]['channels'].items() 
                         if details['type'] == 'FP' and details['qualified']]
            print(f"    * Positive in channels: {', '.join(fp_channels)}")
    
    print(f"\nFalse Negative Tests: {len(false_negative_tests)}")
    for sample in sorted(false_negative_tests):
        print(f"  - {sample}")
        
        if detailed:
            print(f"    * False negative in channels: {', '.join(sorted(false_negative_channels[sample]))}")
    
    # Summary counts
    print("\nSummary:")
    print(f"  Invalid PC Tests: {len(invalid_tests)}")
    print(f"  False Positive Tests: {len(false_positive_tests)}")
    print(f"  False Negative Tests: {len(false_negative_tests)}")
    print(f"  Total: {len(invalid_tests) + len(false_positive_tests) + len(false_negative_tests)}")
    
    # Channel statistics
    if detailed:
        print("\nChannel Statistics:")
        channel_counts = {'ch2': 0, 'ch3': 0, 'ch4': 0, 'ch5': 0}
        
        # Count false positives by channel
        fp_channel_counts = defaultdict(int)
        for sample in false_positive_tests:
            for channel, details in sample_details[sample]['channels'].items():
                if details['type'] == 'FP' and details['qualified']:
                    fp_channel_counts[channel] += 1
        
        print("  False Positives by Channel:")
        for channel in sorted(fp_channel_counts.keys()):
            print(f"    * {channel}: {fp_channel_counts[channel]}")
    
    # Write results to CSV if output path is provided
    if output_csv:
        results_data = []
        
        # Add invalid tests
        for sample_id in invalid_tests:
            entry = {
                'SampleID': sample_id,
                'Result': 'Invalid PC'
            }
            # Add channel details if requested
            if detailed:
                for ch in ['ch1', 'ch2', 'ch3', 'ch4', 'ch5']:
                    if ch in sample_details[sample_id]['channels']:
                        entry[f'{ch}_Result'] = sample_details[sample_id]['channels'][ch]['result']
            results_data.append(entry)
        
        # Add false positive tests
        for sample_id in sorted(false_positive_tests):
            entry = {
                'SampleID': sample_id,
                'Result': 'False Positive'
            }
            # Add channel details if requested
            if detailed:
                for ch in ['ch1', 'ch2', 'ch3', 'ch4', 'ch5']:
                    if ch in sample_details[sample_id]['channels']:
                        entry[f'{ch}_Result'] = sample_details[sample_id]['channels'][ch]['result']
                        if sample_details[sample_id]['channels'][ch]['type'] == 'FP':
                            entry[f'{ch}_Diff'] = sample_details[sample_id]['channels'][ch]['diff']
                            entry[f'{ch}_Cp'] = sample_details[sample_id]['channels'][ch]['cp']
            results_data.append(entry)
        
        # Add false negative tests
        for sample_id in sorted(false_negative_tests):
            entry = {
                'SampleID': sample_id,
                'Result': 'False Negative'
            }
            # Add channel details if requested
            if detailed:
                for ch in ['ch1', 'ch2', 'ch3', 'ch4', 'ch5']:
                    if ch in sample_details[sample_id]['channels']:
                        entry[f'{ch}_Result'] = sample_details[sample_id]['channels'][ch]['result']
            results_data.append(entry)
        
        # Write to CSV
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")
    
    return {
        'invalid': invalid_tests,
        'false_positive': list(false_positive_tests),
        'false_negative': false_negative_tests,
        'details': sample_details if detailed else None
    }

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Analyze detection results from CSV file')
    parser.add_argument('input_file', help='Path to the input CSV file (falseDetectionList.csv)')
    parser.add_argument('-o', '--output', help='Path to save results CSV file (optional)')
    parser.add_argument('-d', '--detailed', action='store_true', 
                       help='Include detailed channel information in the output')
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: File '{args.input_file}' not found!")
        return
    
    # Run the analysis
    analyze_detection_results(args.input_file, args.output, args.detailed)

if __name__ == "__main__":
    main() 