import os
import csv
import shutil
import argparse

def read_filter_list(filter_file):
    """Read the filter list from the CSV file."""
    filter_list = []
    with open(filter_file, 'r') as f:
        reader = csv.reader(f)
        # Skip header row
        next(reader)
        for row in reader:
            if row:  # Check if row is not empty
                filter_list.append(row[0])
    return filter_list

def filter_csv_files(input_dir, output_dir, filter_list):
    """Filter CSV files based on the filter list."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all files in the input directory
    files = os.listdir(input_dir)
    
    # Track matched files and used filters
    matched_files = []
    used_filters = set()
    
    # Filter files
    for file in files:
        if file.lower().endswith('.csv'):
            for filter_item in filter_list:
                if filter_item in file:
                    # Copy the file to the output directory
                    shutil.copy2(os.path.join(input_dir, file), os.path.join(output_dir, file))
                    matched_files.append(file)
                    used_filters.add(filter_item)
                    break
    
    # Find unused filters
    unused_filters = [filter_item for filter_item in filter_list if filter_item not in used_filters]
    
    return matched_files, unused_filters

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Filter CSV files based on a list of Test IDs.')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing CSV files')
    parser.add_argument('--output', '-o', required=True, help='Output directory for filtered CSV files')
    parser.add_argument('--filter', '-f', default='filter.csv', help='Path to the filter CSV file (default: filter.csv)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Read the filter list
    filter_list = read_filter_list(args.filter)
    print(f"Loaded {len(filter_list)} Test IDs from filter file")
    
    # Filter CSV files
    matched_files, unused_filters = filter_csv_files(args.input, args.output, filter_list)
    
    # Print results
    print(f"Found {len(matched_files)} matching CSV files")
    print(f"Files have been copied to {args.output}")
    
    # Print unused filters
    if unused_filters:
        print(f"\n{len(unused_filters)} Test IDs did not match any files:")
        for filter_item in unused_filters:
            print(f"  - {filter_item}")

if __name__ == "__main__":
    main()
