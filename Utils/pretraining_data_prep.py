import os
import pandas as pd
import numpy as np
import csv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_csv_file(file_path):
    """Process a single CSV file to ensure Time and Target data points match"""
    with open(file_path, 'r') as csvfile:
        rows = list(csv.reader(csvfile))
        
        # Find Time and Target rows
        time_row = None
        target_rows = []
        for i, row in enumerate(rows):
            if row and 'Time' in row:
                time_row = (i, row)
            elif row and 'Target' in row:
                target_rows.append((i, row))
        
        if not time_row or not target_rows:
            logging.warning(f"File {file_path} missing Time or Target data")
            return
            
        # Find data start column (usually 4 columns after 'Time' or 'Target')
        time_start_idx = next((i for i, cell in enumerate(time_row[1]) if cell.strip() == 'Time'), 0) + 4
        target_start_idx = next((i for i, cell in enumerate(target_rows[0][1]) if cell.strip() == 'Target'), 0) + 4
        
        # Get time data
        time_data = [cell.strip() for cell in time_row[1][time_start_idx:] if cell.strip()]
        
        # Get lengths of all channel data
        target_lengths = []
        for _, target_row in target_rows:
            target_data = [cell.strip() for cell in target_row[target_start_idx:] if cell.strip()]
            target_lengths.append(len(target_data))
        
        # Find the longest data sequence
        max_length = max(max(target_lengths), len(time_data))
        
        # Generate new time sequence if time data points are insufficient
        if len(time_data) < max_length:
            if time_data:
                # If time data exists, extend using the same interval
                try:
                    interval = 10000
                    last_time = float(time_data[-1])
                    for i in range(max_length - len(time_data)):
                        last_time += interval
                        time_data.append(str(round(last_time, 3)))
                except (ValueError, IndexError):
                    # Use default interval if conversion fails
                    time_data = [str(i*10) for i in range(max_length)]
            else:
                # Generate new sequence if no time data exists
                time_data = [str(i*10) for i in range(max_length)]
        
        # Update time row
        new_time_row = time_row[1][:time_start_idx] + time_data + [''] * (len(rows[time_row[0]]) - len(time_data) - time_start_idx)
        rows[time_row[0]] = new_time_row
        
        # Update each Target row
        for target_row_idx, target_row in target_rows:
            target_data = [cell.strip() for cell in target_row[target_start_idx:] if cell.strip()]
            if len(target_data) < max_length:
                # Fill missing data points with the last valid value
                last_valid = target_data[-1] if target_data else '0'
                target_data.extend([last_valid] * (max_length - len(target_data)))
            new_target_row = target_row[:target_start_idx] + target_data + [''] * (len(rows[target_row_idx]) - len(target_data) - target_start_idx)
            rows[target_row_idx] = new_target_row
        
        # Write back to file
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)
        
        logging.info(f"Processed {file_path}: aligned {max_length} data points")

def rename_files_in_directory(directory_path):
    # Get all files in the directory
    files = os.listdir(directory_path)
    renamed_files = []

    logging.info(f"Found {len(files)} files to rename")

    for file in files:
        # Construct the full file path
        full_file_path = os.path.join(directory_path, file)
        
        # Check if it is a file and has .csv extension
        if os.path.isfile(full_file_path) and file.lower().endswith('.csv'):
            # Replace spaces with underscores in the file name
            new_file_name = file.replace(' ', '_')
            new_full_file_path = os.path.join(directory_path, new_file_name)
            
            # Rename the file
            os.rename(full_file_path, new_full_file_path)
            
            # Process the CSV file to align Time and Target data
            process_csv_file(new_full_file_path)
            
            # Remove .csv extension when adding to the list
            renamed_files.append(os.path.splitext(new_file_name)[0])
    
    logging.info(f"Successfully renamed and processed {len(renamed_files)} files")
    return renamed_files

def update_csv_with_filenames(csv_path, column_name, file_names):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    logging.info(f"Original CSV has {len(df)} rows")
    logging.info(f"Found {len(file_names)} files to process")
    
    # Create new DataFrame with just the header
    new_df = pd.DataFrame(columns=df.columns)
    new_df.loc[0] = df.iloc[0]  # Add header row
    
    # Create and add new rows
    for idx, file_name in enumerate(file_names):
        row_data = {col: None for col in df.columns}
        row_data[column_name] = file_name
        
        # Set Purpose and Sample Type based on file name
        if 'NTC' in file_name:
            row_data['Purpose'] = 'ADF Training-NTC'
            row_data['Sample Type'] = 'Negative'
            row_data['Sample Concentration'] = 0
        elif 'PC' in file_name:
            row_data['Purpose'] = 'ADF Training-PC'
            row_data['Sample Type'] = 'Positive'
            row_data['Sample Concentration'] = 1000
            
        # Set Layout based on Run UID
        if '_A2_' in file_name or '_A2' in file_name:
            row_data['Layout'] = 'A2,A2,A2,A2,A2'
        elif 'MS2' in file_name:
            row_data['Layout'] = 'PC,PC,PC,PC,PC'
        else:
            row_data['Layout'] = ''
        new_df.loc[idx] = row_data
    
    # Save the updated CSV file
    new_df.to_csv(csv_path, index=False)
    logging.info(f"Updated CSV now has {len(new_df)} rows")

def main():
    # Define the target CSV file
    csv_path = 'SC2A2_NB_retrain_testlog.csv'
    directory_path = 'SC2A2_NB_retrain'
    column_name = 'Run UID'

    try:
        # Check directory and file existence
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory {directory_path} does not exist")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file {csv_path} does not exist")

        # First rename all files and get the list
        renamed_files = rename_files_in_directory(directory_path)
        
        # Update the CSV file
        update_csv_with_filenames(csv_path, column_name, renamed_files)
        logging.info("Processing completed")
        
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()