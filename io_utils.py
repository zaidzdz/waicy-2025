import csv
import json
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def write_angles_csv(path, header, row_iterable, metadata):
    """
    Streams rows to a CSV file and writes a companion metadata JSON file.

    Args:
        path (str): The base path for the output files (e.g., 'dances/output/Dance1').
        header (list): A list of strings for the CSV header.
        row_iterable (iterable): An iterable that yields rows (lists or tuples) to write.
        metadata (dict): A dictionary containing metadata to save as JSON.
    """
    csv_path = f"{path}.csv"
    meta_path = f"{path}_meta.json"

    try:
        logging.info(f"Step: Writing CSV... (Why: Saving recorded angles to {csv_path})")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(row_iterable)
        logging.info(f"Result: Successfully wrote {csv_path}")
    except (IOError, PermissionError) as e:
        logging.error(f"E104: CSV write failed for {csv_path}. Reason: {e}")
        return "E104"

    try:
        logging.info(f"Step: Writing metadata... (Why: Saving context to {meta_path})")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logging.info(f"Result: Successfully wrote {meta_path}")
    except (IOError, PermissionError) as e:
        logging.error(f"E107: Metadata write failed for {meta_path}. Reason: {e}")
        return "E107"
        
    print(f"Saved {csv_path} with columns: {header}")
    return "E100"

def read_angles_csv(path):
    """
    Reads an angles CSV file and its companion metadata file.

    Args:
        path (str): The path to the CSV file (e.g., 'dances/output/Dance1.csv').

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The loaded CSV data.
            - dict: The loaded metadata.
    """
    if not path.endswith('.csv'):
        logging.error("E101: Provided path must be a .csv file.")
        return None, None
        
    meta_path = path.replace('.csv', '_meta.json')

    try:
        logging.info(f"Step: Reading CSV... (Why: Loading canonical dance data from {path})")
        df = pd.read_csv(path)
        logging.info(f"Result: Successfully loaded {path}")
    except FileNotFoundError:
        logging.error(f"E101: CSV file not found at {path}")
        return None, None
    except Exception as e:
        logging.error(f"E109: An unknown error occurred while reading {path}. Reason: {e}")
        return None, None

    try:
        logging.info(f"Step: Reading metadata... (Why: Loading context from {meta_path})")
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        logging.info(f"Result: Successfully loaded {meta_path}")
    except FileNotFoundError:
        logging.warning(f"Metadata file not found at {meta_path}. Returning dataframe without metadata.")
        return df, {}
    except Exception as e:
        logging.error(f"E109: An unknown error occurred while reading {meta_path}. Reason: {e}")
        return df, {}

    return df, metadata
