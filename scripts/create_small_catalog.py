#!/usr/bin/env python3
"""
create_small_catalog.py

This script extracts the first 100 events from the catalog_with_geometry_filtered.csv file
and saves them to a new CSV file for demonstration purposes.

This smaller file can be used in examples and tutorials without needing to commit large
data files to the repository.
"""

import os
import pandas as pd
import sys

# Add parent directory to path to allow for importing from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_small_catalog(input_file, output_file, num_events=100):
    """
    Extract a small number of events from a larger catalog file.
    
    Parameters:
    -----------
    input_file : str
        Path to the input catalog file
    output_file : str
        Path where the output catalog will be saved
    num_events : int, optional
        Number of events to extract (default: 100)
    """
    print(f"Reading catalog from: {input_file}")
    try:
        # Read the full catalog
        catalog = pd.read_csv(input_file)
        print(f"Loaded catalog with {len(catalog)} events")
        
        # Extract the first num_events
        small_catalog = catalog.head(num_events)
        print(f"Extracted first {len(small_catalog)} events")
        
        # Save to new file
        small_catalog.to_csv(output_file, index=False)
        print(f"Saved small catalog to: {output_file}")
        
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    # Define input and output paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    input_file = os.path.join(data_dir, "catalog_with_geometry_filtered.csv")
    output_file = os.path.join(data_dir, "small_catalog_100_events.csv")
    
    # Create the small catalog
    success = create_small_catalog(input_file, output_file)
    
    if success:
        print("\nSmall catalog created successfully. Use this file in examples and tutorials.")
        print("Recommended way to use in notebooks:")
        print('catalog = pd.read_csv("../data/small_catalog_100_events.csv")')
    else:
        print("\nFailed to create small catalog. Check the error message above.")
