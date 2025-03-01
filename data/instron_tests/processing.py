from pathlib import Path

import pandas as pd


def process_csv(file_path):
    # Read the first few lines to identify where the data starts
    with open(file_path, 'r') as f:
        lines = []
        for i, line in enumerate(f):
            lines.append(line.strip())
            if i > 10:  # Read enough lines to find the headers
                break

    # Find where the real headers start (looking for "Time,Displacement,...")
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("Time,Displacement"):
            start_idx = i
            break

    # Read the CSV from that point
    df = pd.read_csv(file_path, skiprows=start_idx)

    # The next row contains units, get them
    units = df.iloc[0]

    # Create new column names combining the original name and unit
    new_columns = [f"{col} {units[col]}" if pd.notna(
        units[col]) else col for col in df.columns]

    # Set the new column names
    df.columns = new_columns

    # Drop the units row and convert string values to numeric
    df = df.iloc[1:].reset_index(drop=True)

    # Remove quotes from string values and convert to numeric
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.replace('"', '').astype(float)

    return df


def clean_files(dir='.', target_str='Time'):
    '''
    This will recursively find csv files and, assuming they have instron data, will strip out all the headers up to whatever the target_str says. In my case I care about the lines after and including the column headers which starts with 'Time' for me.

    You can call this file directly to do this like `python path/to/processing.py` to run this script in the current directory
    '''

    for ifp in Path(dir).rglob('*.csv'):
        if 'clean' in ifp.stem:
            continue
        found = False
        ofp = ifp.with_stem('clean_' + ifp.stem)
        with open(ifp, 'r') as infile, open(ofp, 'w') as outfile:
            for line in infile:  # Iterate over infile, not ifp
                if not found:
                    if target_str in line:
                        found = True
                        outfile.write(line)
                    continue
                outfile.write(line)  # Write to outfile, not ofp


if __name__ == "__main__":
    clean_files()
