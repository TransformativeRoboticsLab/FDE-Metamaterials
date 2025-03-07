from pathlib import Path

import numpy as np
import pandas as pd
import pint
import pint_pandas as pp

from . import Q_, ureg


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


def shift_strain(df, strain_threshold):
    # find index of first compressive strain that is > X% to adjust for slack in test specimen
    # the limit is arbitrary based on pre-analyzing and plotting the raw stress/strain data

    # Check if any values are larger than strain_threshold

    df_ = df.copy()
    mask = df_['Compressive strain (Displacement)'] > strain_threshold
    if not mask.any():
        print(
            f"Warning: No values found above strain threshold of {strain_threshold}")
        return df

    idx = df_.index[mask].tolist()[0]
    # adjust all columns to be zeroed at this index
    for col in df_.columns:
        df_[col] = df_[col] - df_[col][idx]

    return df_


def linear_fit(df: pd.DataFrame, x: str, y: str, x_min=None, x_max=None):
    df_ = df.copy()
    x_ = df_[x]
    y_ = df_[y]

    mask = (x_ >= x_min) & (x_ <= x_max)
    m, b = np.polyfit(x_[mask].pint.magnitude, y_[mask].pint.magnitude, 1)
    m *= y_.pint.u / x_.pint.u
    b *= y_.pint.u

    df_[y + ' (Line Fit)'] = m*x_ + b

    return df_, (m.to('MPa'), b)


def load_samples(dir, strain_threshold=-np.inf, fit_strain_limits=(0.0, 0.1), filter_fn=None):
    # strain_threshold sets the value for strain that we want to shift to the x=0 line if there was slack in the specimen. Defaults to -inf so that no shift occurs.
    # fit_strain_limits bounds the linear fit of the stress-strain curve
    dfs = []
    metadata = []
    for file in filter(filter_fn, sorted(Path(dir).rglob('*.csv'))):
        header_start = find_header_line(
            file, header_keyword='time,displacement')

        df = pd.read_csv(
            file, header=[header_start, header_start+1]).pint.quantify(level=-1)
        df = shift_strain(df, strain_threshold)

        df, (m, b) = linear_fit(df,
                                x='Compressive strain (Displacement)',
                                y='Compressive stress',
                                x_min=fit_strain_limits[0],
                                x_max=fit_strain_limits[1])
        dfs.append(df)

        metadata.append({
            'filename': file.stem,
            'slope': m,
            'intercept': b
        })

    return dfs, metadata


def find_header_line(file_path, header_keyword='time,displacement'):
    """
    Find the line number where data headers begin in a CSV file.
    Only non-empty lines are counted in the line numbering.

    Parameters:
    -----------
    file_path : str or Path
        Path to the CSV file
    header_keyword : str, default='time,displacement'
        Case-insensitive string to identify the header line

    Returns:
    --------
    int : Line number (0-based) of non-empty lines where the header is found
    """
    with open(file_path, 'r') as f:
        non_empty_line_num = 0
        for line in f:
            # Skip empty lines without counting them
            if not line.strip():
                continue

            # For debugging
            # print(f"Non-empty line {non_empty_line_num}: {line.strip()}")

            if header_keyword.lower() in line.lower():
                return non_empty_line_num

            non_empty_line_num += 1

    # If we get here, we didn't find the header
    raise ValueError(
        f"Could not find header containing '{header_keyword}' in {file_path}")


if __name__ == "__main__":
    clean_files()
