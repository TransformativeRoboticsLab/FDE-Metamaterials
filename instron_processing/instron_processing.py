import csv
from pathlib import Path

import numpy as np
import pandas as pd

from . import Q_, ureg


def oneumerate(iter):
    # When referencing line numbers it is easier to start indexing at one so we use this as a shorthand to start indexing along with line numbers
    return enumerate(iter, start=1)


class InstronData:
    "Currently only built to handle a single specimen per CSV file"

    def __init__(self, file_path=None):
        self.file_path = file_path
        self.results_tables = []
        self.data = None
        self._data_idx = None
        self._results_table_idxs = []

        self.user_data = None

        if file_path:
            self.parse_file(file_path)

    def parse_file(self, file_path):

        if file_path is not self.file_path:
            print(f"Updating file path from {self.file_path} to {file_path}")
            self.file_path = file_path

        # find out where the data is
        self._results_table_idxs, self._data_idx = self._locate_info(
            self.file_path)
        # now we actually read the CSV file
        self.data = self._extract_raw_data(self.file_path, self._data_idx)
        # Use extend instead of append to maintain the correct nesting level
        self.results_tables.extend(
            self._extract_results_table(self.file_path, self._data_idx))

    def _locate_info(self, f):
        # find the line numbers of the data and results tables in the CSV file
        results_table_idxs = []
        data_idx = None
        with open(f, 'r') as f:
            for i, line in oneumerate(f):
                line = line.strip()
                if not line:
                    continue
                if "results table" in line.lower():
                    results_table_idxs.append(i)
                if "time,displacement" in line.lower():
                    data_idx = i
                    # we assume nothing comes after the data
                    break

        if data_idx == None:
            raise ValueError(f"Could not locate data in {f}")

        return results_table_idxs, data_idx

    def _extract_results_table(self, file_path, data_idx):
        """
        Extract results tables from the Instron CSV file with proper units using pint.

        Parameters:
        -----------
        file_path : str or Path
            Path to the CSV file
        data_idx : int
            Line number where the data section begins

        Returns:
        --------
        list : List of dictionaries containing the results tables with proper units
        """
        results_tables = []
        current_table = None
        header_row = []
        units_row = []
        units_next = False  # Flag to indicate we're expecting units in the next row

        with open(file_path, 'r') as file:
            reader = csv.reader(file)

            for line_num, row in oneumerate(reader):
                # Stop processing when we reach the data section
                if line_num >= data_idx:
                    break

                # Skip completely empty rows
                if not row:
                    continue

                # Check for the start of a new results table
                if row and "results table" in row[0].lower():
                    # Save previous table if exists
                    if current_table:
                        results_tables.append(current_table)

                    # Initialize new table and reset header
                    current_table = {}
                    header_row = []
                    units_row = []
                    units_next = False
                    continue

                # Skip this row if we haven't found a results table yet
                if current_table is None:
                    continue

                # Skip rows with only empty cells
                if all(not cell.strip() for cell in row):
                    continue

                # If we're expecting units in this row
                if units_next:
                    units_next = False
                    units_row = [cell.strip() for cell in row[1:]]
                    continue

                # If this is the first content row after table declaration, it's the header
                if not header_row:
                    header_row = [cell.strip()
                                  for cell in row[1:] if cell.strip()]

                    # Check if next row might contain units (typical pattern in Instron files)
                    # Units are often in a row starting with empty cell followed by cells with parentheses
                    units_next = True
                    continue

                # Process data rows
                if header_row:
                    # Get values from the row (skip the first cell which is usually just a row number)
                    values = [cell.strip() for cell in row[1:]]

                    # Match headers with values and store in the table
                    for i, (header, value) in enumerate(zip(header_row, values)):
                        if not value:  # Skip empty values
                            continue

                        try:
                            # Convert value to float
                            float_value = float(value)

                            # If we have units for this column, create a pint quantity
                            if i < len(units_row) and units_row[i].strip('() '):
                                unit_str = units_row[i].strip('() ')
                                # Create pint quantity with the appropriate unit
                                current_table[header] = Q_(
                                    float_value, unit_str)
                            else:
                                # No unit information, just store the float
                                current_table[header] = float_value
                        except ValueError:
                            # For non-numeric values, just store as strings
                            current_table[header] = value

        # Don't forget to add the last table if we have one
        if current_table:
            results_tables.append(current_table)

        return results_tables

    def _extract_raw_data(self, file_path, idx):
        return pd.read_csv(file_path,
                           skiprows=idx-1,
                           header=[0, 1],
                           skip_blank_lines=True,
                           ).pint.quantify(level=-1)

    def add_linear_fit(self, x, y, x_min=-np.inf, x_max=np.inf):
        x_ = self.data[x]
        y_ = self.data[y]

        mask = (x_ >= x_min) & (x_ <= x_max)
        m, b = np.polyfit(x_[mask].pint.magnitude, y_[mask].pint.magnitude, 1)
        m *= y_.pint.u / x_.pint.u
        b *= y_.pint.u

        self.data[y + ' (Line Fit)'] = m*x_ + b

        return m, b


# def shift_strain(df, strain_threshold):
#     # find index of first compressive strain that is > X% to adjust for slack in test specimen
#     # the limit is arbitrary based on pre-analyzing and plotting the raw stress/strain data

#     # Check if any values are larger than strain_threshold

#     df_ = df.copy()
#     mask = df_['Compressive strain (Displacement)'] > strain_threshold
#     if not mask.any():
#         print(
#             f"Warning: No values found above strain threshold of {strain_threshold}")
#         return df

#     idx = df_.index[mask].tolist()[0]
#     # adjust all columns to be zeroed at this index
#     for col in df_.columns:
#         df_[col] = df_[col] - df_[col][idx]

#     return df_


# def linear_fit(df: pd.DataFrame, x: str, y: str, x_min=None, x_max=None):
#     df_ = df.copy()
#     x_ = df_[x]
#     y_ = df_[y]

#     mask = (x_ >= x_min) & (x_ <= x_max)
#     m, b = np.polyfit(x_[mask].pint.magnitude, y_[mask].pint.magnitude, 1)
#     m *= y_.pint.u / x_.pint.u
#     b *= y_.pint.u

#     df_[y + ' (Line Fit)'] = m*x_ + b

#     return df_, (m.to('MPa'), b)


def load_samples(dir, filter_fn=None):

    print(f"Loading csv files from {dir}")
    files = list(filter(filter_fn, sorted(Path(dir).rglob('*.csv'))))
    print(f"{len(files)} found that match the filter")
    if len(files) == 0:
        print(f"No files found")
        return

    return [InstronData(f) for f in files]
