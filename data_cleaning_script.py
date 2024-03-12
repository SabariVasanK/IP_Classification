import pandas as pd
import numpy as np

# File locations
data_file_location = "data/raw/"
data_file_name = "conn.log.labeled"

# Read the raw dataset
data_df = pd.read_csv(data_file_location + data_file_name, sep='\t', comment="#", header=None)

# Set column names
data_columns = pd.read_csv(data_file_location + data_file_name, sep='\t', skiprows=6, nrows=1, header=None).iloc[0][1:]
data_df.columns = data_columns

# Split the last combined column into three separate ones
tunnel_parents_column = data_df.iloc[:, -1].apply(lambda x: x.split()[0])
label_column = data_df.iloc[:, -1].apply(lambda x: x.split()[1])
data_df["label"] = label_column

# Drop the combined column
data_df.drop(columns=[data_df.columns[-2]], inplace=True)

# Add newly created columns to the dataset
data_df["tunnel_parents"] = tunnel_parents_column

# Drop irrelevant columns
data_df.drop(columns=["ts", "uid", "local_resp", "local_orig", "tunnel_parents"], inplace=True)

# Print out all column names to verify the presence of 'detailed_label'
print("Column names:", data_df.columns)

# Drop the 'detailed_label' column if it exists
if 'detailed_label' in data_df.columns:
    data_df.drop(columns="detailed_label", inplace=True)
    print("'detailed_label' column dropped.")
else:
    print("'detailed_label' column not found.")

# Replace unset values with NaN
data_df.replace({'-': np.nan, "(empty)": np.nan}, inplace=True)

# Convert data types
dtype_convert_dict = {
    "duration": float,
    "orig_bytes": float,
    "resp_bytes": float
}
data_df = data_df.astype(dtype_convert_dict)

# Specify the directory location and filename to save the cleaned dataset
cleaned_data_file_location = "data/interim/"
cleaned_data_file_name = "conn.log.labeled_cleaned.csv"

# Store cleaned dataset to a CSV file
data_df.to_csv(cleaned_data_file_location + cleaned_data_file_name, index=False)

print("Cleaning process completed. Cleaned dataset saved to:", cleaned_data_file_location + cleaned_data_file_name)
