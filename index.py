import pandas as pd

# Step 1: Read the .txt file while skipping lines starting with '%'
file_path = 'index.txt'  # Replace with the actual path to your txt file

# Read the file, skip lines starting with '%', and split by whitespace
data = pd.read_csv(file_path, sep=r'\s+', comment='%', header=None)

# Step 2: Define column names for the dataset
data.columns = ['Decimal_Year', 'Year', 'Month', 'Day', 'Day_of_Year', 'Anomaly']

# Step 3: Convert columns to appropriate data types
data['Year'] = data['Year'].astype(int)
data['Month'] = data['Month'].astype(int)
data['Day'] = data['Day'].astype(int)
data['Anomaly'] = data['Anomaly'].astype(float)

# Step 4: Save the cleaned data as a CSV file
csv_file_path = 'processed_data.csv'  # Adjust the output path as needed
data.to_csv(csv_file_path, index=False)

# Optional: Show the first few rows of the cleaned data
print(data.head())

print(f"Data has been successfully converted to {csv_file_path}")
