import numpy as np
import pandas as pd

# Load the CSV file into a DataFrame
data = pd.read_csv('/home/dhruba/gigs_project/project_a/Predicting-Electric-Vehicle-Efficiency/data/Cheapestelectriccars-EVDatabase 2023.csv')

# Display the first few rows of the DataFrame to understand its structure
#data.drop(['Subtitle', 'Name'], axis=1, inplace=True)
print("DataFrame Shape After Drop:", data.shape)

print("Column Names:")
print(data.columns)
# Drop the 'Subtitle' and 'Name' columns
data.drop(['Subtitle', 'Name'], axis=1,  inplace=True)
data.replace('N/A', None, inplace=True)

# Extract battery capacity from 'Subtitle' column
#data['BatteryCapacity'] = data['Subtitle'].str.extract(r'([\d.]+) kWh')

# Handle non-numeric values in 'Efficiency' column
data['Efficiency_WhKm'] = data['Efficiency'].str.extract(r'(\d+) Wh/km').astype(int)
data['FastChargeSpeed_kmph'] = data['FastChargeSpeed'].str.extract(r'(\d+) km/h').fillna(0).astype(int)




# Convert data types
#data['BatteryCapacity'] = data['BatteryCapacity'].astype(float)

# Handle missing values
columns_to_clean = ['PriceinUK', 'FastChargeSpeed']
for column in columns_to_clean:
    data[column] = data[column].replace('-', pd.NA)
    data[column] = pd.to_numeric(data[column], errors='coerce')
    median = data[column].median()
    data[column].fillna(median, inplace=True)

# Handle missing values in 'PriceinUK'
data['PriceinUK'].replace('N/A', pd.NA, inplace=True)
data['PriceinUK'] = data['PriceinUK'].astype(float)

# Handle currency columns
data['PriceinGermany'] = data['PriceinGermany'].str.replace('€', '').str.replace(',', '').astype(float)

# Check for any remaining missing values in the DataFrame
missing_values = data.isnull().sum()
print("Missing Values:")
print(missing_values)
# data = data.select_dtypes(include=[np.number])

# Save the cleaned DataFrame to a new CSV file
data.to_csv('cleaned_data2.csv', index=False)

# Display the first few rows of the cleaned data
print("Cleaned Data:")
print(data.head())
