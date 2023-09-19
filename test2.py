import pandas as pd
import numpy as np 
# Read the CSV file
data = pd.read_csv('/home/dhruba/gigs_project/project_a/Predicting-Electric-Vehicle-Efficiency/data/Cheapestelectriccars-EVDatabase 2023.csv')

data.drop(['Name', 'Subtitle', 'Drive'], axis=1, inplace=True)

# Clean the 'PriceinGermany' and 'PriceinUK' columns by removing '€' and '£' signs
data['PriceinGermany'] = data['PriceinGermany'].str.replace('[€,]', '', regex=True).astype(float)
data['PriceinUK'] = data['PriceinUK'].str.replace('[£,]', '', regex=True).astype(float)

# Remove units from 'Range', 'Efficiency', and 'FastChargeSpeed' columns
data['Range'] = data['Range'].str.replace(' km', '').astype(float)
data['Efficiency'] = data['Efficiency'].str.replace(' Wh/km', '').astype(float)

# Handle 'FastChargeSpeed' column
data['FastChargeSpeed'] = data['FastChargeSpeed'].str.replace(' km/h', '')
# Replace non-numeric values ('-') with NaN
data['FastChargeSpeed'] = data['FastChargeSpeed'].replace('-', np.nan)
# Convert the column to float
data['FastChargeSpeed'] = data['FastChargeSpeed'].astype(float)

# Remove 'sec' from the 'Acceleration' column
data['Acceleration'] = data['Acceleration'].str.replace(' sec', '').astype(float)

# Remove ' km/h' from the 'TopSpeed' column
data['TopSpeed'] = data['TopSpeed'].str.replace(' km/h', '').astype(float)

# Fill N/A values with the mean of each column
data.fillna(data.mean(), inplace=True)

# Print the cleaned DataFrame
print(data)