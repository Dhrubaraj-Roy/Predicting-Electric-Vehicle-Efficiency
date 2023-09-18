import logging
import pandas as pd
from typing import Union
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class DataPreProcessStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
             
            # Define a list of columns with N/A and "-" values
            columns_to_clean = ['PriceinUK', 'FastChargeSpeed']

            # Handle non-numeric values in 'Subtitle' column
            data['Battery_kWh'] = data['Subtitle'].str.extract(r'(\d+\.\d+) kWh').astype(float)

            # Handle non-numeric values in 'Efficiency' column
            data['Efficiency_WhKm'] = data['Efficiency'].str.extract(r'(\d+) Wh/km').astype(int)

            # Handle non-numeric values in 'FastChargeSpeed' column
            data['FastChargeSpeed_kmph'] = data['FastChargeSpeed'].str.extract(r'(\d+) km/h').astype(int)

            # Drop the columns you want to remove here
            columns_to_drop = ['Name']  # Add the column names you want to drop
            data = data.drop(columns_to_drop, axis=1)

            # Handle missing values
            for column in columns_to_clean:
                # Replace "-" values with NaN
                data[column] = data[column].replace('-', pd.NA)
                # Convert column to numeric (if not already)
                data[column] = pd.to_numeric(data[column], errors='coerce')
                # Calculate the median of the column (ignoring NaN values)
                median = data[column].median()
                # Fill NaN values with the median
                data[column].fillna(median, inplace=True)

            # Handle missing values in 'PriceinUK'
            data['PriceinUK'].replace('N/A', pd.NA, inplace=True)

            # Handle currency columns
            data['PriceinGermany'] = data['PriceinGermany'].str.replace('€', '').str.replace(',', '').astype(float)
            data['PriceinUK'] = data['PriceinUK'].str.replace('£', '').str.replace(',', '').astype(float)

            # Check for any remaining missing values in the DataFrame
            missing_values = data.isnull().sum()
            print("Missing Values:")
            print(missing_values)

            # Save the cleaned DataFrame to a new CSV file
            data.to_csv('cleaned_data.csv', index=False)

            # Display the first few rows of the cleaned data
            print("Cleaned Data:")
            print(data.head())

            return data
        except Exception as e:
            # Handle the exception and raise it again
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Data dividing strategy which divides the data into train and test data.
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divides the data into train and test data.
        """
        try:
            # Assuming "Efficiency" is your target variable
            X = data.drop("Efficiency", axis=1)
            y = data["Efficiency"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e

class DataCleaning:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        """Initializes the DataCleaning class with a specific strategy."""
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(e)
            raise e