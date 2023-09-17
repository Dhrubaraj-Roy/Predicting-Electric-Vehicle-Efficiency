import logging
import pandas as pd
from typing import Union
from abc import ABC, abstractmethod

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class DataPreProcessStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            # Define a list of columns with N/A and "-" values
            columns_to_clean = ['PriceinUK', 'FastChargeSpeed']

            # Replace N/A values with the median of the respective column
            for column in columns_to_clean:
                # Convert "-" values to NaN so they are also replaced by the median
                data[column] = data[column].replace('-', pd.NA)
                # Convert column to numeric (if not already)
                data[column] = pd.to_numeric(data[column], errors='coerce')
                # Calculate the median of the column (ignoring NaN values)
                median = data[column].median()
                # Fill N/A values with the median
                data[column].fillna(median, inplace=True)

            # Save the cleaned DataFrame back to a CSV file
            data.to_csv('cleaned_data.csv', sep='\t', index=False)

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

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_data(self.df)