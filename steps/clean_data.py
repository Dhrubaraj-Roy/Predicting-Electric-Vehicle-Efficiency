import logging
from typing import Tuple

import pandas as pd
from model.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy
from zenml import step
from typing_extensions import Annotated

@step
def clean_df(
    data: pd.DataFrame,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
]:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.

    Args:
        data: pd.DataFrame
    """
    try:
        preprocess_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(data, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()
        logging.info(f"Data Cleaning Complete")

        return x_train, x_test, y_train, y_test
    except Exception as e: 
        logging.error(e)
        raise e
