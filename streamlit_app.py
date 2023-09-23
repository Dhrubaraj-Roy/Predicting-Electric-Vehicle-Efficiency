import json

import numpy as np
import pandas as pd
from pkg_resources import run_main
import streamlit as st
from PIL import Image
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import main


def main():
    st.title("End to End Predicting Electric Vehicle Efficiency Pipeline with ZenML")

    # high_level_image = Image.open("_assets/high_level_overview.png")
    # st.image(high_level_image, caption="High Level Pipeline")

    # whole_pipeline_image = Image.open("_assets/training_and_deployment_pipeline_updated.png")

    # st.markdown(
    #     """ 
    # #### Problem Statement 
    #  The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc. I will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.    """
    # )
    # st.image(whole_pipeline_image, caption="Whole Pipeline")
    # st.markdown(
    #     """ 
    # Above is a figure of the whole pipeline, we first ingest the data, clean it, train the model, and evaluate the model, and if data source changes or any hyperparameter values changes, deployment will be triggered, and (re) trains the model and if the model meets minimum accuracy requirement, the model will be deployed.
    # """
    # )

    st.markdown(
        """ 
    #### Description of Features 
    These features provide a comprehensive view of an electric vehicle's capabilities and economic aspects. **Acceleration** measures the vehicle's speed increase, **Top Speed** indicates its maximum velocity, and **Range** defines how far it can travel on a single charge. **Fast Charge Speed** highlights charging efficiency. Additionally, the **Price in the UK** and **Price in Germany** signify cost considerations and market variations. These attributes collectively help assess the performance, practicality, and affordability of the electric vehicle, crucial factors for potential buyers and enthusiasts. 
   Certainly! Here's a tabular description for your electric vehicle features:

    | Feature           | Description                                                   |
    | ----------------- | ------------------------------------------------------------- |
    | Acceleration      | Measures the rate of speed increase, indicating performance.   |
    | Top Speed         | Indicates the maximum velocity the vehicle can achieve.      |
    | Range             | Defines the distance the electric vehicle can travel on a single charge, crucial for range anxiety. |
    | Fast Charge Speed | Highlights the efficiency of the fast-charging capability.    |
    | Price in the UK   | Represents the cost of the electric vehicle in the United Kingdom market. |
    | Price in Germany  | Signifies the cost of the electric vehicle in the German market. |

    This table provides a concise overview of the significance of each feature in evaluating an electric vehicle's performance and cost.
        """
    )
   
    # Acceleration=st.sidebar.slider("Acceleration")
    # TopSpeed=st.number_input("TopSpeed")
    # Range=st.sidebar.slider("Range")
    # Range = st.sidebar.slider("Range")
    # FastChargeSpeed = st.number_input('st.FastChargeSpeed')
    # PriceinUK= st.number_input("PriceinUK")
    # PriceinGermany= st.number_input("PriceinGermany")
    Acceleration = st.sidebar.slider("Acceleration", key="acceleration_slider")
    TopSpeed = st.sidebar.number_input("TopSpeed", key="top_speed_input")
    Range = st.sidebar.slider("Range", key="range_slider")
    FastChargeSpeed = st.sidebar.number_input('FastChargeSpeed', key="fast_charge_speed_input")
    PriceinUK = st.sidebar.number_input("PriceinUK", key="price_in_uk_input")
    PriceinGermany = st.sidebar.number_input("PriceinGermany", key="price_in_germany_input")


    if st.button("Predict"):
        service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        running=False,
        )
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            run_main()

        df = pd.DataFrame(
            {
               
                "Acceleration":[Acceleration],
                "TopSpeed":[TopSpeed],
                "Range":[Range],
                "FastChargeSpeed":[FastChargeSpeed],
                "PriceinUK":[PriceinUK],
                "PriceinGermany":[PriceinGermany],
            }
        )
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)
        pred = service.predict(data)
        st.success(
            "Your electric vehicle's efficiency rate(range between 0 - 5)  based on the provided product details :-{}".format(
                pred
            )
        )
    # if st.button("Results"):
    #     st.write(
    #         "We have experimented with two ensemble and tree based models and compared the performance of each model. The results are as follows:"
    #     )

    #     df = pd.DataFrame(
    #         {
    #             "Models": ["LightGBM", "Xgboost"],
    #             "MSE": [1.804, 1.781],
    #             "RMSE": [1.343, 1.335],
    #         }
    #     )
    #     st.dataframe(df)

    #     st.write(
    #         "Following figure shows how important each feature is in the model that contributes to the target variable or contributes in predicting customer satisfaction rate."
    #     )
    #     image = Image.open("_assets/feature_importance_gain.png")
    #     st.image(image, caption="Feature Importance Gain")


if __name__ == "__main__":
    main()