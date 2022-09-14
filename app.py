from json import load
import os
import pickle
import pandas as pd
import streamlit as st

from src.load_data import DataLoader


def load_pickle_model(model_path):

    model = pickle.load(open(model_path, 'rb'))
    return model

def make_predictions(df, model):

    predictions = model.predict(df)
    prediction_df = pd.DataFrame({"Item_Outlet_Sales": predictions})
    prediction_df.index = df.index
    prediction_df = prediction_df.reset_index()
    return prediction_df


if __name__ == "__main__":
    root_dir = os.path.dirname(os.getcwd())
    config_path = "params.yaml"
    data_loader = DataLoader(config_path = config_path)
    config = data_loader.read_params()
    test_path = config["split_data"]["test_path"]
    target = config["base"]["target_col"]
    model_path = config["webapp_model_path"]
    output_path = config["webapp_response_path"]
    regression_model = load_pickle_model(model_path)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title("Bigmart Sales Prediction App")
    st.header("Predicting Item Outlet Sales For A Bigmart")
    st.text("Upload the testing data")
    uploaded_file = st.file_uploader(
        "Upload testing file", type="csv")
    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)
        test_df = test_df.set_index(['item_identifier','outlet_identifier'])
        prediction_df = make_predictions(test_df, regression_model)
        prediction_df.to_csv(output_path, index = False)
        st.write("Predicted Item Outlet Sales of a bigmart are :")
        st.dataframe(prediction_df)
