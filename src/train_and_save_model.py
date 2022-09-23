import os
import json
import shutil
import mlflow
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from load_data import DataLoader
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



warnings.filterwarnings('ignore')

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def plot_feature_importances(model,feature_columns, path):

    fig = plt.figure(figsize=(10, 10))
    feat_importances = pd.Series(model.feature_importances_, index=feature_columns)
    feat_importances.nlargest(100).plot(kind='barh')
    fig.savefig(path)


def train_and_evaluate(config_path, indexes):

    data_loader = DataLoader(config_path = config_path)
    config = data_loader.read_params()
    eval_train_path = config["split_data"]["eval_train_path"]
    eval_test_path = config["split_data"]["eval_test_path"]
    train_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    target = config["base"]["target_col"]
    artifacts_dir = config["reports"]["artifacts_dir"]
    artifacts_path = os.path.join(artifacts_dir, "feature_importance.png")
    model_dir = config["model_dir"]

    # for evaluating the model and logging the metrics and artifacts 
    train_df_for_eval = data_loader.load_data(eval_train_path)
    train_df_for_eval = train_df_for_eval.set_index(indexes)
    test_df_for_eval = data_loader.load_data(eval_test_path)
    test_df_for_eval = test_df_for_eval.set_index(indexes)
    train_y_for_eval = train_df_for_eval[target]
    test_y_for_eval = train_df_for_eval[target]
    train_x_for_eval = train_df_for_eval.drop(target, axis=1)
    test_x_for_eval = train_df_for_eval.drop(target, axis=1)

    # for predicting the sales (consists of validation data as well as training data to create a prediciton model)
    train_df_for_testing = data_loader.load_data(train_path)
    train_df_for_testing = train_df_for_testing.set_index(indexes)
    train_y_for_testing = train_df_for_testing[target]
    train_x_for_testing = train_df_for_testing.drop(target, axis=1)



    
    # Start an MLflow run; the "with" keyword ensures we'll close the run even if this cell crashes
    with mlflow.start_run():

        #Evaluation & logging of metrics & artifacts
        regressor = GradientBoostingRegressor(random_state=random_state)
        regressor.fit(train_x_for_eval, train_y_for_eval)
        predicted_item_outlet_sales = regressor.predict(test_x_for_eval)
        (rmse, mae, r2) = eval_metrics(test_y_for_eval, predicted_item_outlet_sales)

        print("Basic GradientBoostingRegressor model :")
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

    
        # Log mlflow attributes for mlflow UI
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        plot_feature_importances(model = regressor, feature_columns = train_x_for_eval.columns,
                                        path = artifacts_path)
        
        # Log artifacts (output files)
        mlflow.log_artifact(artifacts_path)


    #####################################################
        scores_file = config["reports"]["metrics"]

        with open(scores_file, "w") as f:
            scores = {
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }
            json.dump(scores, f, indent=4)
        
        # Saving the model after training it on validation as well as training data
        regressor.fit(train_x_for_testing, train_y_for_testing)
        mlflow.sklearn.log_model(regressor, "model")
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        mlflow.sklearn.save_model(regressor, model_dir)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config, 
                      indexes = ['item_identifier','outlet_identifier'])
