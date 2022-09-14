import os
import json
import pickle
import argparse
import warnings
import numpy as np

from load_data import DataLoader
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2



def train_and_evaluate(config_path, indexes):

    data_loader = DataLoader(config_path = config_path)
    config = data_loader.read_params()
    eval_train_path = config["split_data"]["eval_train_path"]
    eval_test_path = config["split_data"]["eval_test_path"]
    random_state = config["base"]["random_state"]
    target = config["base"]["target_col"]
    train_df = data_loader.load_data(eval_train_path)
    train_df = train_df.set_index(indexes)
    test_df = data_loader.load_data(eval_test_path)
    test_df = test_df.set_index(indexes)
    train_y = train_df[target]
    test_y = test_df[target]
    train_x = train_df.drop(target, axis=1)
    test_x = test_df.drop(target, axis=1)
    regressor = GradientBoostingRegressor(random_state=random_state)
    regressor.fit(train_x, train_y)
    predicted_item_outlet_sales = regressor.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_item_outlet_sales)

    print("Basic GradientBoostingRegressor model :")
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

#####################################################
    scores_file = config["reports"]["scores"]

    with open(scores_file, "w") as f:
        scores = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        json.dump(scores, f, indent=4)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config, 
                      indexes = ['item_identifier','outlet_identifier'])
