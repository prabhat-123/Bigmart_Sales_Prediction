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




def train_and_save_model(config_path, indexes):

    data_loader = DataLoader(config_path = config_path)
    config = data_loader.read_params()
    train_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    target = config["base"]["target_col"]
    model_dir = config["model_dir"]
    train_df = data_loader.load_data(train_path)
    train_df = train_df.set_index(indexes)
    train_y = train_df[target]
    train_x = train_df.drop(target, axis=1)
    regressor = GradientBoostingRegressor(random_state=random_state)
    regressor.fit(train_x, train_y)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")
    pickle.dump(regressor, open(model_path, 'wb'))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_save_model(config_path=parsed_args.config, 
                        indexes = ['item_identifier','outlet_identifier'])
