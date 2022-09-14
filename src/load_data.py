import yaml
import pandas as pd


class DataLoader():
    
    def __init__(self, config_path):
        self.config_path = config_path


    def read_params(self):
        with open(self.config_path) as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config


    def load_data(self, data_path):
        df = pd.read_csv(data_path, sep=",")
        return df

