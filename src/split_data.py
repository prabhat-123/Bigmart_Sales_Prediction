import argparse
import warnings
from load_data import DataLoader
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

class Dataset_Splitter():
    
    def __init__(self, config_path):

        self.data_loader = DataLoader(config_path)
        config = self.data_loader.read_params()
        self.raw_data_path = config["raw_data"]["combined_dataset_csv"]
        self.train_path = config["split_data"]["train_path"]
        self.test_path = config["split_data"]["test_path"]
        self.eval_train_path = config["split_data"]["eval_train_path"]
        self.eval_test_path = config["split_data"]["eval_test_path"]
        self.split_ratio = config["split_data"]["test_size"]
        self.random_state = config["base"]["random_state"]
        self.target_col = config["base"]["target_col"]



    def split_and_save_test_data(self):

        combined_df = self.data_loader.load_data(self.raw_data_path)
        train_df = combined_df.loc[~combined_df[self.target_col].isnull()]
        test_df = combined_df.loc[combined_df[self.target_col].isnull()]
        test_df.drop(self.target_col, axis = 1, inplace = True)
        train_df.to_csv(self.train_path, sep = ",", index = False)
        test_df.to_csv(self.test_path, sep = ",", index = False)



    def split_and_save_eval_data(self):

        df = self.data_loader.load_data(self.train_path)
        train, test = train_test_split(df, test_size = self.split_ratio,
                                    random_state = self.random_state)
        train.to_csv(self.eval_train_path, sep=",", index=False)
        test.to_csv(self.eval_test_path, sep=",", index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data_splitter = Dataset_Splitter(config_path=parsed_args.config)
    data_splitter.split_and_save_test_data()
    data_splitter.split_and_save_eval_data()

