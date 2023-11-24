import warnings

warnings.filterwarnings("ignore")

from exploring_the_data import TrainDataExploring, train_image_dir
import os
import pandas as pd

from sklearn.model_selection import train_test_split


import gc

gc.enable()


data_exploring = TrainDataExploring()


class TrainPreparing:
    def __init__(self):
        self.unique_img_ids = self.make_unique_img_ids()
        self.train_ids, self.valid_ids = self.train_test_split()
        self.train_df = self.train_df_method()
        self.valid_df = self.valid_df_method()

    @staticmethod
    def made_new_data_frame():
        data_exploring.mask["ships"] = data_exploring.mask["EncodedPixels"].map(
            lambda c_row: 1 if isinstance(c_row, str) else 0
        )

        return data_exploring.mask

    def make_unique_img_ids(self):
        unique_img_ids = (
            self.made_new_data_frame()
            .groupby("ImageId")
            .agg({"ships": "sum"})
            .reset_index()
        )
        unique_img_ids["has_ship"] = unique_img_ids["ships"].map(
            lambda x: 1.0 if x > 0 else 0.0
        )
        unique_img_ids = self.calculate_a_weight(unique_img_ids)
        unique_img_ids = self.drop_low_weight_image(unique_img_ids)
        self.drop_ship_in_old_mask()
        return unique_img_ids

    @staticmethod
    def drop_low_weight_image(unique_img_ids):
        unique_img_ids = unique_img_ids[unique_img_ids.file_size_kb > 35]
        unique_img_ids.index += 1
        return unique_img_ids

    @staticmethod
    def calculate_a_weight(unique_img_ids):
        unique_img_ids["file_size_kb"] = unique_img_ids["ImageId"].map(
            lambda c_img_id: os.stat(os.path.join(train_image_dir, c_img_id)).st_size
            / 1024
        )
        return unique_img_ids

    @staticmethod
    def drop_ship_in_old_mask():
        data_exploring.mask.drop(["ships"], axis=1, inplace=True)
        data_exploring.mask.index += 1

    def train_test_split(self):
        return train_test_split(
            self.unique_img_ids, test_size=0.3, stratify=self.unique_img_ids["ships"]
        )

    def train_df_method(self):
        return pd.merge(data_exploring.mask, self.train_ids)

    def valid_df_method(self):
        return pd.merge(data_exploring.mask, self.valid_ids)
